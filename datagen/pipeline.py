"""Build one training world end to end: invent tables, author examples, keep only the verified ones.

An example is kept only when something other than its author vouches for it:

- a plan, when a second model writing pandas from the request alone produces the same table;
- a clarification, when that second model also asks instead of answering;
- a refusal, by construction: the recipe asked for an action the operations cannot do.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from bench.prompts import CODE_SYSTEM, extract_code, extract_json
from bench.scoring import CodeError, run_code
from vasuki import PlanError, Table, run_plan, validate_response
from vasuki.evals import compare
from vasuki.request import planner_input, profile_table

from .prompts import AUTHOR_SYSTEM, REWRITE_STYLES, WORLD_SYSTEM, recipes, world_prompt
from .topics import BANNED_TABLES

_NAME = re.compile(r"[a-z][a-z0-9_]*")


class Rejected(Exception):
    """This world or example is unusable; the message says why."""


def build_world(spec: dict, text: str) -> dict:
    """Validate a model's invented tables. Returns the world record or raises Rejected."""
    reply = extract_json(text)
    tables = reply.get("tables") if isinstance(reply, dict) else None
    if not isinstance(tables, dict) or not tables:
        raise Rejected("no tables object")
    wanted = 2 if "two_tables" in spec["traits"] else 1
    if len(tables) != wanted:
        raise Rejected(f"{len(tables)} tables, wanted {wanted}")
    for name, rows in tables.items():
        if not _NAME.fullmatch(name) or name in BANNED_TABLES:
            raise Rejected(f"table name {name!r} is not allowed")
        try:
            table = Table.from_rows(name, rows)
        except PlanError as e:
            raise Rejected(e.message) from None
        if not 6 <= len(table.rows) <= 20 or not 4 <= len(table.columns) <= 10:
            raise Rejected(f"table {name!r} is {len(table.rows)} rows by {len(table.columns)} columns")
        if not all(_NAME.fullmatch(column) for column in table.columns):
            raise Rejected(f"table {name!r} has a column name that is not snake_case")
        for column in profile_table(table)["columns"]:
            if column["type"] in ("json", "mixed", "null"):
                raise Rejected(f"column {name}.{column['name']} is {column['type']}")
    if wanted == 2:
        first, second = (set(rows[0]) for rows in tables.values())
        if not first & second:
            raise Rejected("the two tables share no column")
    return {**spec, "tables": tables}


def simplify_plan(plan: dict, tables: dict[str, Table]) -> dict:
    """Drop every `input` and `as` the plan does not need, if the result stays the same."""
    steps = [dict(step) for step in plan["steps"]]
    previous = None
    for i, step in enumerate(steps):
        implied = previous if i else (next(iter(tables)) if len(tables) == 1 else None)
        if step.get("input") is not None and step.get("input") == implied:
            del step["input"]
        previous = step.get("as")
    used = {step["input"] for step in steps if "input" in step} | {s["right"] for s in steps if s["op"] == "join"}
    for step in steps:
        if "as" in step and step["as"] not in used:
            del step["as"]
    simpler = {"status": "plan", "steps": steps}
    try:
        before, after = run_plan(plan, tables), run_plan(simpler, tables)
    except PlanError:
        return plan
    same = before.columns == after.columns and before.rows == after.rows
    return simpler if same else plan


def _as_task(world: dict, request: str) -> dict:
    data = {name: Table.from_rows(name, rows) for name, rows in world["tables"].items()}
    return {"request": request, "data": data}


def verify_example(world: dict, recipe: dict, example: dict, verifier) -> dict:
    """Check one authored example. Returns the record to keep, or raises Rejected."""
    request, response = example.get("request"), example.get("response")
    if not isinstance(request, str) or len(request.strip()) < 8:
        raise Rejected("no request text")
    try:
        validate_response(response)
    except PlanError as e:
        raise Rejected(f"invalid response: {e.code}: {e.message}") from None
    wanted = {"plan": "plan", "clarify": "needs_clarification", "unsupported": "unsupported"}[recipe["kind"]]
    if response["status"] != wanted:
        raise Rejected(f"author answered {response['status']}, recipe wanted {wanted}")

    record = {"world": world["id"], "n": recipe["n"], "topic": recipe["topic"], "style": recipe["style"],
              "request": request.strip(), "response": response}
    if wanted == "unsupported":
        return {**record, "verified_by": "construction"}

    task = _as_task(world, request)
    shown = json.dumps(planner_input(request, task["data"]), separators=(",", ":"), ensure_ascii=False)
    if wanted == "plan":
        try:
            result = run_plan(response, task["data"])
        except PlanError as e:
            raise Rejected(f"plan fails: {e.code}: {e.message}") from None
        if not result.rows:
            raise Rejected("plan returns no rows")
        expect = {"columns": result.columns, "rows": result.rows,
                  "ordered": any(step["op"] == "sort" for step in response["steps"])}

    opinion = verifier.complete(CODE_SYSTEM, shown)
    second, record["verifier"] = opinion.text, opinion.model
    code = extract_code(second)
    if wanted == "needs_clarification":
        reply = extract_json(second) if code is None else None
        if not isinstance(reply, dict) or reply.get("status") != "needs_clarification":
            raise Rejected("second model answered instead of asking")
        return {**record, "verified_by": "second model also asked"}
    if code is None:
        raise Rejected("second model wrote no code")
    try:
        columns, rows = run_code(code, task["data"])
    except CodeError as e:
        raise Rejected(f"second model's code failed: {e}") from None
    outcome = compare(expect, columns, rows)
    if outcome != "correct":
        raise Rejected(f"second model's result differs ({outcome})")
    record["response"] = simplify_plan(response, task["data"])
    return {**record, "verified_by": "second model's code gave the same table"}


def author_examples(world: dict, batch: list[dict], author) -> list[dict]:
    """Ask the author model for one example per recipe. Returns whatever well-formed items came back."""
    tables = {name: Table.from_rows(name, rows) for name, rows in world["tables"].items()}
    shown = {
        "tables": {name: {"profile": profile_table(table), "rows": table.rows} for name, table in tables.items()},
        "recipes": [{k: recipe[k] for k in ("n", "kind", "about", "style")} for recipe in batch],
    }
    text = author.complete(AUTHOR_SYSTEM, json.dumps(shown, separators=(",", ":"), ensure_ascii=False)).text
    examples = extract_examples(text)
    if not examples:
        raise Rejected("author returned no examples")
    return examples


_EXAMPLE_START = re.compile(r'\{\s*"n"\s*:')


def extract_examples(text: str) -> list[dict]:
    """Every example object in an author's reply.

    Read one by one, so a single malformed example (an unescaped quote, say) costs that example and
    not the other nine.
    """
    decoder = json.JSONDecoder()
    found = []
    for match in _EXAMPLE_START.finditer(text):
        try:
            example = decoder.raw_decode(text, match.start())[0]
        except json.JSONDecodeError:
            continue
        if isinstance(example, dict) and "request" in example and "response" in example:
            found.append(example)
    return found


def make_world(seed: int, spec: dict, inventor, author, verifier, per_world: int = 10) -> tuple[dict, list, list]:
    """Run the whole pipeline for one world. Returns (world, kept examples, rejections)."""
    world = build_world(spec, inventor.complete(WORLD_SYSTEM, world_prompt(spec)).text)
    world["author"] = getattr(author, "model", None)
    batch = {recipe["n"]: recipe for recipe in recipes(seed, spec, per_world)}
    authored = [(batch[example.get("n")], example) for example in author_examples(world, list(batch.values()), author)
                if example.get("n") in batch]

    def check(pair):
        recipe, example = pair
        try:
            return recipe, verify_example(world, recipe, example, verifier), None
        except Rejected as e:
            return recipe, example, str(e)

    # Each check is one slow model call, so a world's examples are verified side by side.
    with ThreadPoolExecutor(max(1, min(len(authored), 4))) as pool:
        checked = list(pool.map(check, authored))
    kept, rejected = [], []
    for recipe, record, why in checked:
        if why is None:
            kept.append({"id": f"{world['id']}-{recipe['n']:02d}", **record})
        else:
            rejected.append({"world": world["id"], "n": recipe["n"], "topic": recipe["topic"], "why": why,
                             "request": record.get("request"), "response": record.get("response")})
    return world, kept, rejected


def rewrite_world(world: dict, examples: list[dict], rewriter, verifier, style: str = "intent") -> tuple[list, list]:
    """Rewrite a world's plan requests as short statements of intent, keeping those that still verify.

    The plan is unchanged and already trusted. The new wording is kept only if a second model, given
    that wording alone, still produces the plan's table.
    """
    tables = {name: Table.from_rows(name, rows) for name, rows in world["tables"].items()}
    plans = [e for e in examples if e["response"]["status"] == "plan"]
    if not plans:
        return [], []
    shown = {
        "tables": {name: profile_table(table) for name, table in tables.items()},
        "items": [{"n": i, "request": e["request"], "plan": e["response"]["steps"]} for i, e in enumerate(plans, 1)],
    }
    system, suffix = REWRITE_STYLES[style]
    text = rewriter.complete(system, json.dumps(shown, separators=(",", ":"), ensure_ascii=False)).text
    # Read item by item, like extract_examples, so one malformed rewrite does not cost the rest.
    decoder, wording = json.JSONDecoder(), {}
    for match in _EXAMPLE_START.finditer(text):
        try:
            item = decoder.raw_decode(text, match.start())[0]
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            wording[item.get("n")] = item.get("request")
    if not wording:
        raise Rejected("rewriter returned no rewrites")

    def check(pair):
        n, original = pair
        request = wording.get(n)
        if not isinstance(request, str) or len(request.strip()) < 8:
            return original, None, "no rewrite"
        request = request.strip()
        if request == original["request"]:
            return original, None, "left unchanged"
        try:
            result = run_plan(original["response"], tables)
            expect = {"columns": result.columns, "rows": result.rows,
                      "ordered": any(step["op"] == "sort" for step in original["response"]["steps"])}
            prompt = json.dumps(planner_input(request, tables), separators=(",", ":"), ensure_ascii=False)
            opinion = verifier.complete(CODE_SYSTEM, prompt)
            code = extract_code(opinion.text)
            if code is None:
                return original, request, "second model wrote no code"
            columns, rows = run_code(code, tables)
        except (PlanError, CodeError) as e:
            return original, request, f"second model's code failed: {e}"
        outcome = compare(expect, columns, rows)
        if outcome != "correct":
            return original, request, f"second model's result differs ({outcome})"
        return original, {**original, "id": original["id"] + suffix, "request": request, "rewritten_from": original["id"],
                          "verified_by": "second model's code gave the same table", "verifier": opinion.model}, None

    with ThreadPoolExecutor(max(1, min(len(plans), 4))) as pool:
        checked = list(pool.map(check, enumerate(plans, 1)))
    kept = [record for _, record, why in checked if why is None]
    rejected = [{"world": world["id"], "n": original["n"], "topic": original["topic"], "why": why,
                 "request": request if isinstance(request, str) else None, "response": original["response"]}
                for original, request, why in checked if why is not None]
    return kept, rejected
