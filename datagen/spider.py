"""Training examples from questions real people wrote: Spider's train split.

    uv run python -m datagen.spider generate                      # first pass, writes data/train_spider
    uv run python -m datagen.spider generate --author @cf/zai-org/glm-5.3    # try the rejected ones again
    uv run python -m datagen.run --out data/train_spider export   # then export like any other folder

Each example is a Spider question, the one or two tables its SQL reads, and a Vasuki plan. An author model
writes the plan from the question, the table profiles and the SQL. The plan is kept only when running it
gives the result the SQL gives, ignoring column names, so no second model is needed to check it.

Questions in evals/real_dev are left out, and so is any other question with the same SQL as one of them
(Spider words most queries twice); evals/real1 comes from other databases altogether. `prune` moves
examples made before that rule out of a folder. Spider is by
Yu et al., 2018, under CC BY-SA 4.0, and this data is derived from it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.prompts import extract_json
from bench.providers import make_provider
from bench.run import load_env_file
from vasuki import PlanError, Table, describe, run_plan, validate_response
from vasuki.evals import compare
from vasuki.request import planner_input

from .pipeline import simplify_plan

ROOT = Path(__file__).resolve().parent.parent

AUTHOR_SYSTEM = f"""\
You translate SQL into Vasuki plans. You are given a question, a profile of the tables it is about, and \
the SQL that answers it. Write the Vasuki plan that returns exactly what the SQL returns: the same rows, \
and the same number of columns with the same meaning. The names of the result's columns are up to you.

Reply with one JSON object and nothing else: {{"status": "plan", "steps": [...]}}

Rules:
- Table and column names are case-sensitive here. Write them exactly as the profile does, whatever case \
the SQL uses. Write text values exactly as the SQL does.
- Return only what the SQL selects: end with a `select` of those columns unless the last step already \
leaves exactly them.
- A join whose key has a different name in each table uses `left_on` and `right_on`. A join whose key has \
the same name uses `on`. An SQL join without the word LEFT is `"how": "inner"`, the default.
- SELECT DISTINCT is `deduplicate`. GROUP BY with HAVING is `aggregate`, then `filter`. ORDER BY with LIMIT \
is `sort`, then `limit`. count(*) is the metric `{{"as": "...", "fn": "count"}}`. count(DISTINCT x) is \
`count_distinct`. LIKE '%x%' is `contains(...)`, LIKE 'x%' is `startswith(...)`.
- Ordering by an aggregate that is not selected: compute it, sort by it, then `select` it away.
- With two tables the first step must name its `input`. After that give a step `input` only when it does \
not read the previous step's result, and `as` only when a later step uses the name.
- Do exactly what the SQL does, even where the question is loose or the SQL looks odd.

The plan format:
{json.dumps(describe(), separators=(",", ":"))}"""


def load_builder():
    """evals/from_spider.py holds the rules for which questions are usable; reuse them unchanged."""
    spec = importlib.util.spec_from_file_location("from_spider", ROOT / "evals" / "from_spider.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidates(limit: int | None) -> tuple[list[dict], dict, Counter]:
    """Every usable train-split question as (task, world id), the worlds, and why the others were left out."""
    builder = load_builder()
    read = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    held = {item["question"] for item in read(builder.SPIDER / "validation.jsonl")}
    dev = read(ROOT / "evals" / "real_dev" / "tasks.jsonl")
    held |= {task["request"] for task in dev}
    held_sql = {builder.sql_key(task["tags"][0], task["sql"]) for task in dev}
    cache, tasks, worlds, reasons = {}, [], {}, Counter()
    for n, item in enumerate(read(builder.SPIDER / "train.jsonl"), start=1):
        if limit and len(tasks) >= limit:
            break
        if item["question"] in held:
            reasons["held out for an eval set"] += 1
            continue
        if builder.sql_key(item["db_id"], item["query"]) in held_sql:
            reasons["same SQL as an eval question"] += 1
            continue
        built = builder.build_task(n, "s", item, cache)
        if isinstance(built, str):
            reasons[built] += 1
            continue
        task, tables = built
        names = sorted(task["tables"])
        world = "s" + hashlib.md5((item["db_id"] + "|" + "|".join(names)).encode()).hexdigest()[:9]
        worlds.setdefault(world, {"id": world, "topic": f"spider: {item['db_id']}", "traits": [],
                                  "tables": {name: tables[key] for name, key in task["tables"].items()}})
        tasks.append({**task, "world": world, "n": n})
    return tasks, worlds, reasons


def check(reply, task: dict, tables: dict) -> tuple[dict | None, str]:
    """(the verified plan, "") or (None, why not)."""
    try:
        if not isinstance(reply, dict):
            raise PlanError("invalid_response", "no JSON object in the reply")
        validate_response(reply)
        result = run_plan(reply, tables)
    except PlanError as e:
        return None, f"plan does not run: {e.code}"
    if compare(task["expect"], result.columns, result.rows) not in ("correct", "values_only"):
        return None, "plan's result differs from the SQL's"
    return simplify_plan({"status": "plan", "steps": reply["steps"]}, tables), ""


def author_example(provider, task: dict, world: dict, attempts: int = 2) -> tuple[dict | None, str]:
    """(the example, "") when an authored plan reproduces the SQL's result, else (None, why the last try failed)."""
    tables = {name: Table.from_rows(name, rows) for name, rows in world["tables"].items()}
    shown = {"question": task["request"], "tables": planner_input(task["request"], tables)["tables"], "sql": task["sql"]}
    why = ""
    for _ in range(attempts):
        try:
            reply = extract_json(provider.complete(AUTHOR_SYSTEM, json.dumps(shown, ensure_ascii=False)).text or "")
        except Exception as e:      # one failed call must not end a long run; the question is retried next time
            why = f"call failed: {type(e).__name__}"
            continue
        candidates = [reply]
        steps = reply.get("steps") if isinstance(reply, dict) else None
        if len(tables) == 2 and isinstance(steps, list) and steps and isinstance(steps[0], dict) and "input" not in steps[0]:
            # The commonest slip: a two-table plan whose first step does not say which table it reads.
            # Either table is tried; the comparison with the SQL's result decides.
            candidates = [{**reply, "steps": [{**steps[0], "input": name}, *steps[1:]]} for name in tables]
        for candidate in candidates:
            plan, why = check(candidate, task, tables)
            if plan:
                return {"id": task["id"], "world": task["world"], "n": task["n"], "topic": task["category"],
                        "style": "spider", "request": task["request"], "response": plan,
                        "verified_by": "plan result equals the result of Spider's SQL", "author": provider.model}, ""
    return None, why


def generate(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks, worlds, left_out = candidates(args.limit)
    (out / "worlds.jsonl").write_text("".join(json.dumps(w, ensure_ascii=False) + "\n" for w in worlds.values()),
                                      encoding="utf-8")
    examples_path, rejects_path = out / "examples.jsonl", out / "rejects.jsonl"
    done = {json.loads(line)["id"] for line in examples_path.read_text(encoding="utf-8").splitlines()} \
        if examples_path.exists() else set()
    pending = [task for task in tasks if task["id"] not in done]
    provider = make_provider("cloudflare", args.author, None, False)
    provider.retries, provider.timeout, provider.extra = 3, 90, {"reasoning_effort": args.effort}
    lock, kept, why = threading.Lock(), 0, Counter()

    def work(task):
        nonlocal kept
        example, reason = author_example(provider, task, worlds[task["world"]])
        with lock:
            if example:
                kept += 1
                with examples_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            else:
                why[reason] += 1
                with rejects_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": task["id"], "author": args.author, "why": reason}) + "\n")
            finished = kept + sum(why.values())
            if finished % 100 == 0:
                print(f"  {finished}/{len(pending)} tried, {kept} kept", flush=True)

    print(f"{len(tasks)} usable questions over {len(worlds)} table sets; {len(done)} already kept, "
          f"{len(pending)} to try with {args.author}", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, pending))
    return {"usable_questions": len(tasks), "left_out": dict(left_out), "kept_before": len(done), "kept_now": kept,
            "kept_total": len(done) + kept, "rejected_now": dict(why), "author": args.author}


def prune(args) -> dict:
    """Move examples whose question is no longer usable (held out since they were made) to held_out.jsonl."""
    out = Path(args.out)
    usable = {task["id"] for task in candidates(None)[0]}
    lines = (out / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    keep = [line for line in lines if json.loads(line)["id"] in usable]
    drop = [line for line in lines if json.loads(line)["id"] not in usable]
    if drop:
        with (out / "held_out.jsonl").open("a", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in drop))
        (out / "examples.jsonl").write_text("".join(line + "\n" for line in keep), encoding="utf-8")
    return {"examples_before": len(lines), "moved_to_held_out": len(drop), "examples_now": len(keep)}


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    parser = argparse.ArgumentParser(prog="datagen.spider", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("generate", help="author and verify a plan for every usable question not yet kept")
    making.add_argument("--out", default="data/train_spider")
    making.add_argument("--author", default="@cf/openai/gpt-oss-120b")
    making.add_argument("--effort", default="low", help="reasoning effort for the author (default low)")
    making.add_argument("--workers", type=int, default=12)
    making.add_argument("--limit", type=int, help="stop after this many usable questions; for a trial run")
    pruning = commands.add_parser("prune", help="move examples that are held out now to held_out.jsonl")
    pruning.add_argument("--out", default="data/train_spider")
    args = parser.parse_args(argv)
    print(json.dumps((prune if args.command == "prune" else generate)(args), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
