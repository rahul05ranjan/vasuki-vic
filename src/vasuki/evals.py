"""Score planner responses against an evaluation set.

A task is judged on the result of running the response, never on the plan's
text: two different plans that produce the same table are both correct.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .errors import PlanError, is_number
from .executor import Table, run_plan
from .request import planner_input
from .schema import validate_response

OUTCOMES = (
    "correct",
    "values_only",       # right values, different column names
    "wrong_result",
    "plan_error",        # valid plan that failed against the data
    "invalid_response",  # not JSON, or not in the response format
    "wrong_status",      # e.g. a plan where a clarification was expected
    "missing",
)

_MAX_COLUMN_MAPPINGS = 5000


def load_tasks(directory: str | Path) -> list[dict]:
    """Load an eval set; each task gains `data`, its input tables by name."""
    directory = Path(directory)
    raw = json.loads((directory / "tables.json").read_text(encoding="utf-8"))
    tables = {name: Table.from_rows(name, rows) for name, rows in raw.items()}
    lines = (directory / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    tasks = [json.loads(line) for line in lines if line.strip()]
    for task in tasks:
        # A task may carry its own tables, for sets where every task has a different dataset.
        own = task.get("inline_tables")
        if own:
            task["data"] = {name: Table.from_rows(name, rows) for name, rows in own.items()}
        elif isinstance(task["tables"], dict):
            # {name the planner sees: key in tables.json}, for sets where different datasets reuse a table name.
            task["data"] = {name: Table.from_rows(name, raw[key]) for name, key in task["tables"].items()}
        else:
            task["data"] = {name: tables[name] for name in task["tables"]}
    return tasks


def load_predictions(path: str | Path) -> dict:
    """Read JSONL lines of `{"id": ..., "response": ...}`; a response may be JSON text."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {entry["id"]: entry.get("response") for entry in map(json.loads, filter(str.strip, lines))}


def task_input(task: dict) -> dict:
    """What the planner sees for a task. Never includes the gold plan or expected result."""
    return {"id": task["id"], **planner_input(task["request"], task["data"])}


def gold_response(task: dict) -> dict:
    if "gold" in task:
        return task["gold"]
    status = task["expect"]["status"][0]
    key = "question" if status == "needs_clarification" else "reason"
    return {"status": status, key: "gold"}


def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if is_number(a) and is_number(b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def _sort_key(value):
    if value is None:
        return (0, 0)
    if isinstance(value, bool):
        return (1, value)
    if is_number(value):
        return (2, round(float(value), 6))
    if isinstance(value, str):
        return (3, value)
    return (4, json.dumps(value, sort_keys=True))


def _rows_match(expected: list[tuple], got: list[tuple], ordered: bool) -> bool:
    if len(expected) != len(got):
        return False
    if not ordered:
        def key(row):
            return tuple(_sort_key(v) for v in row)
        expected, got = sorted(expected, key=key), sorted(got, key=key)
    return all(_same(a, b) for e, g in zip(expected, got) for a, b in zip(e, g))


def _match_ignoring_names(expected_columns, expected_rows, got_columns, got_rows, ordered) -> bool:
    """True if some pairing of columns makes the two tables equal."""
    def column(rows, name):
        return [(row[name],) for row in rows]

    candidates = [
        [g for g in got_columns if _rows_match(column(expected_rows, e), column(got_rows, g), ordered)]
        for e in expected_columns
    ]
    expected = [tuple(row[c] for c in expected_columns) for row in expected_rows]
    tried = 0

    def assign(i: int, chosen: list[str]) -> bool:
        nonlocal tried
        if i == len(expected_columns):
            tried += 1
            return _rows_match(expected, [tuple(row[c] for c in chosen) for row in got_rows], ordered)
        for name in candidates[i]:
            if name not in chosen and tried < _MAX_COLUMN_MAPPINGS:
                if assign(i + 1, [*chosen, name]):
                    return True
        return False

    return assign(0, [])


def compare(expect: dict, columns: list[str], rows: list[dict]) -> str:
    """Compare a result with a task's `expect`; column order never matters."""
    expected_columns, expected_rows, ordered = expect["columns"], expect["rows"], expect.get("ordered", False)
    if len(columns) != len(expected_columns):
        return "wrong_result"
    if set(columns) == set(expected_columns):
        as_tuples = lambda table: [tuple(row[c] for c in expected_columns) for row in table]
        return "correct" if _rows_match(as_tuples(expected_rows), as_tuples(rows), ordered) else "wrong_result"
    if _match_ignoring_names(expected_columns, expected_rows, columns, rows, ordered):
        return "values_only"
    return "wrong_result"


def _judge(task: dict, response) -> tuple[str, str | None]:
    if response is None:
        return "missing", None
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return "invalid_response", "response is not JSON"
    try:
        validate_response(response)
    except PlanError as e:
        return "invalid_response", f"{e.code}: {e.message}"
    expect = task["expect"]
    if response["status"] not in expect["status"]:
        return "wrong_status", f"got {response['status']!r}, expected {' or '.join(expect['status'])}"
    if response["status"] != "plan":
        return "correct", None
    try:
        result = run_plan(response, task["data"])
    except PlanError as e:
        return "plan_error", f"{e.code}: {e.message}"
    return compare(expect, result.columns, result.rows), None


def score(task: dict, response) -> dict:
    outcome, detail = _judge(task, response)
    result = {"id": task["id"], "category": task["category"], "tags": task.get("tags", []), "outcome": outcome}
    if detail:
        result["detail"] = detail
    return result


def _tally(results: list[dict]) -> dict:
    correct = sum(r["outcome"] == "correct" for r in results)
    return {"tasks": len(results), "correct": correct, "accuracy": round(correct / len(results), 4)}


def summarize(results: list[dict]) -> dict:
    by_category: dict[str, list] = {}
    by_tag: dict[str, list] = {}
    for result in results:
        by_category.setdefault(result["category"], []).append(result)
        for tag in result["tags"]:
            by_tag.setdefault(tag, []).append(result)
    outcomes = Counter(r["outcome"] for r in results)
    lenient = outcomes["correct"] + outcomes["values_only"]
    return {
        **_tally(results),
        "accuracy_ignoring_column_names": round(lenient / len(results), 4),
        "outcomes": {name: outcomes[name] for name in OUTCOMES if outcomes[name]},
        "by_category": {name: _tally(items) for name, items in by_category.items()},
        "by_tag": {name: _tally(items) for name, items in sorted(by_tag.items())},
    }


def evaluate(tasks: list[dict], predictions: dict) -> list[dict]:
    return [score(task, predictions.get(task["id"])) for task in tasks]
