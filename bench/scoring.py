"""Judge a model's reply for either arm, with the same outcomes and comparison as `vasuki eval`."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from vasuki import PlanError, validate_response
from vasuki.evals import compare, score

from .prompts import extract_code, extract_json

_CHILD = Path(__file__).with_name("_child.py")


class CodeError(Exception):
    pass


def run_code(code: str, tables: dict, timeout: float = 30) -> tuple[list[str], list[dict]]:
    """Run model-written code in a separate process, in an empty directory, with a time limit."""
    payload = {"code": code, "tables": {name: {"columns": t.columns, "rows": [list(r.values()) for r in t.rows]}
                                        for name, t in tables.items()}}
    with tempfile.TemporaryDirectory() as workdir:
        out = Path(workdir) / "result.json"
        try:
            done = subprocess.run(
                [sys.executable, "-I", str(_CHILD), str(out)],
                input=json.dumps(payload), capture_output=True, text=True, cwd=workdir, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise CodeError(f"timed out after {timeout:g}s") from None
        if done.returncode != 0 or not out.exists():
            lines = done.stderr.strip().splitlines()
            raise CodeError(lines[-1] if lines else f"exit code {done.returncode}")
        result = json.loads(out.read_text(encoding="utf-8"))
    return result["columns"], [dict(zip(result["columns"], row)) for row in result["rows"]]


def _judge_code(task: dict, text: str) -> tuple[str, str | None]:
    expect = task["expect"]
    code = extract_code(text)
    if code is None:
        reply = extract_json(text)
        try:
            validate_response(reply)
        except PlanError as e:
            return "invalid_response", f"no solve() code block and no status object ({e.code})"
        if reply["status"] == "plan":
            return "invalid_response", "replied with a plan, not code"
        if reply["status"] in expect["status"]:
            return "correct", None
        return "wrong_status", f"got {reply['status']!r}, expected {' or '.join(expect['status'])}"
    if expect["status"] != ["plan"]:
        return "wrong_status", f"wrote code, expected {' or '.join(expect['status'])}"
    try:
        columns, rows = run_code(code, task["data"])
    except CodeError as e:
        return "plan_error", str(e)
    return compare(expect, columns, rows), None


def judge(arm: str, task: dict, text: str | None) -> dict:
    """Score one reply. A code failure is reported as `plan_error`, the same bucket as a failing plan."""
    if arm != "code":
        return score(task, None if text is None else (extract_json(text) or text))
    outcome, detail = ("missing", None) if text is None else _judge_code(task, text)
    result = {"id": task["id"], "category": task["category"], "tags": task.get("tags", []), "outcome": outcome}
    if detail:
        result["detail"] = detail
    return result
