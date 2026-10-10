"""An MCP server that lets any agent hand tabular work to VIC instead of doing it in its own context.

    vasuki-mcp                      # stdio server; needs the `mcp` extra, and `mlx` for the local model

The agent sends a request in words and file paths. The files are read here, a small local model writes a plan,
the executor runs it, and only the result's first rows go back. The agent never reads the data.

Settings, from the environment:
  VASUKI_MODEL     the VIC model: a local folder or a Hugging Face name (default: the published VIC-2B)
  VASUKI_URL       use an OpenAI-style chat endpoint instead of a local MLX model; VASUKI_MODEL then names the
                   model on that server and VASUKI_API_KEY, if set, is sent as the bearer token
  VASUKI_DESCRIBE  set to 1 to send the full plan format with each request, for a model not trained on it
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from .errors import PlanError
from .executor import run_plan as execute_plan
from .io import load_tables, write_table
from .planner import VIC_SYSTEM, HTTPPlanner, MLXPlanner, ask, full_system_prompt
from .request import profile_table
from .schema import describe, validate_response

_planner = None
_lock = threading.Lock()


def _get_planner():
    """The planner is loaded on first use and kept, so only the first request pays for loading the model."""
    global _planner
    with _lock:
        if _planner is None:
            url, model = os.environ.get("VASUKI_URL"), os.environ.get("VASUKI_MODEL")
            if url and not model:
                raise PlanError("planner_unavailable", "VASUKI_URL is set but VASUKI_MODEL is not")
            trained = os.environ.get("VASUKI_DESCRIBE") != "1"    # a VIC model answers without reasoning
            _planner = (HTTPPlanner(url, model, api_key=os.environ.get("VASUKI_API_KEY"), reasoning=not trained)
                        if url else MLXPlanner(model))
        return _planner


def _tables(files: list[str]):
    if not files:
        raise PlanError("invalid_input", "give at least one file")
    specs = []
    for spec in files:
        name, sep, path = spec.partition("=")
        path = str(Path(path if sep else spec).expanduser())
        specs.append(f"{name}={path}" if sep else path)
    return load_tables(specs)


def _guard(work) -> dict:
    try:
        return work()
    except PlanError as e:
        return e.to_dict()


def compute(request: str, files: list[str], output_path: str | None = None, preview_rows: int = 20) -> dict:
    """Answer a question about CSV, TSV, JSON or JSONL files without reading them yourself.

    Use this instead of opening a data file when the task is to filter, sort, add or change columns, total or
    average by group, join two files, remove duplicates, or convert units. Say what you want in plain words
    and what the new columns should be called, e.g. "total amount per region for orders that were not
    cancelled, as total_amount, highest first". A small local model writes a plan and a deterministic
    executor runs it on the full files; only the first rows of the result come back.

    files: paths, each `PATH` or `NAME=PATH`. A table is named after its file unless NAME is given.
    output_path: where to save the whole result (.csv, .tsv, .json or .jsonl) when it is needed later.

    Returns a dict whose `status` is:
      ok                   `columns`, `row_count`, the first `rows` (lists of values in column order), and the
                           `plan` that produced them
      needs_clarification  `question`: answer it and call again with the answer folded into the request
      unsupported          `reason`: the request is outside these operations; do it another way
      failed               the model's plan did not run; reword the request, or write a plan and use run_plan
      error                a file could not be read or the model could not be loaded; `message` says which
    Check the plan when the answer matters: it shows exactly which rows and columns were used.
    """
    def work():
        tables = _tables(files)
        system = full_system_prompt() if os.environ.get("VASUKI_DESCRIBE") == "1" else VIC_SYSTEM
        return ask(_get_planner(), request, tables, system=system, preview=preview_rows, output=output_path)
    return _guard(work)


def run_plan(steps: list[dict], files: list[str], output_path: str | None = None, preview_rows: int = 20) -> dict:
    """Run a Vasuki plan you wrote yourself on data files, without reading the files.

    Use this when `compute` got the plan wrong, or when you want exact control. `steps` is the list of plan
    steps; call `plan_format` once to learn the operations. Returns the same `ok` or `error` dict as `compute`.
    """
    def work():
        plan = {"status": "plan", "steps": steps}
        validate_response(plan)
        result = execute_plan(plan, _tables(files))
        answer = {"status": "ok", "columns": result.columns, "row_count": len(result.rows),
                  "rows": [[row[column] for column in result.columns] for row in result.rows[:preview_rows]]}
        if len(result.rows) > preview_rows:
            answer["truncated"] = (f"showing {preview_rows} of {len(result.rows)} rows; pass an output path for "
                                   "all of them")
        if output_path:
            write_table(output_path, result.columns, result.rows)
            answer["output_path"] = output_path
        return answer
    return _guard(work)


def describe_tables(files: list[str]) -> dict:
    """Look at data files cheaply: each table's row count, column names and types, and three sample rows.

    Use this before `compute` when you do not know what columns a file has.
    """
    return _guard(lambda: {"status": "ok",
                           "tables": {name: profile_table(table) for name, table in _tables(files).items()}})


def plan_format() -> dict:
    """The Vasuki plan format: every operation, expression function and unit. Needed only to write plans for
    `run_plan`; `compute` does not need it. About 2,000 tokens."""
    return describe()


TOOLS = (compute, run_plan, describe_tables, plan_format)


def main() -> None:
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        try:  # the SDK before 2.0 called it FastMCP
            from mcp.server.fastmcp import FastMCP as MCPServer
        except ImportError:
            raise SystemExit("the MCP SDK is not installed; install vasuki[mcp]") from None
    server = MCPServer("vasuki", instructions=(
        "Hand work on data files to these tools instead of reading the files: `compute` answers a request in "
        "words, `describe_tables` shows what a file holds, `run_plan` runs a plan you write."))
    for tool in TOOLS:
        server.tool()(tool)
    server.run()


if __name__ == "__main__":
    main()
