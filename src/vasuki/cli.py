from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import evals
from .errors import PlanError
from .executor import check_plan, run_plan
from .io import load_tables, write_table
from .schema import describe, validate_response

EXIT_OK, EXIT_ERROR, EXIT_NOT_A_PLAN = 0, 1, 3


def _load_plan(source: str):
    try:
        text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
        return json.loads(text)
    except (OSError, UnicodeDecodeError) as e:
        raise PlanError("invalid_input", f"cannot read plan: {e}") from None
    except json.JSONDecodeError as e:
        raise PlanError("invalid_plan", f"plan is not valid JSON: {e}") from None


def _planner(args):
    """The planner `ask` talks to, and the system prompt that suits it."""
    from . import planner

    if args.url:
        if not args.model:
            raise PlanError("invalid_input", "--url needs --model, the model's name on that server")
        model = planner.HTTPPlanner(args.url, args.model, api_key=os.environ.get("VASUKI_API_KEY"),
                                    reasoning=args.describe)    # a VIC model answers without reasoning
    else:
        model = planner.MLXPlanner(args.model)
    return model, planner.full_system_prompt() if args.describe else planner.VIC_SYSTEM


def _ask(args) -> tuple[dict, int]:
    from .planner import ask

    tables = load_tables(args.input)
    if not tables:
        raise PlanError("invalid_input", "give at least one input table with -i")
    model, system = _planner(args)
    answer = ask(model, args.request, tables, system=system, preview=args.preview, output=args.output)
    if not args.show_plan:
        answer.pop("plan", None)
    return answer, {"ok": EXIT_OK, "failed": EXIT_ERROR}.get(answer["status"], EXIT_NOT_A_PLAN)


def _evaluate(args) -> tuple[dict, int]:
    try:
        tasks = evals.load_tasks(args.tasks)
        if args.gold:
            predictions = {task["id"]: evals.gold_response(task) for task in tasks}
        elif args.predictions:
            predictions = evals.load_predictions(args.predictions)
        else:
            raise PlanError("invalid_input", "give a predictions file, or --gold to score the gold plans")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise PlanError("invalid_input", f"cannot load eval files: {e}") from None
    results = evals.evaluate(tasks, predictions)
    report = evals.summarize(results)
    if args.details:
        report["failures"] = [
            {k: r[k] for k in ("id", "outcome", "detail") if k in r} for r in results if r["outcome"] != "correct"
        ]
    return report, EXIT_OK


def _execute(args) -> tuple[dict | list, int]:
    if args.command == "schema":
        return describe(), EXIT_OK
    if args.command == "eval":
        return _evaluate(args)
    if args.command == "ask":
        return _ask(args)
    if args.command == "eval-inputs":
        try:
            return [evals.task_input(task) for task in evals.load_tasks(args.tasks)], EXIT_OK
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as e:
            raise PlanError("invalid_input", f"cannot load eval files: {e}") from None
    plan = _load_plan(args.plan)
    validate_response(plan)
    if plan["status"] != "plan":
        # A clarification or refusal is a valid answer; pass it through for the caller to handle.
        return plan, EXIT_NOT_A_PLAN
    tables = load_tables(args.input)
    if args.command == "check":
        columns = check_plan(plan, {name: table.columns for name, table in tables.items()})
        return {"status": "ok", "columns": columns}, EXIT_OK
    result = run_plan(plan, tables)
    report = result.report(args.preview)
    if args.output:
        write_table(args.output, result.columns, result.rows)
        report["output_path"] = args.output
    return report, EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vasuki", description="Run typed operation plans over CSV and JSON tables.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schema", help="print the plan format, ops, functions and units as JSON")
    for name, summary in (("check", "validate a plan against the inputs without running it"), ("run", "run a plan")):
        command = commands.add_parser(name, help=summary)
        command.add_argument("plan", help="plan JSON file, or - for stdin")
        command.add_argument(
            "-i", "--input", action="append", default=[], metavar="[NAME=]PATH",
            help="input table (.csv, .tsv, .json, .jsonl); the name defaults to the file stem",
        )
        if name == "run":
            command.add_argument("-o", "--output", help="write the result here")
            command.add_argument("--preview", type=int, default=5, help="rows to include in the report (default 5)")
    asking = commands.add_parser("ask", help="answer a request in words: a planner model writes the plan, then it runs")
    asking.add_argument("request", help="what you want, in plain words")
    asking.add_argument("-i", "--input", action="append", default=[], metavar="[NAME=]PATH",
                        help="input table (.csv, .tsv, .json, .jsonl); the name defaults to the file stem")
    asking.add_argument("-o", "--output", help="write the whole result here")
    asking.add_argument("--preview", type=int, default=20, help="rows to print (default 20)")
    asking.add_argument("--show-plan", action="store_true", help="include the plan the model wrote")
    asking.add_argument("--model", help="a VIC model: a local folder or a Hugging Face name (default: $VASUKI_MODEL, "
                                        "else the published preview). With --url, the model's name on that server")
    asking.add_argument("--url", help="use an OpenAI-style chat endpoint instead of a local MLX model, e.g. "
                                      "http://localhost:11434/v1; $VASUKI_API_KEY is sent if set")
    asking.add_argument("--describe", action="store_true",
                        help="send the full plan format with the request, for a model that was not trained on it")
    inputs = commands.add_parser("eval-inputs", help="print what a planner sees for each eval task, as JSONL")
    inputs.add_argument("tasks", help="eval set directory, e.g. evals/v1")
    scoring = commands.add_parser("eval", help="score planner responses against an eval set")
    scoring.add_argument("tasks", help="eval set directory, e.g. evals/v1")
    scoring.add_argument("predictions", nargs="?", help='JSONL of {"id": ..., "response": ...}')
    scoring.add_argument("--gold", action="store_true", help="score the gold plans instead of a predictions file")
    scoring.add_argument("--details", action="store_true", help="list every task that was not correct")
    args = parser.parse_args(argv)

    try:
        report, code = _execute(args)
    except PlanError as e:
        report, code = e.to_dict(), EXIT_ERROR
    if isinstance(report, list):
        for line in report:
            sys.stdout.write(json.dumps(line, ensure_ascii=False) + "\n")
    else:
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
