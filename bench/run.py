"""Benchmark a model on the eval set, as a planner (`plan` arm) or writing pandas itself (`code` arm).

    uv run python -m bench.run run --arm plan --provider anthropic --model claude-opus-5-5
    uv run python -m bench.run run --arm code --provider anthropic --model claude-opus-5-5
    uv run python -m bench.run delegate --provider anthropic --model claude-opus-5-5
    uv run python -m bench.run report

A run appends to bench/results/<eval set>/<arm>__<model>.jsonl and resumes from it, so an interrupted run never
pays twice for a task.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vasuki import run_plan
from vasuki.evals import gold_response, load_tasks, summarize

from .prompts import SYSTEM, user_message
from .providers import PRICES, FatalProviderError, make_provider
from .scoring import judge

ARMS = ("plan", "code", "vic")
# Without a price list, weigh output tokens 5x input, the ratio across Anthropic's current models.
RELATIVE_WEIGHTS = (1.0, 5.0, 1.0, 1.0)


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model)


def _results_path(out: Path, arm: str, model: str) -> Path:
    return out / f"{arm}__{_slug(model)}.jsonl"


def select_tasks(tasks: list[dict], arm: str, per_category: int | None = None) -> list[dict]:
    """The tasks an arm is run on. Code can do most `unsupported` requests, so that label only fits plans."""
    if arm == "code":
        tasks = [task for task in tasks if task["category"] != "unsupported"]
    if per_category:
        seen: Counter = Counter()
        kept = []
        for task in tasks:
            seen[task["category"]] += 1
            if seen[task["category"]] <= per_category:
                kept.append(task)
        tasks = kept
    return tasks


def load_results(path: Path) -> tuple[dict, dict]:
    """Return (meta, records by task id). A later line for the same id replaces an earlier one."""
    meta, records = {}, {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if "meta" in entry:
                meta = entry["meta"]
            else:
                records[entry["id"]] = entry
    return meta, records


def run(args) -> dict:
    tasks = select_tasks(load_tasks(args.tasks), args.arm, args.per_category)
    provider = make_provider(args.provider, args.model, None if args.effort == "none" else args.effort,
                             not args.no_fallbacks, adapter=args.adapter)
    if args.arm == "vic" and args.provider == "ollama":
        provider.think = False      # a VIC model was trained to answer with no reasoning
    path = _results_path(Path(args.out), args.arm, args.label or args.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    system = SYSTEM[args.arm]
    _, done = load_results(path)
    pending = [task for task in tasks if "text" not in done.get(task["id"], {})]

    lock, stop = threading.Lock(), threading.Event()
    finished = 0

    def append(entry: dict) -> None:
        with lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def work(task: dict) -> None:
        nonlocal finished
        if stop.is_set():
            return
        start = time.monotonic()
        try:
            record = provider.complete(system, user_message(task)).record()
        except FatalProviderError:
            stop.set()
            raise
        except Exception as e:  # one failed request must not end a long run; it is retried on resume
            record = {"error": f"{type(e).__name__}: {e}"}
        append({"id": task["id"], "seconds": round(time.monotonic() - start, 2), **record})
        with lock:
            finished += 1
            if finished % 10 == 0 or finished == len(pending):
                print(f"  {finished}/{len(pending)} done", file=sys.stderr, flush=True)

    if not path.exists():
        counter = getattr(provider, "count_tokens", None)
        append({"meta": {"arm": args.arm, "provider": args.provider, "model": args.label or args.model,
                         "base_model": args.model, "adapter": args.adapter, "effort": args.effort,
                         "system_tokens": counter(system) if counter else None}})
    print(f"{args.arm} arm, {args.model}: {len(pending)} to run, {len(tasks) - len(pending)} already done",
          file=sys.stderr, flush=True)
    if pending:
        # The first call alone: it surfaces a bad key or model before fanning out, and warms the prompt cache.
        work(pending[0])
        with ThreadPoolExecutor(provider.workers) as pool:
            list(pool.map(work, pending[1:]))
    return summarize_file(path, load_tasks(args.tasks), delegate=None, details=False)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


def _cost(usage: dict, weights: tuple) -> float:
    price_in, price_out, price_read, price_write = weights
    return (usage["input_tokens"] * price_in + usage["output_tokens"] * price_out
            + usage["cache_read_tokens"] * price_read + usage["cache_write_tokens"] * price_write)


def summarize_file(path: Path, tasks: list[dict], delegate: dict | None, details: bool) -> dict:
    meta, records = load_results(path)
    arm, model = meta.get("arm"), meta.get("model")
    answered = [task for task in select_tasks(tasks, arm) if "text" in records.get(task["id"], {})]
    if not answered:
        return {"file": path.name, **meta, "answered": 0}
    results = [judge(arm, task, records[task["id"]]["text"]) for task in answered]
    used = [records[task["id"]] for task in answered]
    total_input = [sum(r["usage"][k] for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens")) for r in used]
    prices = PRICES.get(model)

    report = {
        "file": path.name,
        **meta,
        "answered": len(answered),
        "errors_awaiting_retry": sum("error" in r for r in records.values()),
        **summarize(results),
        "tokens_per_task": {
            "input_including_system_prompt": _mean(total_input),
            "output": _mean([r["usage"]["output_tokens"] for r in used]),
        },
        "seconds_per_task": _mean([r["seconds"] for r in used]),
    }
    if prices:
        report["billed_usd_total"] = round(sum(_cost(r["usage"], prices) for r in used) / 1e6, 4)
    served = lambda name: name.rsplit("/", 1)[-1]
    other_models = Counter(r["model"] for r in used if not served(r["model"]).startswith(served(model)))
    if other_models:
        report["answered_by_another_model"] = dict(other_models)
    stops = Counter(r["stop_reason"] for r in used if r["stop_reason"] in ("refusal", "max_tokens", "length"))
    if stops:
        report["abnormal_stops"] = dict(stops)

    if arm == "code" and delegate:
        report["versus_delegating"] = _versus_delegating(answered, used, total_input, meta, prices, delegate)
    if details:
        report["failures"] = [{k: r[k] for k in ("id", "outcome", "detail") if k in r}
                              for r in results if r["outcome"] != "correct"]
    return report


def _versus_delegating(answered, used, total_input, meta, prices, delegate) -> dict:
    """What the calling model spends doing a task itself, against handing it to a local planner.

    Alone, it reads the table profile, writes code, and reads the result. Delegating, it writes a short
    call and reads the result. A local planner that is right a fraction p of the time, with the caller
    redoing the rest itself, costs delegated + (1 - p) * alone, so it saves money only when p exceeds
    delegated / alone.
    """
    system_tokens = meta.get("system_tokens") or 0
    weight_in, weight_out = (prices or RELATIVE_WEIGHTS)[:2]
    alone, delegated = [], []
    for task, record, task_input_tokens in zip(answered, used, total_input):
        sizes = delegate["tasks"].get(task["id"])
        if not sizes:
            continue
        alone.append((task_input_tokens - system_tokens + sizes["result"]) * weight_in
                     + record["usage"]["output_tokens"] * weight_out)
        delegated.append(sizes["result"] * weight_in + sizes["call"] * weight_out)
    if not alone or not sum(alone):
        return {}
    unit = "usd_per_1000_tasks" if prices else "relative_units_per_task_output_weighted_5x"
    scale = 1e-3 if prices else 1.0
    return {
        "tasks_compared": len(alone),
        "token_counts_from": delegate["counter"],
        "system_prompt_excluded": bool(system_tokens),
        "unit": unit,
        "caller_alone": round(sum(alone) / len(alone) * scale, 3),
        "caller_delegating": round(sum(delegated) / len(delegated) * scale, 3),
        "break_even_planner_accuracy": round(sum(delegated) / sum(alone), 3),
    }


def delegate(args) -> dict:
    """Measure the caller's side of a delegation for every task: the call it writes and the result it reads."""
    tasks = load_tasks(args.tasks)
    if args.provider == "anthropic":
        provider = make_provider("anthropic", args.model, None, False)
        count, counter = provider.count_tokens, args.model
    else:
        count, counter = (lambda text: math.ceil(len(text) / 3.5)), "estimate: characters / 3.5"

    def measure(task: dict) -> tuple[str, dict]:
        call = {"request": task["request"], "inputs": {name: f"{name}.csv" for name in task["tables"]}}
        reply = gold_response(task)
        result = run_plan(reply, task["data"]).report() if reply["status"] == "plan" else reply
        dumps = lambda value: json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        return task["id"], {"call": count(dumps(call)), "result": count(dumps(result))}

    with ThreadPoolExecutor(4) as pool:
        sizes = dict(pool.map(measure, tasks))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "delegate.json").write_text(json.dumps({"counter": counter, "tasks": sizes}, indent=1), encoding="utf-8")
    return {
        "counter": counter,
        "tasks": len(sizes),
        "call_tokens_per_task": _mean([s["call"] for s in sizes.values()]),
        "result_tokens_per_task": _mean([s["result"] for s in sizes.values()]),
    }


def report(args) -> dict:
    out, tasks = Path(args.out), load_tasks(args.tasks)
    delegate_path = out / "delegate.json"
    sizes = json.loads(delegate_path.read_text(encoding="utf-8")) if delegate_path.exists() else None
    files = sorted(out.glob("*__*.jsonl"))
    return {"runs": [summarize_file(path, tasks, sizes, args.details) for path in files]}


def load_env_file(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines into the environment so credentials can live in a local file. Existing values win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    parser = argparse.ArgumentParser(prog="bench.run", description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", default="evals/v2", help="eval set directory (default evals/v2)")
    parser.add_argument("--out", help="results directory (default bench/results/<eval set name>)")
    commands = parser.add_subparsers(dest="command", required=True)

    running = commands.add_parser("run", help="ask a model every task in one arm")
    running.add_argument("--arm", choices=ARMS, required=True)
    running.add_argument("--provider", choices=("anthropic", "cloudflare", "ollama", "mlx"), required=True)
    running.add_argument("--adapter", help="mlx only: directory of a LoRA adapter to load on top of the model")
    running.add_argument("--label", help="name for this run's results file, e.g. VIC-2B-pilot (default: the model)")
    running.add_argument("--model", required=True)
    running.add_argument("--effort", default="medium", help="Anthropic effort level, or 'none' to omit (default medium)")
    running.add_argument("--no-fallbacks", action="store_true", help="do not let a declined request fall back to another model")
    running.add_argument("--per-category", type=int, help="run only the first N tasks of each category")

    sizing = commands.add_parser("delegate", help="measure the caller's tokens when a task is delegated")
    sizing.add_argument("--provider", choices=("anthropic", "estimate"), default="estimate")
    sizing.add_argument("--model", default="claude-opus-5-5")

    reporting = commands.add_parser("report", help="score every results file")
    reporting.add_argument("--details", action="store_true", help="list every task that was not correct")

    args = parser.parse_args(argv)
    # Answers to one eval set cannot be scored against another, so each set gets its own folder.
    args.out = args.out or f"bench/results/{Path(args.tasks).name}"
    try:
        result = {"run": run, "delegate": delegate, "report": report}[args.command](args)
    except FatalProviderError as e:
        print(json.dumps({"status": "error", "message": str(e)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
