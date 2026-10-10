"""Does VIC help an agent that could also write a script? Thirteen real questions about this project's own
files, each answered two ways: the pandas line the agent would write, and one request to VIC.

    uv run python -m bench.agent_trial OUTPUT_FOLDER [MODEL]      # needs data/; MODEL defaults to models/VIC-k3-mlx-6bit

The requests below were written once, before VIC saw any of them, and are not tuned afterwards.
Development-set results only; the test set's per-task results are deliberately left out.
Token counts use the model's own tokenizer for both routes, as a common yardstick, not a bill.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

from bench.scoring import judge
from vasuki import load_table
from vasuki.evals import load_tasks
from vasuki.planner import MLXPlanner, ask

ROOT = Path(".")


def build_files(out: Path) -> None:
    """Flat files from real project data, the way an agent would export them."""
    tasks = {t["id"]: t for t in load_tasks("evals/v2")}
    rows = []
    for path in sorted((ROOT / "bench/results/v2").glob("*.jsonl")):
        arm, model = path.stem.split("__")
        if arm == "code":
            continue
        for record in [json.loads(line) for line in path.read_text().splitlines()][1:]:
            if "text" not in record:
                continue
            outcome = judge(arm, tasks[record["id"]], record["text"])["outcome"]
            usage = record.get("usage", {})
            rows.append({"task_id": record["id"], "category": tasks[record["id"]]["category"], "model": model,
                         "outcome": outcome, "correct": outcome == "correct", "seconds": record.get("seconds"),
                         "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")})
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)

    worlds = [json.loads(line) for line in (ROOT / "data/train_mix_k3/worlds.jsonl").read_text().splitlines()]
    pd.DataFrame([{"world": w["id"], "subject": w["topic"], "tables": len(w["tables"])} for w in worlds]) \
        .drop_duplicates("world").to_csv(out / "worlds.csv", index=False)
    examples = [json.loads(line) for line in (ROOT / "data/train_mix_k3/examples.jsonl").read_text().splitlines()]
    pd.DataFrame([{"example_id": e["id"], "world": e["world"], "topic": e["topic"], "style": e["style"],
                   "status": e["response"]["status"], "steps": len(e["response"].get("steps", [])) or None}
                  for e in examples]).to_csv(out / "examples.csv", index=False)


R, E, W = "results.csv", "examples.csv", "worlds.csv"
# (files, request to VIC, the pandas the agent would write instead, ordered?)
QUESTIONS = [
    ([R], "Share of correct answers per model, as accuracy, to 3 decimals, best first. Columns model, accuracy.",
     "r.groupby('model', as_index=False)['correct'].mean().rename(columns={'correct':'accuracy'}).round({'accuracy':3}).sort_values('accuracy', ascending=False)", True),
    ([R], "For the model VIC-k3-mlx-6bit, how many tasks were correct in each category? Columns category, correct_tasks.",
     "r[(r.model=='VIC-k3-mlx-6bit') & r.correct].groupby('category', as_index=False).size().rename(columns={'size':'correct_tasks'})", False),
    ([R], "Average seconds per answer for each model, as avg_seconds, to 2 decimals, slowest first. Columns model, avg_seconds.",
     "r.groupby('model', as_index=False)['seconds'].mean().rename(columns={'seconds':'avg_seconds'}).round({'avg_seconds':2}).sort_values('avg_seconds', ascending=False)", True),
    ([R], "Total output tokens per model, as total_output, highest first. Columns model, total_output.",
     "r.groupby('model', as_index=False)['output_tokens'].sum().rename(columns={'output_tokens':'total_output'}).sort_values('total_output', ascending=False)", True),
    ([R], "How many answers of each outcome did the model VIC-k2-mlx-6bit give? Columns outcome, answers.",
     "r[r.model=='VIC-k2-mlx-6bit'].groupby('outcome', as_index=False).size().rename(columns={'size':'answers'})", False),
    ([R], "The tasks that VIC-k3-mlx-6bit did not get correct: task_id, category and outcome, sorted by task_id.",
     "r[(r.model=='VIC-k3-mlx-6bit') & ~r.correct][['task_id','category','outcome']].sort_values('task_id')", True),
    ([R], "The 5 slowest answers overall: model, task_id and seconds, slowest first.",
     "r.sort_values('seconds', ascending=False).head(5)[['model','task_id','seconds']]", True),
    ([R], "Which tasks did no model at all answer correctly? Return task_id and category.",
     "(lambda g: g[g.n==0][['task_id','category']])(r.assign(c=r.correct.astype(int)).groupby(['task_id','category'], as_index=False).agg(n=('c','sum')))", False),
    ([R], "Tasks that the model VIC-k2-mlx-6bit got wrong and the model VIC-k3-mlx-6bit got right. Return task_id only.",
     "pd.merge(r[(r.model=='VIC-k2-mlx-6bit') & ~r.correct][['task_id']], r[(r.model=='VIC-k3-mlx-6bit') & r.correct][['task_id']], on='task_id')", False),
    ([E], "Number of examples for each topic and style. Columns topic, style, examples.",
     "e.groupby(['topic','style'], as_index=False).size().rename(columns={'size':'examples'})", False),
    ([E], "What share of the examples in each style are plans? Columns style, plan_share, a fraction to 2 decimals.",
     "e.assign(p=(e.status=='plan').astype(int)).groupby('style', as_index=False)['p'].mean().rename(columns={'p':'plan_share'}).round({'plan_share':2})", False),
    ([E], "Average number of steps of the plan examples per topic, as avg_steps to 2 decimals, longest first. Columns topic, avg_steps.",
     "e[e.status=='plan'].groupby('topic', as_index=False)['steps'].mean().rename(columns={'steps':'avg_steps'}).round({'avg_steps':2}).sort_values('avg_steps', ascending=False)", True),
    ([E, W], "The 5 subjects with the most examples. Columns subject, examples, most first.",
     "e.merge(w, on='world').groupby('subject', as_index=False).size().rename(columns={'size':'examples'}).sort_values('examples', ascending=False).head(5)", True),
]


def same(got_cols, got_rows, want: pd.DataFrame, ordered: bool) -> str:
    want_rows = json.loads(want.to_json(orient="records"))
    norm = lambda rows, cols: [tuple(round(float(row[c]), 6) if isinstance(row[c], (int, float)) and not isinstance(row[c], bool)
                                     else row[c] for c in cols) for row in rows]
    if len(got_rows) != len(want_rows):
        return f"WRONG ({len(got_rows)} rows, expected {len(want_rows)})"
    if set(got_cols) == set(want.columns):
        a, b = norm(got_rows, list(want.columns)), norm(want_rows, list(want.columns))
        if a == b or (not ordered and sorted(map(str, a)) == sorted(map(str, b))):
            return "correct"
        if sorted(map(str, a)) == sorted(map(str, b)):
            return "WRONG (right rows, wrong order)"
        return "WRONG (values differ)"
    if len(got_cols) == len(want.columns):
        a = sorted(str(sorted(map(str, norm([row], got_cols)[0]))) for row in got_rows)
        b = sorted(str(sorted(map(str, norm([row], list(want.columns))[0]))) for row in want_rows)
        if a == b:
            return "values right, column names differ"
    return f"WRONG (columns {got_cols})"


def main() -> None:
    OUT = Path(sys.argv[1])
    build_files(OUT)
    planner = MLXPlanner(sys.argv[2] if len(sys.argv) > 2 else "models/VIC-k3-mlx-6bit")
    count = lambda text: len(planner._tokenizer.encode(text))   # one tokenizer for both sides, as a common yardstick
    r, e, w = (pd.read_csv(OUT / name) for name in (R, E, W))
    file_tokens = {name: count((OUT / name).read_text()) for name in (R, E, W)}
    print("files:", {name: f"{len(pd.read_csv(OUT / name))} rows, ~{file_tokens[name]:,} tokens" for name in (R, E, W)})

    report, totals = [], {"vic_out": 0, "vic_in": 0, "script_out": 0, "script_in": 0, "correct": 0}
    for n, (files, request, code, ordered) in enumerate(QUESTIONS, 1):
        want = eval(code)
        script = ("python3 -c \"import pandas as pd; " + "; ".join(f"{name[0]}=pd.read_csv('{name}')" for name in files)
                  + f"; print(({code}).to_string(index=False))\"")
        script_out, script_in = count(script), count(want.head(20).to_string(index=False))
        tables = {name.removesuffix(".csv"): load_table(OUT / name) for name in files}
        start = time.time()
        answer = ask(planner, request, tables, preview=20)
        seconds = time.time() - start
        call = json.dumps({"request": request, "files": files})
        vic_out, vic_in = count(call), count(json.dumps(answer, separators=(",", ":")))
        named = lambda a: [dict(zip(a["columns"], row)) for row in a["rows"]]
        verdict = same(answer["columns"], named(answer), want.head(20), ordered) \
            if answer["status"] == "ok" and len(want) <= 20 else None
        if answer["status"] == "ok" and len(want) > 20:   # compare on the full result, not the preview
            full = ask(planner, request, tables, preview=10 ** 6)
            verdict = same(full["columns"], named(full), want, ordered)
        if answer["status"] != "ok":
            verdict = answer["status"].upper() + ": " + str(answer.get("question") or answer.get("reason") or answer.get("error"))[:110]
        totals["correct"] += verdict == "correct"
        for key, value in (("vic_out", vic_out), ("vic_in", vic_in), ("script_out", script_out), ("script_in", script_in)):
            totals[key] += value
        report.append({"n": n, "request": request, "verdict": verdict, "seconds": round(seconds, 2),
                       "vic_tokens": [vic_out, vic_in], "script_tokens": [script_out, script_in],
                       "plan": answer.get("plan"), "want_head": want.head(3).to_dict("records")})
        print(f"Q{n:<2} {verdict:<40} {seconds:4.1f}s  VIC write {vic_out:>3} + read {vic_in:>4}   script write {script_out:>3} + read {script_in:>4}   | {request[:60]}")

    print("\ntotals:", totals, "| questions:", len(QUESTIONS))
    (OUT / "agent_test_report.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
