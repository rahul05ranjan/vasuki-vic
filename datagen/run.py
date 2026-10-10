"""Generate verified training data for VIC.

    uv run python -m datagen.run generate --worlds 900 --target 5000   # stop once 5,000 examples are kept
    uv run python -m datagen.run stats                     # yield, mix, and why examples were rejected
    uv run python -m datagen.run export                    # write MLX-LM chat files, split by world

Everything lands in data/train_v1/. A run resumes: finished worlds are skipped and failed ones retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.providers import PRICES, FatalProviderError, make_provider
from bench.run import load_env_file
from vasuki import PlanError, Table, run_plan
from vasuki.request import VIC_SYSTEM, planner_input

from .pipeline import Rejected, make_world, rewrite_world
from .prompts import world_spec

EVAL_SETS = ("evals/v1", "evals/v2", "evals/test1", "evals/real1", "evals/real_dev", "evals/real2")


class Metered:
    """Wraps a provider to total its token use across threads."""

    def __init__(self, provider, model: str):
        self.provider, self.model = provider, model
        self.calls = self.input = self.output = 0
        self._lock = threading.Lock()

    def complete(self, system: str, user: str):
        done = self.provider.complete(system, user)
        with self._lock:
            self.calls += 1
            self.input += done.input_tokens + done.cache_read_tokens + done.cache_write_tokens
            self.output += done.output_tokens
        return done

    def usage(self) -> dict:
        spent = {"model": self.model, "calls": self.calls, "input_tokens": self.input, "output_tokens": self.output}
        if getattr(self.provider, "retried", 0):
            spent["retried_attempts"] = self.provider.retried
        if self.model in PRICES:
            price_in, price_out = PRICES[self.model][:2]
            spent["usd_uncached"] = round((self.input * price_in + self.output * price_out) / 1e6, 4)
        return spent


class Rotation:
    """Hands each call to the next of several providers, to spread load over their separate rate limits."""

    def __init__(self, providers: list):
        self.providers, self._next, self._lock = providers, 0, threading.Lock()

    def complete(self, system: str, user: str):
        with self._lock:
            provider = self.providers[self._next % len(self.providers)]
            self._next += 1
        return provider.complete(system, user)


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_requests() -> set[str]:
    """Every request in the eval sets, normalised, so no training example can repeat one."""
    seen = set()
    for directory in EVAL_SETS:
        for task in _read(Path(directory) / "tasks.jsonl"):
            seen.add(" ".join(task["request"].lower().split()))
    return seen


def generate(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    def metered(model: str, long_replies: bool) -> Metered:
        provider = make_provider(args.provider, model, None, False)
        if hasattr(provider, "retries"):
            # A hung call would hold a worker for many minutes. Give up quickly instead: an abandoned
            # world costs little and is retried on the next run.
            provider.retries, provider.timeout = 3, (150 if long_replies else 60)
        if long_replies and hasattr(provider, "max_tokens"):
            # Inventors and authors write long replies after long reasoning; a reply cut off mid-JSON
            # loses the whole world.
            provider.max_tokens = max(provider.max_tokens, 24000)
        if args.effort != "default" and hasattr(provider, "extra"):
            # Measured on 2026-10-07: low effort cut the inventor from 140s to 22s and the GLM author
            # from 180s to 79s. Verification catches what the lighter reasoning gets wrong.
            provider.extra = {"reasoning_effort": args.effort}
        return Metered(provider, model)

    inventor = metered(args.inventor, True)
    # Several authors and verifiers share the work in rotation, spreading the load over each model's rate limit.
    authors = [metered(model.strip(), True) for model in args.author.split(",")]
    verifiers = [metered(model.strip(), False) for model in args.verifier.split(",")]
    verifier = Rotation(verifiers)
    models = [inventor, *authors, *verifiers]
    done = {world["id"] for world in _read(out / "worlds.jsonl")}
    pending = [(i, spec) for i, spec in ((i, world_spec(args.seed, i, args.profile)) for i in range(args.worlds))
               if spec["id"] not in done]
    already = len(_read(out / "examples.jsonl"))
    held_out = eval_requests()
    lock, stop = threading.Lock(), threading.Event()
    totals = Counter()

    def append(name: str, records: list[dict]) -> None:
        with (out / name).open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def work(item: tuple[int, dict]) -> None:
        index, spec = item
        if stop.is_set() or (args.target and already + totals["kept"] >= args.target):
            return
        try:
            world, kept, rejected = make_world(args.seed, spec, inventor, authors[index % len(authors)],
                                               verifier, args.per_world)
        except FatalProviderError:
            stop.set()
            raise
        except Rejected as e:
            with lock:
                append("failures.jsonl", [{**spec, "why": str(e)}])
                totals["worlds_failed"] += 1
            return
        except Exception as e:  # a rate limit or timeout: leave the world for the next run
            with lock:
                append("failures.jsonl", [{**spec, "why": f"{type(e).__name__}: {e}"[:300]}])
                totals["worlds_failed"] += 1
            return
        repeats = [r for r in kept if " ".join(r["request"].lower().split()) in held_out]
        kept = [r for r in kept if r not in repeats]
        rejected += [{"world": r["world"], "n": r["n"], "topic": r["topic"], "why": "repeats an eval request",
                      "request": r["request"], "response": r["response"]} for r in repeats]
        with lock:
            # The world line goes last: its presence is what marks the world as finished.
            append("examples.jsonl", kept)
            append("rejects.jsonl", rejected)
            append("worlds.jsonl", [world])
            totals.update(worlds=1, kept=len(kept), rejected=len(rejected))
            if totals["worlds"] % 10 == 0:
                load = ", ".join(f"{m.model.rsplit('/', 1)[-1][:14]} {m.calls} calls/{getattr(m.provider, 'retried', 0)} retried"
                                 for m in models)
                print(f"  -- {totals['worlds']} worlds, {totals['worlds_failed']} failed | {load}", file=sys.stderr, flush=True)
            print(f"  {world['id']} {spec['topic'][:30]:30} {world['author'].rsplit('/', 1)[-1][:12]:12} "
                  f"kept {len(kept)}, rejected {len(rejected)}  (total {already + totals['kept']})",
                  file=sys.stderr, flush=True)

    print(f"{len(pending)} worlds to build, {len(done)} already done", file=sys.stderr, flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, pending))
    return {**totals, "examples_total": already + totals["kept"], "spend": [model.usage() for model in models]}


def rewrite(args) -> dict:
    """Rewrite the plan requests of a finished data folder as short statements of intent."""
    source, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def metered(model: str) -> Metered:
        provider = make_provider(args.provider, model, None, False)
        if hasattr(provider, "retries"):
            provider.retries, provider.timeout = 5, 120
        if hasattr(provider, "extra"):
            provider.extra = {"reasoning_effort": "low"}
        return Metered(provider, model)

    # One rewrite call per world comes in fast bursts, so it is spread over several models as well.
    rewriters = [metered(model.strip()) for model in args.rewriter.split(",")]
    rewriter = Rotation(rewriters)
    verifiers = [metered(model.strip()) for model in args.verifier.split(",")]
    verifier = Rotation(verifiers)
    by_world: dict[str, list] = {}
    for example in _read(source / "examples.jsonl"):
        by_world.setdefault(example["world"], []).append(example)
    done = {world["id"] for world in _read(out / "worlds.jsonl")}
    worlds_file = source / "worlds.jsonl"
    seen, pending = set(), []
    for world in _read(worlds_file):  # a rewrite folder lists a world once per run that touched it
        if world["id"] not in done and world["id"] in by_world and world["id"] not in seen:
            seen.add(world["id"])
            pending.append(world)
    lock, stop = threading.Lock(), threading.Event()
    totals = Counter()

    def append(name: str, records: list[dict]) -> None:
        with (out / name).open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def work(world: dict) -> None:
        if stop.is_set():
            return
        try:
            kept, rejected = rewrite_world(world, by_world[world["id"]], rewriter, verifier, args.style)
        except FatalProviderError:
            stop.set()
            raise
        except Exception as e:  # a bad reply, rate limit or timeout: leave the world for the next run
            with lock:
                append("failures.jsonl", [{"id": world["id"], "why": f"{type(e).__name__}: {e}"[:300]}])
                totals["worlds_failed"] += 1
            return
        with lock:
            append("examples.jsonl", kept)
            append("rejects.jsonl", rejected)
            append("worlds.jsonl", [world])
            totals.update(worlds=1, kept=len(kept), rejected=len(rejected))
            if totals["worlds"] % 25 == 0:
                print(f"  {totals['worlds']} worlds, kept {totals['kept']}, rejected {totals['rejected']}, "
                      f"failed {totals['worlds_failed']}", file=sys.stderr, flush=True)

    print(f"{len(pending)} worlds to rewrite, {len(done)} already done", file=sys.stderr, flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, pending))
    return {**totals, "spend": [model.usage() for model in (*rewriters, *verifiers)]}


def _by_author(worlds, examples, rejects) -> dict:
    author = {world["id"]: world.get("author") or "unknown" for world in worlds}
    kept = Counter(author[e["world"]] for e in examples)
    authored = kept + Counter(author[r["world"]] for r in rejects if r["world"] in author)
    return {name: f"{kept[name]}/{n} over {sum(a == name for a in author.values())} worlds"
            for name, n in authored.most_common()}


def stats(args) -> dict:
    out = Path(args.out)
    worlds, examples, rejects = (_read(out / name) for name in ("worlds.jsonl", "examples.jsonl", "rejects.jsonl"))
    authored = len(examples) + len(rejects)
    reasons = Counter(r["why"].split(":")[0].split(" (")[0][:60] for r in rejects)
    by_topic = Counter(e["topic"] for e in examples)
    authored_by_topic = by_topic + Counter(r["topic"] for r in rejects)
    return {
        "worlds": len(worlds),
        "world_failures_logged": len(_read(out / "failures.jsonl")),
        "examples_kept": len(examples),
        "examples_rejected": len(rejects),
        "yield": round(len(examples) / authored, 3) if authored else None,
        "kept_per_world": round(len(examples) / len(worlds), 2) if worlds else None,
        "by_status": dict(Counter(e["response"]["status"] for e in examples)),
        "by_topic_kept_of_authored": {t: f"{by_topic[t]}/{n}" for t, n in authored_by_topic.most_common()},
        "by_style": dict(Counter(e["style"] for e in examples)),
        "steps_per_plan": dict(sorted(Counter(len(e["response"]["steps"]) for e in examples
                                              if e["response"]["status"] == "plan").items())),
        "traits": dict(Counter(trait for world in worlds for trait in world["traits"])),
        "by_author_kept_of_authored": _by_author(worlds, examples, rejects),
        "kept_by_verifier": dict(Counter((e.get("verifier") or "none").rsplit("/", 1)[-1] for e in examples)),
        "rejected_because": dict(reasons.most_common()),
    }


def _in_validation(world_id: str, percent: int) -> bool:
    return int(hashlib.md5(world_id.encode()).hexdigest(), 16) % 100 < percent


def export(args) -> dict:
    """Write chat-format files for MLX-LM. Worlds are split whole, so no table appears on both sides."""
    out = Path(args.out)
    worlds = {world["id"]: world for world in _read(out / "worlds.jsonl")}
    target = out / "mlx"
    target.mkdir(exist_ok=True)
    splits = {"train": [], "valid": []}
    for example in _read(out / "examples.jsonl"):
        tables = {name: Table.from_rows(name, rows) for name, rows in worlds[example["world"]]["tables"].items()}
        shown = json.dumps(planner_input(example["request"], tables), separators=(",", ":"), ensure_ascii=False)
        answer = json.dumps(example["response"], separators=(",", ":"), ensure_ascii=False)
        messages = [{"role": "system", "content": VIC_SYSTEM}, {"role": "user", "content": shown},
                    {"role": "assistant", "content": answer}]
        split = "valid" if _in_validation(example["world"], args.valid_percent) else "train"
        splits[split].append({"messages": messages})
    for name, rows in splits.items():
        (target / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    sizes = [sum(len(m["content"]) for m in row["messages"]) for rows in splits.values() for row in rows]
    return {"train": len(splits["train"]), "valid": len(splits["valid"]),
            "characters_per_example": round(sum(sizes) / len(sizes)) if sizes else 0, "files": str(target)}


def export_eval(args) -> dict:
    """Write the validation worlds as an eval set, to measure a model on unseen data of the training kind."""
    out = Path(args.out)
    worlds = {world["id"]: world for world in _read(out / "worlds.jsonl")}
    target = out / "valid_eval"
    target.mkdir(exist_ok=True)
    tasks = []
    for example in _read(out / "examples.jsonl"):
        if not _in_validation(example["world"], args.valid_percent):
            continue
        tables = worlds[example["world"]]["tables"]
        response = example["response"]
        task = {"id": example["id"], "category": example["topic"], "tags": [example["style"]],
                "request": example["request"], "tables": list(tables), "inline_tables": tables}
        if response["status"] == "plan":
            try:
                result = run_plan(response, {name: Table.from_rows(name, rows) for name, rows in tables.items()})
            except PlanError:
                continue
            task["gold"] = response
            task["expect"] = {"status": ["plan"], "columns": result.columns, "rows": result.rows,
                              "ordered": any(step["op"] == "sort" for step in response["steps"])}
        else:
            task["expect"] = {"status": [response["status"]]}
        tasks.append(task)
    (target / "tables.json").write_text("{}\n", encoding="utf-8")
    (target / "tasks.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks), encoding="utf-8")
    return {"tasks": len(tasks), "files": str(target)}


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    parser = argparse.ArgumentParser(prog="datagen.run", description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/train_v1", help="output directory (default data/train_v1)")
    commands = parser.add_subparsers(dest="command", required=True)

    making = commands.add_parser("generate", help="build training worlds and their verified examples")
    making.add_argument("--worlds", type=int, required=True, help="build worlds 0..N-1 that are not done yet")
    making.add_argument("--target", type=int, help="stop starting new worlds once this many examples are kept")
    making.add_argument("--per-world", type=int, default=10, help="examples to author per world (default 10)")
    making.add_argument("--seed", type=int, default=1)
    making.add_argument("--profile", choices=("broad", "focus", "target"), default="broad",
                        help="broad: the general mix. focus: two-table worlds with unit columns, weighted to "
                             "joins, conversions, data quirks and rarer operations. target: aimed at the k2 "
                             "model's remaining failures on evals/v2 (see datagen/prompts.py)")
    making.add_argument("--provider", choices=("cloudflare", "anthropic", "ollama"), default="cloudflare")
    making.add_argument("--inventor", default="@cf/openai/gpt-oss-120b", help="model that invents the tables")
    making.add_argument("--author", default="@cf/zai-org/glm-5.3",
                        help="model that writes requests and plans; several, comma-separated, take turns")
    making.add_argument("--effort", default="low", help="reasoning effort for every model, or 'default' to "
                        "leave it to the model (default low)")
    making.add_argument("--verifier", default="@cf/openai/gpt-oss-120b,@cf/qwen/qwen3.8-27b",
                        help="model that re-solves each request in pandas; several, comma-separated, take turns")
    making.add_argument("--workers", type=int, default=30, help="worlds built at the same time (default 30)")

    commands.add_parser("stats", help="yield, mix, and reasons for rejection")
    exporting = commands.add_parser("export", help="write MLX-LM chat files")
    exporting.add_argument("--valid-percent", type=int, default=5)
    rewording = commands.add_parser("rewrite", help="rewrite a finished folder's plan requests as short statements of intent")
    rewording.add_argument("--source", required=True, help="finished data folder to rewrite, e.g. data/train_all")
    rewording.add_argument("--style", choices=("intent", "natural"), default="intent",
                           help="intent: short goal, no operation names. natural: columns and values by meaning, not by name")
    rewording.add_argument("--provider", choices=("cloudflare", "anthropic", "ollama"), default="cloudflare")
    rewording.add_argument("--rewriter", default="@cf/zai-org/glm-5.3,@cf/openai/gpt-oss-120b,@cf/qwen/qwen3.8-27b",
                           help="models that reword the requests; several, comma-separated, take turns")
    rewording.add_argument("--verifier", default="@cf/openai/gpt-oss-120b,@cf/qwen/qwen3.8-27b")
    rewording.add_argument("--workers", type=int, default=15)
    as_eval = commands.add_parser("export-eval", help="write the validation worlds as an eval set for bench.run")
    as_eval.add_argument("--valid-percent", type=int, default=5)

    args = parser.parse_args(argv)
    try:
        result = {"generate": generate, "stats": stats, "export": export, "export-eval": export_eval,
                  "rewrite": rewrite}[args.command](args)
    except FatalProviderError as e:
        print(json.dumps({"status": "error", "message": str(e)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
