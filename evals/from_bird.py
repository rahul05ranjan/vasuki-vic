"""Build an eval set of questions real people wrote, from a second source: the BIRD text-to-SQL benchmark.

    uv run python -I evals/from_bird.py        # writes evals/real2

Needs data/bird/dev_20240627/dev.json and data/bird/dev_20240627/dev_databases/<db>/<db>.sqlite, from
https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip (unpack it, then the dev_databases.zip inside it).
BIRD is by Li et al., 2023, released under CC BY-SA 4.0; the questions, the tables and the expected
results in this set are derived from it and carry the same licence.

Why: evals/real1 comes from Spider, and from run k5 onward VIC is trained on Spider's train split, so
real1 measures questions worded the way the training data's are. BIRD was written by other people about
other, real databases, and nothing of it is in any training data here.

  evals/real2     from BIRD's dev split (6 of its 11 databases: three have no table small enough, and two
                  are databases Spider also has). A TEST set: score on it, never steer by it.

The same rules as evals/from_spider.py decide which questions are kept, and the same code applies them.
BIRD gives each question a line of "evidence" saying what its terms mean in the database ("full name
refers to first_name, last_name"); every model on BIRD is given it, so it is part of the request here,
after "Note:". Score by `accuracy_ignoring_column_names`.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIRD = ROOT / "data" / "bird" / "dev_20240627"
NAME = "real2"
# The same databases under another name in Spider's train split (formula_1, soccer_1), which VIC is trained on.
ALSO_IN_SPIDER = {"formula_1", "european_football_2"}


def load_rules():
    """evals/from_spider.py holds the rules for which questions are usable; reuse them unchanged."""
    spec = importlib.util.spec_from_file_location("from_spider", ROOT / "evals" / "from_spider.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATABASES = BIRD / "dev_databases"
    return module


def build() -> dict:
    rules = load_rules()
    items = json.loads((BIRD / "dev.json").read_text(encoding="utf-8"))
    cache, tasks, tables, reasons = {}, [], {}, Counter()
    for item in items:
        if item["db_id"] in ALSO_IN_SPIDER:
            reasons["the database is also in Spider"] += 1
            continue
        evidence = " ".join(item.get("evidence", "").split())
        request = item["question"].strip() + (f"\nNote: {evidence}" if evidence else "")
        built = rules.build_task(len(tasks) + 1, NAME, {"db_id": item["db_id"], "query": item["SQL"], "question": request}, cache)
        if isinstance(built, str):
            reasons[built] += 1
            continue
        tasks.append({**built[0], "tags": [item["db_id"], item["difficulty"]]})
        tables.update(built[1])
    out = ROOT / "evals" / NAME
    out.mkdir(exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks), encoding="utf-8")
    lines = [f"  {json.dumps(key)}: [\n" + ",\n".join("    " + json.dumps(row, ensure_ascii=False) for row in rows) + "\n  ]"
             for key, rows in sorted(tables.items())]
    (out / "tables.json").write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")
    return {"set": NAME, "questions_seen": len(items), "tasks": len(tasks),
            "databases": len({t["tags"][0] for t in tasks}), "tables": len(tables),
            "by_kind": dict(Counter(t["category"] for t in tasks)),
            "by_difficulty": dict(Counter(t["tags"][1] for t in tasks)),
            "ordered": sum(t["expect"]["ordered"] for t in tasks), "left_out": dict(reasons),
            "megabytes": round((out / "tables.json").stat().st_size / 1e6, 2)}


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
