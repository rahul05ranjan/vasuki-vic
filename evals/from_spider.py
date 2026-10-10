"""Build eval sets of questions real people wrote, from the Spider text-to-SQL dataset.

    uv run python -I evals/from_spider.py        # writes evals/real1 and evals/real_dev

Needs data/spider/{train,validation}.jsonl and data/spider/database/<db>/<db>.sqlite (questions from
the Hugging Face dataset xlangai/spider, databases from prem-research/spider). Spider is by Yu et al.,
2018, released under CC BY-SA 4.0; the questions, the tables and the expected results in these two sets
are derived from it and carry the same licence.

Why: the hand-written sets in this repository pin their output (column names, rounding, order) the way the
training data does. Spider's questions were written by people who wanted an answer, and say none of that.
Each task's expected result is what Spider's own SQL returns, so nothing here was authored for VIC.

  evals/real1     from Spider's validation split (20 databases). The TEST set for realistic requests: score
                  on it, never steer by it.
  evals/real_dev  a sample from Spider's train split (other databases). The development set: read its
                  failures freely. Training data may be built from the rest of the train split, never from
                  these questions (datagen.run.EVAL_SETS lists both sets), nor from another question
                  with the same SQL: Spider words most of its queries twice. A task whose SQL has such a
                  second question carries the tag "paraphrased". A model trained on Spider data made
                  before 2026-10-10 (run k5) has seen those second questions, so read its score on the
                  "paraphrased" tasks and on the rest as two numbers.

A question is kept when its SQL is a single SELECT over one or two different tables, every table has at
most MAX_ROWS rows and plain column names, and the result has 1 to MAX_RESULT rows and does not depend on
the order rows happen to be stored in (checked by running the SQL again on a reversed copy). Because the
questions name no output columns, score these sets by `accuracy_ignoring_column_names`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPIDER = ROOT / "data" / "spider"
DATABASES = SPIDER / "database"     # one folder per database, holding <db>.sqlite; evals/from_bird.py points it elsewhere
MAX_ROWS, MAX_RESULT = 1000, 200
SETS = {"real1": ("validation", None, None), "real_dev": ("train", 250, 4)}   # split, size cap, cap per database
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
TABLE_REF = re.compile(r"\b(?:from|join)\s+([A-Za-z_]\w*)", re.IGNORECASE)


def sql_key(db: str, sql: str) -> tuple[str, str]:
    """What makes two questions the same question: the database and the SQL, spacing and case aside."""
    return db, " ".join(sql.lower().rstrip("; ").split())


def connect(db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DATABASES / db / (db + '.sqlite')}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def simple(sql: str) -> list[str] | None:
    """The tables a query reads, if it is one SELECT over one or two different tables; otherwise None."""
    low = f" {sql.lower()} "
    if low.count("select") != 1 or any(f" {word} " in low for word in ("intersect", "union", "except")):
        return None
    tables = [name.lower() for name in TABLE_REF.findall(sql)]
    return tables if tables and len(tables) <= 2 and len(set(tables)) == len(tables) else None


def export(conn: sqlite3.Connection, wanted: str):
    """(real table name, rows as dicts), or None when the table is too big or has names or values we cannot use."""
    names = {row[0].lower(): row[0] for row in conn.execute("select name from sqlite_master where type='table'")}
    if wanted not in names:
        return None
    cursor = conn.execute(f'select * from "{names[wanted]}"')
    columns = [d[0] for d in cursor.description]
    rows = cursor.fetchmany(MAX_ROWS + 1)
    if not rows or len(rows) > MAX_ROWS or not all(IDENT.fullmatch(c) for c in columns):
        return None
    if len({c.lower() for c in columns}) != len(columns) or any(isinstance(v, bytes) for row in rows for v in row):
        return None
    return names[wanted], [dict(zip(columns, row)) for row in rows]


def reversed_copy(db: str, tables: list[str]) -> sqlite3.Connection:
    """The same tables with their rows stored in the opposite order, to expose answers that depend on it."""
    conn = sqlite3.connect(":memory:")
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    source = connect(db)
    for table in tables:
        cursor = source.execute(f'select * from "{table}"')
        columns = ", ".join(f'"{d[0]}"' for d in cursor.description)
        declared = {row[1]: row[2] for row in source.execute(f'pragma table_info("{table}")')}
        typed = ", ".join(f'"{d[0]}" {declared.get(d[0], "")}' for d in cursor.description)
        conn.execute(f'create table "{table}" ({typed})')
        rows = cursor.fetchall()[::-1]
        conn.executemany(f'insert into "{table}" ({columns}) values ({", ".join("?" * len(cursor.description))})', rows)
    return conn


def run(conn: sqlite3.Connection, sql: str):
    cursor = conn.execute(sql)
    return [d[0] for d in cursor.description], cursor.fetchmany(MAX_RESULT + 1)


def kind(sql: str, tables: list[str]) -> str:
    low = sql.lower()
    if len(tables) == 2:
        return "join"
    if "group by" in low or re.search(r"\b(count|sum|avg|min|max)\s*\(", low):
        return "aggregate"
    return "sort" if "order by" in low else "basic"


def build_task(n: int, prefix: str, item: dict, cache: dict) -> tuple[dict, dict] | str:
    """A task and the tables it needs, or the reason the question was left out."""
    db, sql = item["db_id"], item["query"]
    tables = simple(sql)
    if tables is None:
        return "not a single SELECT over one or two tables"
    conn = cache.setdefault(db, connect(db))
    exported = {}
    for table in tables:
        key = (db, table)
        if key not in cache:
            cache[key] = export(conn, table)
        if cache[key] is None:
            return "a table is too big, empty, or has unusable names or values"
        exported[table] = cache[key]
    try:
        columns, rows = run(conn, sql)
        _, again = run(reversed_copy(db, [exported[t][0] for t in tables]), sql)
    except sqlite3.Error:
        return "the SQL does not run"
    if not 1 <= len(rows) <= MAX_RESULT:
        return "the result is empty or too long"
    ordered = "order by" in sql.lower()
    key = lambda row: tuple((v is None, str(type(v).__name__), v if v is not None else 0) for v in row)
    if rows != again:
        if sorted(rows, key=key) != sorted(again, key=key):
            return "the answer depends on the order rows are stored in"
        ordered = False     # the same rows either way: a sort with ties, or no sort at all
    unique = [name if columns.count(name) == 1 else f"{name}#{i}" for i, name in enumerate(columns)]
    task = {
        "id": f"{prefix}-{n:04d}", "category": kind(sql, tables), "tags": [db], "request": item["question"],
        "tables": {exported[t][0]: f"{db}.{exported[t][0]}" for t in tables},
        "expect": {"status": ["plan"], "ordered": ordered, "columns": unique, "rows": [dict(zip(unique, r)) for r in rows]},
        "sql": sql,
    }
    return task, {f"{db}.{exported[t][0]}": exported[t][1] for t in tables}


def build(name: str) -> dict:
    split, cap, per_db = SETS[name]
    items = [json.loads(line) for line in (SPIDER / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    if cap:
        # A fixed shuffle, so the sample spreads over the databases and never changes between builds.
        items.sort(key=lambda item: hashlib.md5((item["db_id"] + item["question"]).encode()).hexdigest())
    if split != "validation":
        # A generic question ("How many employees are there?") can occur in both splits; the test set keeps it.
        held = {json.loads(line)["question"] for line in (SPIDER / "validation.jsonl").read_text(encoding="utf-8").splitlines()}
        items = [item for item in items if item["question"] not in held]
    worded = Counter(sql_key(item["db_id"], item["query"]) for item in items)
    cache, tasks, tables, reasons, used = {}, [], {}, Counter(), Counter()
    for item in items:
        if cap and (len(tasks) >= cap or used[item["db_id"]] >= per_db):
            continue
        built = build_task(len(tasks) + 1, name, item, cache)
        if isinstance(built, str):
            reasons[built] += 1
            continue
        tasks.append(built[0])
        if cap and worded[sql_key(item["db_id"], item["query"])] > 1:
            built[0]["tags"].append("paraphrased")
        tables.update(built[1])
        used[item["db_id"]] += 1
    out = ROOT / "evals" / name
    out.mkdir(exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks), encoding="utf-8")
    lines = [f"  {json.dumps(key)}: [\n" + ",\n".join("    " + json.dumps(row, ensure_ascii=False) for row in rows) + "\n  ]"
             for key, rows in sorted(tables.items())]
    (out / "tables.json").write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")
    return {"set": name, "questions_seen": len(items), "tasks": len(tasks), "databases": len(used),
            "tables": len(tables), "by_kind": dict(Counter(t["category"] for t in tasks)),
            "ordered": sum(t["expect"]["ordered"] for t in tasks), "left_out": dict(reasons),
            "megabytes": round((out / "tables.json").stat().st_size / 1e6, 2)}


if __name__ == "__main__":
    for name in SETS:
        print(json.dumps(build(name), indent=1))
