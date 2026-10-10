from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .errors import PlanError
from .executor import Table

# No leading zeros: "007" and "01234" are identifiers or postcodes, not numbers.
_INT = re.compile(r"[+-]?(0|[1-9]\d*)")
_FLOAT = re.compile(r"[+-]?((0|[1-9]\d*)(\.\d+)?|\.\d+)([eE][+-]?\d+)?")


def _parse_column(values: list[str]) -> list:
    """Type a CSV column as a whole: int, else float, else true/false, else text. Empty cells are null."""
    present = [v.strip() for v in values if v.strip()]
    if present and all(_INT.fullmatch(v) for v in present):
        parse = int
    elif present and all(_FLOAT.fullmatch(v) for v in present):
        parse = float
    elif present and all(v.lower() in ("true", "false") for v in present):
        # What pandas, spreadsheets and most exports write for a boolean column.
        parse = lambda v: v.lower() == "true"
    else:
        return [v if v != "" else None for v in values]
    return [parse(v.strip()) if v.strip() else None for v in values]


def _load_delimited(path: Path, delimiter: str) -> Table:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        if not header:
            raise PlanError("invalid_input", f"{path.name}: file is empty")
        if len(set(header)) != len(header):
            raise PlanError("invalid_input", f"{path.name}: header repeats a column name")
        records = []
        for record in reader:
            if not record:
                continue
            if len(record) > len(header):
                raise PlanError(
                    "invalid_input",
                    f"{path.name} line {reader.line_num}: {len(record)} fields, header has {len(header)}",
                )
            records.append(record + [""] * (len(header) - len(record)))
    parsed = [_parse_column([record[i] for record in records]) for i in range(len(header))]
    return Table(header, [dict(zip(header, values)) for values in zip(*parsed)])


def load_table(path: str | Path) -> Table:
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return _load_delimited(path, ",")
        if suffix == ".tsv":
            return _load_delimited(path, "\t")
        if suffix == ".json":
            return Table.from_rows(path.name, json.loads(path.read_text(encoding="utf-8")))
        if suffix == ".jsonl":
            lines = path.read_text(encoding="utf-8").splitlines()
            return Table.from_rows(path.name, [json.loads(line) for line in lines if line.strip()])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error) as e:
        raise PlanError("invalid_input", f"{path.name}: {e}") from None
    raise PlanError("invalid_input", f"{path.name}: unsupported file type, use .csv, .tsv, .json or .jsonl")


def load_tables(specs: list[str]) -> dict[str, Table]:
    """Load several inputs, each `PATH` or `NAME=PATH`; a table's name defaults to its file's stem."""
    tables = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep:
            name, path = Path(spec).stem, spec
        if name in tables:
            raise PlanError("invalid_input", f"two inputs are named {name!r}; use NAME=PATH to rename one")
        tables[name] = load_table(path)
    return tables


def write_table(path: str | Path, columns: list[str], rows: list[dict]) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t" if suffix == ".tsv" else ",")
            writer.writerow(columns)
            writer.writerows(["" if row[c] is None else row[c] for c in columns] for row in rows)
    elif suffix == ".json":
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif suffix == ".jsonl":
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    else:
        raise PlanError("invalid_input", f"{path.name}: unsupported file type, use .csv, .tsv, .json or .jsonl")
