"""The operations a plan step can name.

Each op is one `prepare(step, columns, schemas)` function. It checks the step
against the input columns without touching data, then returns the output
columns and a `run(rows, tables)` closure. Static checking a plan is therefore
just preparing every step and not running any.

Ops never mutate the rows they are given: a table may feed several steps.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from .errors import PlanError, is_number, type_name, unknown
from .expr import Expr, cast_value
from .units import check_units, convert

_NAME = {"type": "string", "minLength": 1}
_NAMES = {"type": "array", "items": _NAME, "minItems": 1}

AGGREGATES = ("count", "count_distinct", "sum", "mean", "median", "min", "max")


@dataclass(frozen=True)
class Op:
    name: str
    summary: str
    properties: dict
    required: tuple[str, ...]
    prepare: Callable


OPS: dict[str, Op] = {}


def op(name: str, summary: str, properties: dict, required=()):
    def register(prepare):
        OPS[name] = Op(name, summary, properties, tuple(required), prepare)
        return prepare
    return register


def _need(columns, available) -> None:
    for column in columns:
        if column not in available:
            raise unknown("column", column, available)


def _unique(columns) -> None:
    seen = set()
    for column in columns:
        if column in seen:
            raise PlanError("duplicate_column", f"column {column!r} would appear twice in the result")
        seen.add(column)


def _compile(source: str, columns) -> Expr:
    expr = Expr(source)
    _need(expr.columns, columns)
    return expr


def _per_row(rows, fn) -> list:
    out = []
    for i, row in enumerate(rows):
        try:
            out.append(fn(row))
        except PlanError as e:
            if e.row is None:
                e.row = i
            raise
    return out


def _hashable(value):
    # Keep True apart from 1, and make nested JSON values usable as keys.
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (dict, list)):
        return ("json", json.dumps(value, sort_keys=True))
    return value


_NUMBER_TEXT = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)")


def _key_readers(left_values, right_values):
    """How to read one pair of join key columns so that equal keys compare equal.

    A database export often holds the same id as a number in one table and as text in the other. Text is
    then read as the number it spells; text that spells no number matches nothing. Two text columns are
    left alone, so "007" and "7" stay different there.
    """
    def kinds(values):
        return {"number" if is_number(v) else type_name(v) for v in values if v is not None}

    def as_number(value):
        text = value.strip()
        if not _NUMBER_TEXT.fullmatch(text):
            return ("not a number", value)
        return int(text) if text.lstrip("+-").isdigit() else float(text)

    left, right = kinds(left_values), kinds(right_values)
    if left == {"number"} and right == {"str"}:
        return _hashable, as_number
    if left == {"str"} and right == {"number"}:
        return as_number, _hashable
    return _hashable, _hashable


def _same_kind(values, column: str, what: str) -> None:
    kinds = {"number" if is_number(v) else type_name(v) for v in values}
    if len(kinds) > 1 or kinds - {"number", "str", "bool"}:
        raise PlanError(
            "type_error",
            f"cannot {what} column {column!r}: it mixes {', '.join(sorted(kinds))}",
        )


@op("filter", "Keep the rows where `where` is true.", {"where": _NAME}, required=["where"])
def _filter(step, columns, schemas):
    expr = _compile(step["where"], columns)

    def test(row):
        value = expr.eval(row)
        if value is None or isinstance(value, bool):
            return bool(value)
        raise PlanError(
            "type_error", f"in `{expr.source}`: a filter must be true or false, got {type_name(value)} {value!r}"
        )

    def run(rows, tables):
        keep = _per_row(rows, test)
        return [row for row, kept in zip(rows, keep) if kept]

    return columns, run


@op(
    "derive",
    "Set `column` to the value of `expr` for every row, adding the column if it is new.",
    {"column": _NAME, "expr": _NAME},
    required=["column", "expr"],
)
def _derive(step, columns, schemas):
    column, expr = step["column"], _compile(step["expr"], columns)
    out = columns if column in columns else [*columns, column]

    def run(rows, tables):
        return _per_row(rows, lambda row: {**row, column: expr.eval(row)})

    return out, run


@op(
    "rename",
    "Rename columns; `columns` maps old name to new name.",
    {"columns": {"type": "object", "additionalProperties": _NAME, "minProperties": 1}},
    required=["columns"],
)
def _rename(step, columns, schemas):
    mapping = step["columns"]
    _need(mapping, columns)
    out = [mapping.get(c, c) for c in columns]
    _unique(out)

    def run(rows, tables):
        return [{mapping.get(k, k): v for k, v in row.items()} for row in rows]

    return out, run


@op("select", "Keep only `columns`, in the order given.", {"columns": _NAMES}, required=["columns"])
def _select(step, columns, schemas):
    wanted = step["columns"]
    _need(wanted, columns)
    _unique(wanted)

    def run(rows, tables):
        return [{c: row[c] for c in wanted} for row in rows]

    return list(wanted), run


@op("drop", "Remove `columns`.", {"columns": _NAMES}, required=["columns"])
def _drop(step, columns, schemas):
    _need(step["columns"], columns)
    dropped = set(step["columns"])
    out = [c for c in columns if c not in dropped]

    def run(rows, tables):
        return [{c: row[c] for c in out} for row in rows]

    return out, run


@op(
    "cast",
    "Convert `column` to another type; a value that cannot convert is an error.",
    {"column": _NAME, "to": {"enum": ["int", "float", "str", "bool"]}},
    required=["column", "to"],
)
def _cast(step, columns, schemas):
    column, to = step["column"], step["to"]
    _need([column], columns)

    def run(rows, tables):
        return _per_row(rows, lambda row: {**row, column: cast_value(row[column], to)})

    return columns, run


@op(
    "sort",
    "Order rows by one or more columns; nulls go last.",
    {
        "by": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {"column": _NAME, "order": {"enum": ["asc", "desc"]}},
                "required": ["column"],
                "additionalProperties": False,
            },
        }
    },
    required=["by"],
)
def _sort(step, columns, schemas):
    keys = [(key["column"], key.get("order", "asc") == "desc") for key in step["by"]]
    _need([column for column, _ in keys], columns)

    def run(rows, tables):
        rows = list(rows)
        # Stable sorts applied from the last key to the first give a multi-key sort.
        for column, descending in reversed(keys):
            present = [row for row in rows if row[column] is not None]
            missing = [row for row in rows if row[column] is None]
            _same_kind((row[column] for row in present), column, "sort")
            present.sort(key=lambda row: row[column], reverse=descending)
            rows = present + missing
        return rows

    return columns, run


@op(
    "deduplicate",
    "Drop rows that repeat an earlier row on `keys` (all columns if omitted).",
    {"keys": _NAMES, "keep": {"enum": ["first", "last"]}},
)
def _deduplicate(step, columns, schemas):
    keys = step.get("keys", columns)
    _need(keys, columns)
    keep_last = step.get("keep", "first") == "last"

    def run(rows, tables):
        seen, out = set(), []
        for row in reversed(rows) if keep_last else rows:
            key = tuple(_hashable(row[k]) for k in keys)
            if key not in seen:
                seen.add(key)
                out.append(row)
        return out[::-1] if keep_last else out

    return columns, run


def _aggregate_values(fn: str, column: str | None, rows):
    if column is None:
        return len(rows)
    values = [row[column] for row in rows if row[column] is not None]
    if fn == "count":
        return len(values)
    if fn == "count_distinct":
        return len({_hashable(v) for v in values})
    if fn in ("min", "max"):
        if not values:
            return None
        _same_kind(values, column, f"take the {fn} of")
        return min(values) if fn == "min" else max(values)
    for value in values:
        if not is_number(value):
            raise PlanError(
                "type_error",
                f"cannot take the {fn} of column {column!r}: found {type_name(value)} {value!r}",
            )
    if fn == "sum":
        return sum(values) if all(isinstance(v, int) for v in values) else math.fsum(values)
    if not values:
        return None
    return math.fsum(values) / len(values) if fn == "mean" else statistics.median(values)


@op(
    "aggregate",
    "Group rows by `group_by` and compute one value per metric. Nulls are skipped; "
    "an empty `group_by` gives a single total row. Only `count` may omit `column`.",
    {
        "group_by": {"type": "array", "items": _NAME},
        "metrics": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {"as": _NAME, "fn": {"enum": list(AGGREGATES)}, "column": _NAME},
                "required": ["as", "fn"],
                "additionalProperties": False,
            },
        },
    },
    required=["metrics"],
)
def _aggregate(step, columns, schemas):
    group_by = step.get("group_by", [])
    metrics = step["metrics"]
    _need(group_by, columns)
    for metric in metrics:
        if "column" in metric:
            _need([metric["column"]], columns)
        elif metric["fn"] != "count":
            raise PlanError("missing_field", f"metric {metric['as']!r} ({metric['fn']}) needs a `column`")
    out = [*group_by, *(metric["as"] for metric in metrics)]
    _unique(out)

    def run(rows, tables):
        groups: dict = {}
        for row in rows:
            groups.setdefault(tuple(_hashable(row[k]) for k in group_by), []).append(row)
        if not group_by and not groups:
            groups[()] = []
        result = []
        for members in groups.values():
            record = {k: members[0][k] for k in group_by}
            for metric in metrics:
                record[metric["as"]] = _aggregate_values(metric["fn"], metric.get("column"), members)
            result.append(record)
        return result

    return out, run


@op(
    "join",
    "Add the columns of table `right` to matching rows. Give `on` when the key columns share "
    "names, else `left_on` and `right_on`. Null keys never match. A key held as a number in one table "
    "and as text in the other is matched by the number. The right table's key columns "
    "are not repeated in the result; after a left join, an unmatched row has null in every "
    "column that came from the right. Any other right column whose name already exists gets "
    "`right_suffix` (default `_right`).",
    {
        "right": _NAME,
        "on": _NAMES,
        "left_on": _NAMES,
        "right_on": _NAMES,
        "how": {"enum": ["inner", "left"]},
        "right_suffix": _NAME,
    },
    required=["right"],
)
def _join(step, columns, schemas):
    right_name = step["right"]
    if right_name not in schemas:
        raise unknown("table", right_name, schemas)
    right_columns = schemas[right_name]

    if "on" in step:
        if "left_on" in step or "right_on" in step:
            raise PlanError("invalid_plan", "give either `on` or `left_on` with `right_on`, not both")
        left_keys = right_keys = step["on"]
    elif "left_on" in step and "right_on" in step:
        left_keys, right_keys = step["left_on"], step["right_on"]
        if len(left_keys) != len(right_keys):
            raise PlanError("invalid_plan", "`left_on` and `right_on` must list the same number of columns")
    else:
        raise PlanError("missing_field", "join needs `on`, or `left_on` with `right_on`")
    _need(left_keys, columns)
    _need(right_keys, right_columns)

    suffix = step.get("right_suffix", "_right")
    carried = {c: (c + suffix if c in columns else c) for c in right_columns if c not in right_keys}
    out = [*columns, *carried.values()]
    _unique(out)
    keep_unmatched = step.get("how", "inner") == "left"
    blank = dict.fromkeys(carried.values())

    def run(rows, tables):
        right_rows = tables[right_name]
        readers = [
            _key_readers([row[l] for row in rows], [row[r] for row in right_rows])
            for l, r in zip(left_keys, right_keys)
        ]
        index = defaultdict(list)
        for row in right_rows:
            if all(row[k] is not None for k in right_keys):
                index[tuple(read(row[k]) for k, (_, read) in zip(right_keys, readers))].append(row)
        result = []
        for row in rows:
            matches = ()
            if all(row[k] is not None for k in left_keys):
                matches = index.get(tuple(read(row[k]) for k, (read, _) in zip(left_keys, readers)), ())
            for match in matches:
                result.append({**row, **{new: match[old] for old, new in carried.items()}})
            if not matches and keep_unmatched:
                result.append({**row, **blank})
        return result

    return out, run


@op(
    "convert_unit",
    "Convert the numbers in `column` between units of the same dimension. "
    "Writes to `to_column` if given, else in place.",
    {"column": _NAME, "from": _NAME, "to": _NAME, "to_column": _NAME},
    required=["column", "from", "to"],
)
def _convert_unit(step, columns, schemas):
    column, src, dst = step["column"], step["from"], step["to"]
    target = step.get("to_column", column)
    _need([column], columns)
    check_units(src, dst)
    out = columns if target in columns else [*columns, target]

    def run(rows, tables):
        return _per_row(rows, lambda row: {**row, target: convert(row[column], src, dst)})

    return out, run


@op("limit", "Keep the first `n` rows.", {"n": {"type": "integer", "minimum": 0}}, required=["n"])
def _limit(step, columns, schemas):
    n = step["n"]

    def run(rows, tables):
        return rows[:n]

    return columns, run
