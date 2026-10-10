"""What a planner model is shown: the request and a profile of each table.

The profile is computed locally, so the caller never has to read the data
itself. It carries column names and types, a few sample rows, and the distinct
values of low-cardinality text columns so literals can be matched exactly.
"""

from __future__ import annotations

from .errors import is_number
from .executor import Table

# What a fine-tuned planner is given in place of the full format description from `describe()`.
VIC_SYSTEM = (
    "You are VIC, a planner for the Vasuki executor. Turn the request into one JSON object: a plan, a "
    "needs_clarification question, or an unsupported reason. Reply with the JSON only."
)


def _kind(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return "json"


def _column_type(values: list) -> str:
    kinds = {_kind(v) for v in values}
    if not kinds:
        return "null"
    if kinds <= {"int", "float"} and all(is_number(v) for v in values):
        return "float" if "float" in kinds else "int"
    return kinds.pop() if len(kinds) == 1 else "mixed"


def profile_table(table: Table, sample: int = 3, max_values: int = 20) -> dict:
    # A text column's distinct values are listed when there are at most `max_values`. At 8, a column of 14
    # model names went unlisted and the planner dropped filters on it; at 20 it kept them (found in a trial with an agent's own questions).
    columns = []
    for name in table.columns:
        present = [row[name] for row in table.rows if row[name] is not None]
        column = {"name": name, "type": _column_type(present)}
        nulls = len(table.rows) - len(present)
        if nulls:
            column["nulls"] = nulls
        if column["type"] == "str":
            distinct = list(dict.fromkeys(present))
            if len(distinct) <= max_values:
                column["values"] = distinct
        columns.append(column)
    return {"rows": len(table.rows), "columns": columns, "sample": table.rows[:sample]}


def planner_input(request: str, tables: dict[str, Table]) -> dict:
    return {"request": request, "tables": {name: profile_table(table) for name, table in tables.items()}}
