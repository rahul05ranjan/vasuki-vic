from __future__ import annotations

from dataclasses import dataclass, field

from .errors import PlanError, unknown
from .ops import OPS
from .schema import validate_response

# Key for "the previous step's result"; an object so no table name can collide with it.
_PREVIOUS = object()


@dataclass
class Table:
    columns: list[str]
    rows: list[dict]

    @classmethod
    def from_rows(cls, name: str, rows) -> Table:
        """Build a table from JSON-style rows; a key missing from a row becomes null."""
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise PlanError("invalid_input", f"table {name!r} must be a list of objects")
        columns = list(dict.fromkeys(key for row in rows for key in row))
        return cls(columns, [{c: row.get(c) for c in columns} for row in rows])


@dataclass
class Result:
    columns: list[str]
    rows: list[dict]
    steps: list[dict] = field(default_factory=list)

    def report(self, preview: int = 5) -> dict:
        """A compact summary to hand back to the calling model instead of the full data."""
        return {
            "status": "ok",
            "rows": len(self.rows),
            "columns": self.columns,
            "steps": self.steps,
            "preview": self.rows[:preview],
        }


def _compile(plan, schemas: dict):
    validate_response(plan)
    if plan["status"] != "plan":
        raise PlanError("not_a_plan", f"response has status {plan['status']!r}; there is nothing to run")
    schemas = dict(schemas)
    compiled = []
    for i, step in enumerate(plan["steps"]):
        try:
            source = step.get("input")
            if source is None:
                if _PREVIOUS in schemas:
                    source = _PREVIOUS
                elif len(schemas) == 1:
                    source = next(iter(schemas))
                else:
                    raise PlanError(
                        "missing_field",
                        f"the first step needs `input`: choose one of {', '.join(schemas)}",
                    )
            elif source not in schemas:
                raise unknown("table", source, schemas)
            columns, run = OPS[step["op"]].prepare(step, schemas[source], schemas)
        except PlanError as e:
            e.step, e.op = i, step["op"]
            raise
        schemas[_PREVIOUS] = columns
        # A name may be reused, like reassigning a variable: later steps see the newest result.
        name = step.get("as")
        if name is not None:
            schemas[name] = columns
        compiled.append((source, name, columns, run))
    return compiled


def check_plan(plan, schemas: dict[str, list[str]]) -> list[str]:
    """Validate `plan` against the input column names alone; return the output columns."""
    return _compile(plan, schemas)[-1][2]


def run_plan(plan, tables: dict[str, Table | list[dict]]) -> Result:
    """Run `plan` over `tables`. The whole plan is checked before any step runs."""
    data = {
        name: table if isinstance(table, Table) else Table.from_rows(name, table)
        for name, table in tables.items()
    }
    compiled = _compile(plan, {name: table.columns for name, table in data.items()})
    rows_by_table: dict = {name: table.rows for name, table in data.items()}
    evidence = []
    for i, (source, name, columns, run) in enumerate(compiled):
        op_name = plan["steps"][i]["op"]
        rows_in = rows_by_table[source]
        try:
            rows_out = run(rows_in, rows_by_table)
        except PlanError as e:
            e.step, e.op = i, op_name
            raise
        rows_by_table[_PREVIOUS] = rows_out
        if name is not None:
            rows_by_table[name] = rows_out
        evidence.append({"op": op_name, "rows_in": len(rows_in), "rows_out": len(rows_out)})
    return Result(columns, rows_out, evidence)
