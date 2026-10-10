from __future__ import annotations

import difflib


class PlanError(Exception):
    """A plan that cannot run as written.

    `code` is a stable identifier callers can branch on. `step` and `row` are
    0-based and filled in by whichever layer knows them.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.step: int | None = None
        self.op: str | None = None
        self.row: int | None = None

    def to_dict(self) -> dict:
        out = {"status": "error", "code": self.code, "message": self.message}
        for field in ("step", "op", "row"):
            value = getattr(self, field)
            if value is not None:
                out[field] = value
        return out


def unknown(kind: str, name: str, options) -> PlanError:
    """Build an `unknown_<kind>` error that lists the options and the closest match."""
    options = [o for o in options if isinstance(o, str)]
    close = difflib.get_close_matches(name, options, n=1)
    hint = f", did you mean {close[0]!r}?" if close else ""
    listed = ", ".join(options) or "none"
    return PlanError(f"unknown_{kind}", f"unknown {kind} {name!r}{hint} (available: {listed})")


def type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
