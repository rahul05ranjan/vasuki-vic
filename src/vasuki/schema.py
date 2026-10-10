"""The response format a planner model emits, as JSON Schema.

A response is a plan, a clarification request, or a refusal. The schema is
built from the op registry, so it can be handed to a constrained decoder and
never drifts from what the executor accepts.
"""

from __future__ import annotations

from . import units
from .errors import PlanError, type_name, unknown
from .expr import FUNCTIONS, SYNTAX
from .ops import OPS, Op

_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}

_TEXT = {"type": "string", "minLength": 1}

_CLARIFICATION = {
    "type": "object",
    "properties": {
        "status": {"const": "needs_clarification"},
        "question": _TEXT,
        "missing": {"type": "array", "items": _TEXT},
    },
    "required": ["status", "question"],
    "additionalProperties": False,
}
_UNSUPPORTED = {
    "type": "object",
    "properties": {"status": {"const": "unsupported"}, "reason": _TEXT},
    "required": ["status", "reason"],
    "additionalProperties": False,
}
_STATUSES = ("plan", "needs_clarification", "unsupported")


def validate(value, schema: dict, path: str = "$") -> None:
    """Check `value` against the subset of JSON Schema this package uses."""
    expected = schema.get("type")
    if expected and not _TYPES[expected](value):
        raise PlanError("invalid_plan", f"{path}: expected {expected}, got {type_name(value)}")
    if "const" in schema and value != schema["const"]:
        raise PlanError("invalid_plan", f"{path}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        options = ", ".join(repr(o) for o in schema["enum"])
        raise PlanError("invalid_plan", f"{path}: must be one of {options}, got {value!r}")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        raise PlanError("invalid_plan", f"{path}: must not be empty")
    if "minimum" in schema and value < schema["minimum"]:
        raise PlanError("invalid_plan", f"{path}: must be at least {schema['minimum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise PlanError("invalid_plan", f"{path}: needs at least {schema['minItems']} item(s)")
        if "items" in schema:
            for i, item in enumerate(value):
                validate(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key in schema.get("required", ()):
            if key not in value:
                raise PlanError("missing_field", f"{path}: missing required field `{key}`")
        if len(value) < schema.get("minProperties", 0):
            raise PlanError("invalid_plan", f"{path}: must not be empty")
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], f"{path}.{key}")
            elif extra is False:
                error = unknown("field", key, properties)
                raise PlanError("invalid_plan", f"{path}: {error.message}")
            elif isinstance(extra, dict):
                validate(item, extra, f"{path}.{key}")


def step_schema(op: Op) -> dict:
    return {
        "type": "object",
        "description": op.summary,
        "properties": {"op": {"const": op.name}, "input": _TEXT, "as": _TEXT, **op.properties},
        "required": ["op", *op.required],
        "additionalProperties": False,
    }


def _plan_schema(step_items: dict | None) -> dict:
    steps = {"type": "array", "minItems": 1}
    if step_items:
        steps["items"] = step_items
    return {
        "type": "object",
        "properties": {"status": {"const": "plan"}, "steps": steps},
        "required": ["status", "steps"],
        "additionalProperties": False,
    }


def response_schema() -> dict:
    steps = {"oneOf": [step_schema(op) for op in OPS.values()]}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "oneOf": [_plan_schema(steps), _CLARIFICATION, _UNSUPPORTED],
    }


def validate_response(response) -> None:
    if not isinstance(response, dict):
        raise PlanError("invalid_plan", f"$: expected object, got {type_name(response)}")
    status = response.get("status")
    if status not in _STATUSES:
        raise PlanError("invalid_plan", f"$.status: must be one of {', '.join(_STATUSES)}")
    if status == "needs_clarification":
        return validate(response, _CLARIFICATION)
    if status == "unsupported":
        return validate(response, _UNSUPPORTED)
    validate(response, _plan_schema(None))
    for i, step in enumerate(response["steps"]):
        name = step.get("op") if isinstance(step, dict) else None
        try:
            if not isinstance(name, str):
                raise PlanError("invalid_plan", f"$.steps[{i}]: each step must be an object with an `op`")
            if name not in OPS:
                raise unknown("op", name, OPS)
            validate(step, step_schema(OPS[name]), f"$.steps[{i}]")
        except PlanError as e:
            e.step, e.op = i, name if isinstance(name, str) else None
            raise


def describe() -> dict:
    """Everything a planner needs in its prompt: the format, the ops, the functions, the units."""
    return {
        "response_schema": response_schema(),
        "ops": {name: op.summary for name, op in OPS.items()},
        "steps": (
            "Steps run in order. A step reads the previous step's result unless `input` names a "
            "table. With more than one input table, the first step must name its `input`. "
            "`as` names a step's result so a later step or join can use it, and may reuse a name. "
            "The last step's result is the output."
        ),
        "expressions": {"syntax": SYNTAX, "functions": [fn.signature for fn in FUNCTIONS.values()]},
        "units": units.DIMENSIONS,
    }
