"""Expression language for `filter` and `derive`.

Expressions use Python syntax but are never passed to eval(): the source is
parsed to an AST, checked against a whitelist, and interpreted here. Nulls
propagate through arithmetic and functions, and compare as false.
"""

from __future__ import annotations

import ast
import decimal
import io
import keyword
import math
import operator
import re
import tokenize
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable, NamedTuple

from .errors import PlanError, is_number, type_name, unknown

MAX_SOURCE = 2000

_CONSTANTS = {"null": None, "true": True, "false": False}

_ARITH = {
    ast.Add: (operator.add, "+"),
    ast.Sub: (operator.sub, "-"),
    ast.Mult: (operator.mul, "*"),
    ast.Div: (operator.truediv, "/"),
    ast.FloorDiv: (operator.floordiv, "//"),
    ast.Mod: (operator.mod, "%"),
}
_ORDER = {
    ast.Lt: (operator.lt, "<"),
    ast.LtE: (operator.le, "<="),
    ast.Gt: (operator.gt, ">"),
    ast.GtE: (operator.ge, ">="),
}
_MEMBERSHIP = (ast.In, ast.NotIn)
# `x is null` and `x is not null` are accepted as spellings of == and !=.
_EQUALITY = (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)

_INT = re.compile(r"[+-]?\d+")
_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}


def _to_str(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def cast_value(value, to: str):
    """Convert one value to `int`, `float`, `str` or `bool`; null stays null."""
    if value is None:
        return None
    try:
        if to == "str":
            return _to_str(value)
        if to == "int":
            if isinstance(value, int):
                return int(value)
            if isinstance(value, str) and _INT.fullmatch(value.strip()):
                return int(value.strip())
            number = float(value)
            if number == int(number):
                return int(number)
        elif to == "float":
            number = float(value)
            if math.isfinite(number):
                return number
        elif to == "bool":
            if isinstance(value, bool):
                return value
            text = _to_str(value).strip().lower()
            if text in _TRUE:
                return True
            if text in _FALSE:
                return False
    except (TypeError, ValueError, OverflowError):
        pass
    raise PlanError("type_error", f"cannot cast {type_name(value)} {value!r} to {to}")


def _num(fname: str, value):
    if not is_number(value):
        raise PlanError("type_error", f"{fname}() expects a number, got {type_name(value)} {value!r}")
    return value


def _text(fname: str, value) -> str:
    if not isinstance(value, str):
        raise PlanError("type_error", f"{fname}() expects text, got {type_name(value)} {value!r}")
    return value


def _date(fname: str, value) -> datetime:
    text = _text(fname, value)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        raise PlanError("type_error", f"{fname}() expects an ISO date, got {text!r}") from None


def _round(x, digits=0):
    _num("round", x)
    if not isinstance(digits, int) or isinstance(digits, bool):
        raise PlanError("type_error", f"round() digits must be an integer, got {digits!r}")
    if isinstance(x, int) and digits >= 0:
        return x
    try:
        # Decimal(str(x)) rounds the number as written, so 2.675 -> 2.68.
        result = Decimal(str(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    except decimal.InvalidOperation:
        raise PlanError("type_error", f"cannot round {x!r} to {digits} digits") from None
    return int(result) if digits <= 0 else float(result)


def _split_part(text, separator, n):
    _text("split_part", text)
    if not _text("split_part", separator):
        raise PlanError("type_error", "split_part() needs a non-empty separator")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise PlanError("type_error", f"split_part() counts pieces from 1, got {n!r}")
    pieces = text.split(separator)
    return pieces[n - 1] if n <= len(pieces) else None


def _concat(*parts):
    for part in parts:
        if not (isinstance(part, str) or is_number(part)):
            raise PlanError("type_error", f"concat() expects text or numbers, got {type_name(part)}")
    return "".join(_to_str(part) for part in parts)


def _coalesce(*values):
    return next((v for v in values if v is not None), None)


class Fn(NamedTuple):
    call: Callable
    min_args: int
    max_args: int | None
    signature: str
    # Null-propagating functions return null when any argument is null.
    propagates_null: bool = True


FUNCTIONS: dict[str, Fn] = {
    "abs": Fn(lambda x: abs(_num("abs", x)), 1, 1, "abs(x)"),
    "round": Fn(_round, 1, 2, "round(x, digits=0) rounds half up"),
    "floor": Fn(lambda x: math.floor(_num("floor", x)), 1, 1, "floor(x)"),
    "ceil": Fn(lambda x: math.ceil(_num("ceil", x)), 1, 1, "ceil(x)"),
    "min": Fn(lambda *xs: min(_num("min", x) for x in xs), 2, None, "min(a, b, ...) of numbers in one row"),
    "max": Fn(lambda *xs: max(_num("max", x) for x in xs), 2, None, "max(a, b, ...) of numbers in one row"),
    "lower": Fn(lambda s: _text("lower", s).lower(), 1, 1, "lower(text)"),
    "upper": Fn(lambda s: _text("upper", s).upper(), 1, 1, "upper(text)"),
    "strip": Fn(lambda s: _text("strip", s).strip(), 1, 1, "strip(text)"),
    "len": Fn(lambda s: len(_text("len", s)), 1, 1, "len(text)"),
    "concat": Fn(_concat, 1, None, "concat(a, b, ...) joins text and numbers"),
    "contains": Fn(lambda s, sub: _text("contains", sub) in _text("contains", s), 2, 2, "contains(text, part)"),
    "startswith": Fn(lambda s, p: _text("startswith", s).startswith(_text("startswith", p)), 2, 2, "startswith(text, prefix)"),
    "endswith": Fn(lambda s, p: _text("endswith", s).endswith(_text("endswith", p)), 2, 2, "endswith(text, suffix)"),
    "split_part": Fn(_split_part, 3, 3, "split_part(text, separator, n) nth piece counting from 1, null if there is none"),
    "is_null": Fn(lambda x: x is None, 1, 1, "is_null(x)", propagates_null=False),
    "coalesce": Fn(_coalesce, 1, None, "coalesce(a, b, ...) first non-null value", propagates_null=False),
    "int": Fn(lambda x: cast_value(x, "int"), 1, 1, "int(x)"),
    "float": Fn(lambda x: cast_value(x, "float"), 1, 1, "float(x)"),
    "str": Fn(lambda x: cast_value(x, "str"), 1, 1, "str(x)"),
    "year": Fn(lambda d: _date("year", d).year, 1, 1, "year(iso_date)"),
    "month": Fn(lambda d: _date("month", d).month, 1, 1, "month(iso_date)"),
    "day": Fn(lambda d: _date("day", d).day, 1, 1, "day(iso_date)"),
    "hour": Fn(lambda d: _date("hour", d).hour, 1, 1, "hour(iso_datetime)"),
    "minute": Fn(lambda d: _date("minute", d).minute, 1, 1, "minute(iso_datetime)"),
    "days_between": Fn(
        lambda a, b: (_date("days_between", b).date() - _date("days_between", a).date()).days,
        2, 2, "days_between(start, end) in whole days",
    ),
}

# Planners reach for these Python habits unprompted, so they are accepted as other spellings of the
# documented forms: a keyword for round's second argument, and x.f(...) for a few text functions.
_KEYWORD_POSITION = {"round": {"digits": 1, "ndigits": 1, "decimals": 1}}
_METHODS = frozenset({"lower", "upper", "strip", "startswith", "endswith"})

SYNTAX = (
    "Python-style expression over the row's columns. Bare names are columns; "
    'use col("name with spaces") for names that are not identifiers. '
    "Literals: numbers, 'text', true, false, null. "
    "Operators: + - * / // %, == != < <= > >=, in / not in [list], and / or / not, "
    "x if cond else y. Functions are called as f(x, y). There is no indexing or slicing; "
    "use split_part to take a piece of text."
)


# Python keywords the expression language itself uses; every other keyword may be a column name.
_OPERATOR_WORDS = frozenset({"and", "or", "not", "in", "is", "if", "else", "True", "False", "None"})
_RESERVED = frozenset(keyword.kwlist) - _OPERATOR_WORDS
_ESCAPE = "__column_"


def _escape_reserved(source: str) -> str:
    """Let a column named like a Python keyword (`class`, `pass`, `from`) be written as a bare name."""
    if not any(word in source for word in _RESERVED):
        return source
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return source
    renamed = [(tok.type, _ESCAPE + tok.string if tok.type == tokenize.NAME and tok.string in _RESERVED else tok.string)
               for tok in tokens]
    return tokenize.untokenize(renamed)


class Expr:
    """A parsed, whitelisted expression. `columns` lists the columns it reads."""

    def __init__(self, source: str):
        self.source = source
        if len(source) > MAX_SOURCE:
            raise self._bad(f"expression is longer than {MAX_SOURCE} characters")
        try:
            self._body = ast.parse(_escape_reserved(source.strip()).strip(), mode="eval").body
        except SyntaxError as e:
            raise self._bad(f"syntax error: {e.msg}") from None
        except (RecursionError, ValueError, MemoryError):
            raise self._bad("expression is too deeply nested") from None
        self._columns: dict[str, None] = {}
        self._check(self._body)
        self.columns = list(self._columns)

    def eval(self, row: dict):
        try:
            return self._eval(self._body, row)
        except PlanError as e:
            if not e.message.startswith("in `"):
                e.message = f"in `{self.source}`: {e.message}"
                e.args = (e.message,)
            raise

    def _bad(self, reason: str) -> PlanError:
        return PlanError("bad_expression", f"in `{self.source}`: {reason}")

    def _column_ref(self, node) -> str | None:
        if isinstance(node, ast.Name) and node.id not in _CONSTANTS:
            return node.id.removeprefix(_ESCAPE)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "col"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return node.args[0].value
        return None

    def _call(self, node: ast.Call) -> tuple[str, list]:
        """Resolve a call to (function name, argument nodes in positional order)."""
        if isinstance(node.func, ast.Attribute) and node.func.attr in _METHODS:
            name, args = node.func.attr, [node.func.value, *node.args]
        elif isinstance(node.func, ast.Name):
            name, args = node.func.id, list(node.args)
        else:
            raise self._bad("only plain function calls are allowed")
        if name == "col":
            raise self._bad('col() takes one text literal, like col("unit price")')
        if name not in FUNCTIONS:
            error = unknown("function", name, FUNCTIONS)
            raise PlanError(error.code, f"in `{self.source}`: {error.message}")
        if any(isinstance(a, ast.Starred) for a in args):
            raise self._bad(f"{name}() takes plain arguments only")
        positions = _KEYWORD_POSITION.get(name, {})
        for keyword in node.keywords:
            # A keyword is accepted only where it names the next positional argument.
            if positions.get(keyword.arg) != len(args):
                raise self._bad(f"{name}() does not take the keyword {keyword.arg!r} here; use {FUNCTIONS[name].signature}")
            args.append(keyword.value)
        return name, args

    def _check(self, node) -> None:
        column = self._column_ref(node)
        if column is not None:
            self._columns[column] = None
        elif isinstance(node, ast.Name):
            pass
        elif isinstance(node, ast.Constant):
            if not (node.value is None or isinstance(node.value, (bool, int, float, str))):
                raise self._bad(f"unsupported literal {node.value!r}")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _ARITH:
                raise self._bad(f"unsupported operator {type(node.op).__name__}")
            self._check(node.left)
            self._check(node.right)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
                raise self._bad(f"unsupported operator {type(node.op).__name__}")
            self._check(node.operand)
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                self._check(value)
        elif isinstance(node, ast.Compare):
            self._check(node.left)
            for op, right in zip(node.ops, node.comparators):
                if isinstance(op, _MEMBERSHIP):
                    if not isinstance(right, (ast.List, ast.Tuple)):
                        raise self._bad("`in` needs a literal list on the right, like x in [1, 2]")
                    for element in right.elts:
                        self._check(element)
                elif isinstance(op, _EQUALITY) or type(op) in _ORDER:
                    self._check(right)
                else:
                    raise self._bad(f"unsupported comparison {type(op).__name__}")
        elif isinstance(node, ast.IfExp):
            for part in (node.test, node.body, node.orelse):
                self._check(part)
        elif isinstance(node, ast.Call):
            name, args = self._call(node)
            fn = FUNCTIONS[name]
            if len(args) < fn.min_args or (fn.max_args is not None and len(args) > fn.max_args):
                raise self._bad(f"wrong number of arguments for {fn.signature}")
            for arg in args:
                self._check(arg)
        else:
            raise self._bad(f"unsupported syntax ({type(node).__name__})")

    def _eval(self, node, row: dict):
        column = self._column_ref(node)
        if column is not None:
            return row[column]
        if isinstance(node, ast.Name):
            return _CONSTANTS[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return _arith(type(node.op), self._eval(node.left, row), self._eval(node.right, row))
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand, row)
            if isinstance(node.op, ast.Not):
                return not _truth(value, "not")
            if value is None:
                return None
            if not is_number(value):
                raise PlanError("type_error", f"cannot negate {type_name(value)} {value!r}")
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BoolOp):
            is_and = isinstance(node.op, ast.And)
            for value in node.values:
                if _truth(self._eval(value, row), "and" if is_and else "or") != is_and:
                    return not is_and
            return is_and
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, row)
            for op, right_node in zip(node.ops, node.comparators):
                if isinstance(op, _MEMBERSHIP):
                    right = [self._eval(element, row) for element in right_node.elts]
                else:
                    right = self._eval(right_node, row)
                if not _compare(op, left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            branch = node.body if _truth(self._eval(node.test, row), "if") else node.orelse
            return self._eval(branch, row)
        name, arg_nodes = self._call(node)
        fn = FUNCTIONS[name]
        args = [self._eval(arg, row) for arg in arg_nodes]
        if fn.propagates_null and any(arg is None for arg in args):
            return None
        return fn.call(*args)


def _truth(value, where: str) -> bool:
    if value is None or isinstance(value, bool):
        return bool(value)
    raise PlanError("type_error", f"`{where}` needs true or false, got {type_name(value)} {value!r}")


def _arith(op, left, right):
    if left is None or right is None:
        return None
    fn, symbol = _ARITH[op]
    if not (is_number(left) and is_number(right)):
        raise PlanError(
            "type_error",
            f"`{symbol}` needs numbers, got {type_name(left)} {left!r} and {type_name(right)} {right!r}",
        )
    try:
        return fn(left, right)
    except ZeroDivisionError:
        raise PlanError("division_by_zero", f"division by zero ({left!r} {symbol} {right!r})") from None


def _compare(op, left, right) -> bool:
    if isinstance(op, (ast.Eq, ast.Is)):
        return left == right
    if isinstance(op, (ast.NotEq, ast.IsNot)):
        return left != right
    if isinstance(op, ast.In):
        return left in right
    if isinstance(op, ast.NotIn):
        return left not in right
    if left is None or right is None:
        return False
    fn, symbol = _ORDER[type(op)]
    if (is_number(left) and is_number(right)) or (isinstance(left, str) and isinstance(right, str)):
        return fn(left, right)
    raise PlanError(
        "type_error",
        f"`{symbol}` cannot compare {type_name(left)} {left!r} with {type_name(right)} {right!r}",
    )
