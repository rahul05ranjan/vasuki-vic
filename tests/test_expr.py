import pytest

from vasuki.errors import PlanError
from vasuki.expr import Expr


def ev(source, **row):
    return Expr(source).eval(row)


def error_code(source, **row):
    with pytest.raises(PlanError) as info:
        Expr(source).eval(row)
    return info.value.code


def test_arithmetic_follows_precedence():
    assert ev("a + b * 2", a=1, b=3) == 7
    assert ev("(a + b) * 2", a=1, b=3) == 8
    assert ev("-a % 3", a=4) == 2
    assert ev("a // 2", a=7) == 3


def test_null_propagates_through_arithmetic_and_functions():
    assert ev("a * 2", a=None) is None
    assert ev("upper(a)", a=None) is None
    assert ev("-a", a=None) is None


def test_arithmetic_rejects_text():
    assert error_code("a * 2", a="12") == "type_error"
    assert error_code("a + b", a="x", b="y") == "type_error"


def test_division_by_zero_is_reported():
    assert error_code("a / b", a=1, b=0) == "division_by_zero"


def test_comparisons():
    assert ev("0 < a <= 10", a=10) is True
    assert ev("0 < a <= 10", a=11) is False
    assert ev("a == 'x'", a="x") is True
    assert ev("a != null", a=None) is False
    assert ev("a < 'b'", a="a") is True


def test_ordering_against_null_is_false():
    assert ev("a > 0", a=None) is False
    assert ev("a <= 0", a=None) is False


def test_ordering_mixed_types_is_an_error():
    assert error_code("a > 0", a="5") == "type_error"


def test_boolean_logic_treats_null_as_false_and_short_circuits():
    assert ev("a > 1 and b > 1", a=2, b=None) is False
    assert ev("not (a > 1)", a=None) is True
    assert ev("a == 1 or b / 0 > 1", a=1, b=1) is True


def test_boolean_logic_rejects_non_booleans():
    assert error_code("a and true", a=1) == "type_error"
    assert error_code("1 if a else 2", a="yes") == "type_error"


def test_membership_and_conditional():
    assert ev("a in ['x', 'y']", a="y") is True
    assert ev("a not in [1, 2]", a=3) is True
    assert ev("'big' if a > 100 else 'small'", a=500) == "big"


def test_json_style_constants():
    assert ev("true") is True
    assert ev("false") is False
    assert ev("null") is None


def test_round_is_half_up_on_the_written_decimal():
    assert ev("round(a, 2)", a=2.675) == 2.68
    assert ev("round(a)", a=2.5) == 3
    assert ev("round(a)", a=3.5) == 4
    assert ev("round(a, -1)", a=1235) == 1240
    assert ev("round(a, 2)", a=7) == 7


def test_text_and_null_functions():
    assert ev("concat(lower(a), '-', b)", a="AB", b=3) == "ab-3"
    assert ev("contains(a, 'ell')", a="hello") is True
    assert ev("coalesce(a, b, 0)", a=None, b=None) == 0
    assert ev("is_null(a)", a=None) is True
    assert ev("len(strip(a))", a="  hi ") == 2


def test_conversion_functions():
    assert ev("int(a) + 1", a="41") == 42
    assert ev("float(a)", a="1.5") == 1.5
    assert ev("str(a)", a=True) == "true"
    assert error_code("int(a)", a="1.5") == "type_error"


def test_date_functions():
    assert ev("year(d)", d="2026-11-12") == 2026
    assert ev("month(d)", d="2026-11-12T08:30:00") == 11
    assert ev("days_between(a, b)", a="2026-11-12", b="2026-11-15") == 3
    assert error_code("year(d)", d="12/11/2026") == "type_error"


def test_python_habits_are_accepted_as_other_spellings():
    assert ev("round(a, digits=2)", a=2.675) == 2.68
    assert ev("round(a, ndigits=1)", a=0.25) == 0.3
    assert ev("a.lower().startswith('ab')", a="ABC") is True
    assert ev("a.strip().upper()", a=" x ") == "X"
    assert ev("a.endswith('z')", a=None) is None
    assert ev("a is null", a=None) is True
    assert ev("a is not null and a > 1", a=5) is True
    assert Expr("b.startswith(col('a b'))").columns == ["b", "a b"]


def test_columns_named_like_python_keywords_can_be_written_bare():
    expr = Expr("round(pass / total, 2) if class == 'A' else 0")
    assert sorted(expr.columns) == ["class", "pass", "total"]
    assert expr.eval({"pass": 2, "total": 3, "class": "A"}) == 0.67
    assert ev("from < return and 'pass' in [note]", **{"from": 1, "return": 2, "note": "pass"}) is True
    assert ev("col('import') + 1", **{"import": 4}) == 5
    assert error_code("pass / 0", **{"pass": 1}) == "division_by_zero"


def test_time_and_text_pieces():
    assert ev("hour(t)", t="2026-09-01T18:45:00") == 18
    assert ev("minute(t)", t="2026-09-01T18:45:00") == 45
    assert ev("hour(t)", t="2026-09-01") == 0
    assert ev("split_part(e, '@', 2)", e="raj@kite.dev") == "kite.dev"
    assert ev("split_part(e, '@', 3)", e="raj@kite.dev") is None
    assert ev("split_part(e, '@', 1)", e=None) is None
    assert error_code("split_part(e, '@', 0)", e="a@b") == "type_error"
    assert error_code("split_part(e, '', 1)", e="a@b") == "type_error"


def test_col_reads_names_that_are_not_identifiers():
    expr = Expr('col("unit price") * qty')
    assert expr.columns == ["unit price", "qty"]
    assert expr.eval({"unit price": 2.5, "qty": 4}) == 10


def test_columns_are_listed_once_in_order():
    assert Expr("b + a + b + null").columns == ["b", "a"]


def test_errors_name_the_expression():
    with pytest.raises(PlanError) as info:
        ev("a * 2", a="12")
    assert "`a * 2`" in info.value.message


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('true')",
        "open('/etc/passwd')",
        "a.b",
        "a.__class__",
        "a[0]",
        "lambda: 1",
        "[x for x in a]",
        "a ** b",
        "f'{a}'",
        "a := 1",
        "{'k': a}",
        "round(digits=2)",
        "round(a, 1, digits=2)",
        "round(a, places=2)",
        "lower(text=a)",
        "a.split('@')",
        "a.lower",
        "a in b",
        "col(a)",
        "a +",
        "",
    ],
)
def test_anything_outside_the_whitelist_is_rejected_at_parse_time(source):
    with pytest.raises(PlanError) as info:
        Expr(source)
    assert info.value.code in ("bad_expression", "unknown_function")


def test_unknown_function_suggests_the_closest_name():
    with pytest.raises(PlanError) as info:
        Expr("roundd(a)")
    assert info.value.code == "unknown_function"
    assert "'round'" in info.value.message
