import pytest

from vasuki import PlanError, run_plan

PEOPLE = [
    {"name": "Asha", "team": "ops", "age": 31, "pay": 50.0},
    {"name": "Bo", "team": "dev", "age": 25, "pay": 70.0},
    {"name": "Cy", "team": "dev", "age": None, "pay": 90.0},
    {"name": "Di", "team": "ops", "age": 44, "pay": None},
]


def run(*steps, tables=None):
    return run_plan({"status": "plan", "steps": list(steps)}, tables or {"people": PEOPLE}).rows


def fails(*steps, tables=None):
    with pytest.raises(PlanError) as info:
        run(*steps, tables=tables)
    return info.value


def names(rows):
    return [row["name"] for row in rows]


def test_filter_drops_false_and_null():
    assert names(run({"op": "filter", "where": "age > 30"})) == ["Asha", "Di"]


def test_filter_requires_a_boolean():
    error = fails({"op": "filter", "where": "age"})
    assert (error.code, error.row) == ("type_error", 0)


def test_derive_adds_and_overwrites():
    rows = run(
        {"op": "derive", "column": "bonus", "expr": "pay * 0.1"},
        {"op": "derive", "column": "pay", "expr": "pay + bonus"},
    )
    assert list(rows[0]) == ["name", "team", "age", "pay", "bonus"]
    assert rows[0]["pay"] == pytest.approx(55.0)
    assert rows[3]["pay"] is None


def test_derive_reports_the_failing_row():
    error = fails({"op": "derive", "column": "x", "expr": "pay / (age - 25)"})
    assert (error.code, error.step, error.row) == ("division_by_zero", 0, 1)


def test_rename_select_drop():
    rows = run(
        {"op": "rename", "columns": {"pay": "salary"}},
        {"op": "select", "columns": ["salary", "name"]},
    )
    assert rows[0] == {"salary": 50.0, "name": "Asha"}
    assert list(run({"op": "drop", "columns": ["age", "pay"]})[0]) == ["name", "team"]


def test_rename_cannot_collide():
    assert fails({"op": "rename", "columns": {"pay": "age"}}).code == "duplicate_column"


def test_cast():
    rows = run({"op": "cast", "column": "n", "to": "int"}, tables={"t": [{"n": "7"}, {"n": None}, {"n": 3.0}]})
    assert [row["n"] for row in rows] == [7, None, 3]
    error = fails({"op": "cast", "column": "n", "to": "int"}, tables={"t": [{"n": "7"}, {"n": "x"}]})
    assert (error.code, error.row) == ("type_error", 1)


def test_sort_is_multi_key_with_nulls_last():
    rows = run({"op": "sort", "by": [{"column": "team"}, {"column": "age", "order": "desc"}]})
    assert names(rows) == ["Bo", "Cy", "Di", "Asha"]
    assert names(run({"op": "sort", "by": [{"column": "age", "order": "desc"}]})) == ["Di", "Asha", "Bo", "Cy"]


def test_sort_rejects_mixed_types():
    error = fails({"op": "sort", "by": [{"column": "v"}]}, tables={"t": [{"v": 1}, {"v": "a"}]})
    assert error.code == "type_error"


def test_deduplicate():
    rows = [{"k": 1, "v": "a"}, {"k": 1, "v": "b"}, {"k": 2, "v": "c"}, {"k": 1, "v": "a"}]
    assert run({"op": "deduplicate"}, tables={"t": rows}) == rows[:3]
    assert run({"op": "deduplicate", "keys": ["k"]}, tables={"t": rows}) == [rows[0], rows[2]]
    assert run({"op": "deduplicate", "keys": ["k"], "keep": "last"}, tables={"t": rows}) == [rows[2], rows[3]]


def test_deduplicate_keeps_true_apart_from_one():
    rows = [{"v": 1}, {"v": True}, {"v": 1.0}]
    assert run({"op": "deduplicate"}, tables={"t": rows}) == [{"v": 1}, {"v": True}]


def test_aggregate_by_group_skips_nulls():
    rows = run({
        "op": "aggregate",
        "group_by": ["team"],
        "metrics": [
            {"as": "people", "fn": "count"},
            {"as": "known_ages", "fn": "count", "column": "age"},
            {"as": "pay", "fn": "sum", "column": "pay"},
            {"as": "avg_age", "fn": "mean", "column": "age"},
            {"as": "oldest", "fn": "max", "column": "age"},
            {"as": "first", "fn": "min", "column": "name"},
        ],
    })
    assert rows == [
        {"team": "ops", "people": 2, "known_ages": 2, "pay": 50.0, "avg_age": 37.5, "oldest": 44, "first": "Asha"},
        {"team": "dev", "people": 2, "known_ages": 1, "pay": 160.0, "avg_age": 25.0, "oldest": 25, "first": "Bo"},
    ]


def test_aggregate_without_groups_gives_one_row_even_when_empty():
    metrics = [
        {"as": "n", "fn": "count"},
        {"as": "total", "fn": "sum", "column": "pay"},
        {"as": "mid", "fn": "median", "column": "pay"},
        {"as": "teams", "fn": "count_distinct", "column": "team"},
    ]
    assert run({"op": "aggregate", "metrics": metrics}) == [{"n": 4, "total": 210.0, "mid": 70.0, "teams": 2}]
    assert run(
        {"op": "filter", "where": "pay > 1000"},
        {"op": "aggregate", "metrics": metrics},
    ) == [{"n": 0, "total": 0, "mid": None, "teams": 0}]


def test_aggregate_sums_integers_exactly():
    rows = run({"op": "aggregate", "metrics": [{"as": "s", "fn": "sum", "column": "v"}]},
               tables={"t": [{"v": 10**20}, {"v": 1}]})
    assert rows == [{"s": 10**20 + 1}]


def test_aggregate_rejects_text_in_a_sum_and_a_metric_without_column():
    error = fails({"op": "aggregate", "metrics": [{"as": "s", "fn": "sum", "column": "name"}]})
    assert error.code == "type_error"
    assert fails({"op": "aggregate", "metrics": [{"as": "s", "fn": "sum"}]}).code == "missing_field"


TEAMS = [{"team": "ops", "lead": "Zed", "name": "Operations"}, {"team": "qa", "lead": "Yan", "name": "Quality"}]


def test_inner_join_suffixes_colliding_columns():
    rows = run(
        {"op": "join", "input": "people", "right": "teams", "on": ["team"]},
        tables={"people": PEOPLE, "teams": TEAMS},
    )
    assert names(rows) == ["Asha", "Di"]
    assert list(rows[0]) == ["name", "team", "age", "pay", "lead", "name_right"]
    assert rows[0]["name_right"] == "Operations"


def test_left_join_keeps_unmatched_rows():
    rows = run(
        {"op": "join", "input": "people", "right": "teams", "on": ["team"], "how": "left"},
        tables={"people": PEOPLE, "teams": TEAMS},
    )
    assert names(rows) == ["Asha", "Bo", "Cy", "Di"]
    assert rows[1]["lead"] is None


def test_join_on_differently_named_keys_multiplies_matches_and_skips_null_keys():
    orders = [{"id": 1, "cust": "a"}, {"id": 2, "cust": None}, {"id": 3, "cust": "b"}]
    contacts = [{"customer": "a", "email": "a1"}, {"customer": "a", "email": "a2"}, {"customer": None, "email": "n"}]
    rows = run(
        {"op": "join", "input": "orders", "right": "contacts", "left_on": ["cust"], "right_on": ["customer"]},
        tables={"orders": orders, "contacts": contacts},
    )
    assert rows == [{"id": 1, "cust": "a", "email": "a1"}, {"id": 1, "cust": "a", "email": "a2"}]


def test_join_matches_a_number_key_with_the_same_number_held_as_text():
    budget = [{"school": 2, "amount": 10}, {"school": 3, "amount": 5}, {"school": 9, "amount": 1}]
    schools = [{"school": "2", "name": "Glenn"}, {"school": " 3.0", "name": "Bremen"}, {"school": "n/a", "name": "?"}]
    rows = run({"op": "join", "input": "budget", "right": "schools", "on": ["school"]},
               tables={"budget": budget, "schools": schools})
    assert rows == [{"school": 2, "amount": 10, "name": "Glenn"}, {"school": 3, "amount": 5, "name": "Bremen"}]
    rows = run({"op": "join", "input": "schools", "right": "budget", "on": ["school"]},
               tables={"budget": budget, "schools": schools})
    assert [(row["school"], row["amount"]) for row in rows] == [("2", 10), (" 3.0", 5)]


def test_join_keeps_text_keys_apart_when_both_sides_are_text():
    left = [{"code": "007"}, {"code": "7"}]
    right = [{"code": "7", "who": "seven"}]
    rows = run({"op": "join", "input": "left", "right": "right", "on": ["code"]}, tables={"left": left, "right": right})
    assert rows == [{"code": "7", "who": "seven"}]


def test_join_needs_keys_and_a_known_table():
    tables = {"people": PEOPLE, "teams": TEAMS}
    assert fails({"op": "join", "input": "people", "right": "teams"}, tables=tables).code == "missing_field"
    error = fails({"op": "join", "input": "people", "right": "team", "on": ["team"]}, tables=tables)
    assert error.code == "unknown_table" and "'teams'" in error.message


def test_convert_unit_in_place_and_to_a_new_column():
    configs = [{"timeout": 30}, {"timeout": 0.3}, {"timeout": None}]
    rows = run(
        {"op": "convert_unit", "column": "timeout", "from": "s", "to": "ms", "to_column": "timeout_ms"},
        {"op": "drop", "columns": ["timeout"]},
        tables={"configs": configs},
    )
    assert rows == [{"timeout_ms": 30000}, {"timeout_ms": 300.0}, {"timeout_ms": None}]
    assert isinstance(rows[0]["timeout_ms"], int)


def test_convert_unit_handles_offsets_and_case():
    rows = run({"op": "convert_unit", "column": "t", "from": "c", "to": "F"}, tables={"t": [{"t": 100}, {"t": -40}]})
    assert rows == [{"t": 212}, {"t": -40}]
    rows = run({"op": "convert_unit", "column": "r", "from": "percent", "to": "fraction"}, tables={"t": [{"r": 18}]})
    assert rows == [{"r": 0.18}]


def test_convert_unit_rejects_mismatched_and_unknown_units_before_running():
    tables = {"t": [{"v": "not a number"}]}
    assert fails({"op": "convert_unit", "column": "v", "from": "kg", "to": "km"}, tables=tables).code == "unit_error"
    error = fails({"op": "convert_unit", "column": "v", "from": "usd", "to": "inr"}, tables=tables)
    assert error.code == "unit_error"
    assert fails({"op": "convert_unit", "column": "v", "from": "kg", "to": "g"}, tables=tables).code == "type_error"


def test_limit():
    assert names(run({"op": "limit", "n": 2})) == ["Asha", "Bo"]
    assert run({"op": "limit", "n": 0}) == []
