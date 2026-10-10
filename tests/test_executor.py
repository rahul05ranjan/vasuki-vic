import copy

import pytest

from vasuki import PlanError, check_plan, response_schema, run_plan, validate_response
from vasuki.ops import OPS

INVOICES = [
    {"vendor": "Acme", "invoice_number": "A-1", "amount": 1000, "gst_rate": 0.18},
    {"vendor": "Acme", "invoice_number": "A-1", "amount": 1000, "gst_rate": 0.18},
    {"vendor": "Acme", "invoice_number": "A-2", "amount": 500, "gst_rate": 0.18},
    {"vendor": "Bolt", "invoice_number": "B-7", "amount": 200, "gst_rate": 0.05},
]
COLUMNS = ["vendor", "invoice_number", "amount", "gst_rate"]


def plan(*steps):
    return {"status": "plan", "steps": list(steps)}


def error_of(fn, *args):
    with pytest.raises(PlanError) as info:
        fn(*args)
    return info.value


TOTALS = plan(
    {"op": "deduplicate", "keys": ["vendor", "invoice_number"]},
    {"op": "derive", "column": "gst", "expr": "amount * gst_rate"},
    {
        "op": "aggregate",
        "group_by": ["vendor"],
        "metrics": [
            {"as": "amount", "fn": "sum", "column": "amount"},
            {"as": "gst", "fn": "sum", "column": "gst"},
        ],
    },
)


def test_multi_step_plan_runs_and_reports_evidence():
    result = run_plan(TOTALS, {"invoices": INVOICES})
    assert result.columns == ["vendor", "amount", "gst"]
    assert result.rows == [
        {"vendor": "Acme", "amount": 1500, "gst": pytest.approx(270.0)},
        {"vendor": "Bolt", "amount": 200, "gst": pytest.approx(10.0)},
    ]
    assert result.steps == [
        {"op": "deduplicate", "rows_in": 4, "rows_out": 3},
        {"op": "derive", "rows_in": 3, "rows_out": 3},
        {"op": "aggregate", "rows_in": 3, "rows_out": 2},
    ]
    report = result.report(preview=1)
    assert report["status"] == "ok" and report["rows"] == 2 and len(report["preview"]) == 1


def test_inputs_are_not_mutated():
    before = copy.deepcopy(INVOICES)
    run_plan(TOTALS, {"invoices": INVOICES})
    assert INVOICES == before


def test_check_plan_needs_only_column_names():
    assert check_plan(TOTALS, {"invoices": COLUMNS}) == ["vendor", "amount", "gst"]


def test_a_bad_later_step_is_caught_before_any_step_runs():
    bad = plan(
        # This step would fail at run time, so reaching it would raise division_by_zero instead.
        {"op": "derive", "column": "boom", "expr": "amount / 0"},
        {"op": "derive", "column": "gst", "expr": "amount * gst_rat"},
    )
    error = error_of(run_plan, bad, {"invoices": INVOICES})
    assert (error.code, error.step, error.op) == ("unknown_column", 1, "derive")
    assert "'gst_rate'" in error.message


def test_a_column_removed_by_an_earlier_step_is_unknown_later():
    stale = plan(
        {"op": "rename", "columns": {"amount": "total"}},
        {"op": "filter", "where": "amount > 100"},
    )
    error = error_of(check_plan, stale, {"invoices": COLUMNS})
    assert (error.code, error.step) == ("unknown_column", 1)


def test_named_results_feed_later_steps():
    result = run_plan(
        plan(
            {"op": "aggregate", "input": "invoices", "group_by": ["vendor"],
             "metrics": [{"as": "total", "fn": "sum", "column": "amount"}], "as": "totals"},
            {"op": "join", "input": "invoices", "right": "totals", "on": ["vendor"]},
            {"op": "derive", "column": "share", "expr": "amount / total"},
            {"op": "filter", "where": "invoice_number == 'B-7'"},
        ),
        {"invoices": INVOICES},
    )
    assert result.rows[0]["share"] == 1.0


def test_first_step_must_pick_a_table_when_there_are_several():
    tables = {"a": [{"x": 1}], "b": [{"x": 2}]}
    error = error_of(run_plan, plan({"op": "limit", "n": 1}), tables)
    assert (error.code, error.step) == ("missing_field", 0)
    assert run_plan(plan({"op": "limit", "input": "b", "n": 1}), tables).rows == [{"x": 2}]


def test_unknown_table():
    tables = {"invoices": INVOICES}
    assert error_of(run_plan, plan({"op": "limit", "input": "invoice", "n": 1}), tables).code == "unknown_table"


def test_a_step_may_reuse_a_name_and_later_steps_see_the_newest_result():
    result = run_plan(
        plan(
            {"op": "filter", "input": "invoices", "where": "vendor == 'Acme'", "as": "invoices"},
            {"op": "deduplicate", "input": "invoices", "keys": ["invoice_number"], "as": "invoices"},
            {"op": "aggregate", "input": "invoices", "metrics": [{"as": "n", "fn": "count"}]},
        ),
        {"invoices": INVOICES},
    )
    assert result.rows == [{"n": 2}]
    assert len(INVOICES) == 4


def test_rows_with_missing_keys_get_nulls():
    result = run_plan(plan({"op": "filter", "where": "is_null(b)"}), {"t": [{"a": 1}, {"a": 2, "b": 3}]})
    assert result.rows == [{"a": 1, "b": None}]


def test_input_must_be_rows():
    assert error_of(run_plan, TOTALS, {"invoices": {"vendor": "Acme"}}).code == "invalid_input"


@pytest.mark.parametrize(
    "response, code",
    [
        ([], "invalid_plan"),
        ({"steps": []}, "invalid_plan"),
        ({"status": "plan"}, "missing_field"),
        ({"status": "plan", "steps": []}, "invalid_plan"),
        ({"status": "plan", "steps": ["filter"]}, "invalid_plan"),
        ({"status": "plan", "steps": [{"op": "fliter", "where": "a"}]}, "unknown_op"),
        ({"status": "plan", "steps": [{"op": "filter"}]}, "missing_field"),
        ({"status": "plan", "steps": [{"op": "filter", "where": "a", "when": "b"}]}, "invalid_plan"),
        ({"status": "plan", "steps": [{"op": "limit", "n": "3"}]}, "invalid_plan"),
        ({"status": "plan", "steps": [{"op": "limit", "n": True}]}, "invalid_plan"),
        ({"status": "plan", "steps": [{"op": "limit", "n": -1}]}, "invalid_plan"),
        ({"status": "plan", "steps": [{"op": "cast", "column": "a", "to": "date"}]}, "invalid_plan"),
        ({"status": "plan", "steps": [{"op": "select", "columns": []}]}, "invalid_plan"),
        ({"status": "plan", "steps": [{"op": "limit", "n": 1}], "note": "hi"}, "invalid_plan"),
        ({"status": "needs_clarification"}, "missing_field"),
        ({"status": "unsupported", "reason": ""}, "invalid_plan"),
    ],
)
def test_malformed_responses_are_rejected(response, code):
    assert error_of(validate_response, response).code == code


def test_unknown_op_suggests_the_closest_name():
    error = error_of(validate_response, plan({"op": "fliter", "where": "a"}))
    assert error.step == 0 and "'filter'" in error.message


def test_clarification_and_refusal_are_valid_but_not_runnable():
    ask = {"status": "needs_clarification", "question": "Is GST included in amount?", "missing": ["gst_inclusive"]}
    refuse = {"status": "unsupported", "reason": "needs a currency exchange rate"}
    validate_response(ask)
    validate_response(refuse)
    assert error_of(run_plan, ask, {"invoices": INVOICES}).code == "not_a_plan"


def test_response_schema_covers_every_op():
    plan_schema = response_schema()["oneOf"][0]
    step_schemas = plan_schema["properties"]["steps"]["items"]["oneOf"]
    assert [s["properties"]["op"]["const"] for s in step_schemas] == list(OPS)
    assert all(s["additionalProperties"] is False for s in step_schemas)
