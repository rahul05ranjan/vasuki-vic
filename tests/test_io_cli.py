import json
from pathlib import Path

import pytest

from vasuki import PlanError, load_table, write_table
from vasuki.cli import main

EXAMPLES = Path(__file__).parent.parent / "examples"


def cli(capsys, *argv):
    code = main([str(a) for a in argv])
    return code, json.loads(capsys.readouterr().out)


def test_csv_columns_are_typed_as_a_whole(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(
        "id,zip,qty,price,note\n"
        "1,01234,3,2.50,first\n"
        "2,56789,,4,\n"
        "3,00501,7,1e2,007\n",
        encoding="utf-8",
    )
    table = load_table(path)
    assert table.columns == ["id", "zip", "qty", "price", "note"]
    assert table.rows == [
        {"id": 1, "zip": "01234", "qty": 3, "price": 2.5, "note": "first"},
        {"id": 2, "zip": "56789", "qty": None, "price": 4.0, "note": None},
        {"id": 3, "zip": "00501", "qty": 7, "price": 100.0, "note": "007"},
    ]


def test_csv_with_only_a_header_keeps_its_columns(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("﻿a,b\n", encoding="utf-8")
    table = load_table(path)
    assert (table.columns, table.rows) == (["a", "b"], [])


def test_csv_problems_are_reported(tmp_path):
    wide = tmp_path / "wide.csv"
    wide.write_text("a,b\n1,2,3\n", encoding="utf-8")
    repeated = tmp_path / "repeated.csv"
    repeated.write_text("a,a\n1,2\n", encoding="utf-8")
    for path in (wide, repeated, tmp_path / "missing.csv", tmp_path / "data.xlsx"):
        with pytest.raises(PlanError) as info:
            load_table(path)
        assert info.value.code == "invalid_input"


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".json", ".jsonl"])
def test_tables_round_trip(tmp_path, suffix):
    rows = [{"name": "Asha, Jr.", "n": 1, "x": 1.5}, {"name": "Bo", "n": 2, "x": None}]
    path = tmp_path / f"out{suffix}"
    write_table(path, ["name", "n", "x"], rows)
    assert load_table(path).rows == rows


def test_run_writes_the_result_and_prints_a_compact_report(capsys, tmp_path):
    out = tmp_path / "totals.csv"
    code, report = cli(
        capsys, "run", EXAMPLES / "invoice_totals.json", "-i", EXAMPLES / "invoices.csv", "-o", out, "--preview", 1
    )
    assert code == 0
    assert report["status"] == "ok"
    assert report["columns"] == ["vendor", "invoices", "amount", "gst"]
    assert report["steps"][0] == {"op": "deduplicate", "rows_in": 6, "rows_out": 5}
    assert report["preview"] == [{"vendor": "Acme", "invoices": 2, "amount": 1500.0, "gst": 270.0}]
    assert report["output_path"] == str(out)
    assert load_table(out).rows[2] == {"vendor": "Cora", "invoices": 1, "amount": 0, "gst": 0}


def test_check_reports_output_columns_without_running(capsys):
    code, report = cli(capsys, "check", EXAMPLES / "invoice_totals.json", "-i", f"inv={EXAMPLES / 'invoices.csv'}")
    assert (code, report) == (0, {"status": "ok", "columns": ["vendor", "invoices", "amount", "gst"]})


def test_plan_errors_exit_1_with_a_structured_error(capsys, tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"status": "plan", "steps": [{"op": "filter", "where": "amout > 1"}]}))
    code, report = cli(capsys, "run", plan, "-i", EXAMPLES / "invoices.csv")
    assert code == 1
    assert report["code"] == "unknown_column" and report["step"] == 0 and "'amount'" in report["message"]

    plan.write_text("{not json")
    code, report = cli(capsys, "run", plan, "-i", EXAMPLES / "invoices.csv")
    assert (code, report["code"]) == (1, "invalid_plan")


def test_a_clarification_passes_through_with_its_own_exit_code(capsys, tmp_path):
    response = {"status": "needs_clarification", "question": "Is GST included in amount?"}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(response))
    assert cli(capsys, "run", plan) == (3, response)


def test_schema_command_describes_the_format(capsys):
    code, described = cli(capsys, "schema")
    assert code == 0
    assert set(described) == {"response_schema", "ops", "steps", "expressions", "units"}
    assert "convert_unit" in described["ops"]
    assert "ms" in described["units"]["time"]
