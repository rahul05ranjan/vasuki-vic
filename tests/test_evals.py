import json

import pytest

from vasuki import Table
from vasuki.cli import main
from vasuki.evals import compare, score, summarize, task_input
from vasuki.request import profile_table

ROWS = [{"team": "ops", "total": 3}, {"team": "dev", "total": 4.5}, {"team": None, "total": None}]


def expect(ordered=False):
    return {"status": ["plan"], "ordered": ordered, "columns": ["team", "total"], "rows": ROWS}


def task(**overrides):
    base = {
        "id": "t-1",
        "category": "demo",
        "tags": ["steps:1"],
        "request": "Total per team.",
        "tables": ["sales"],
        "data": {"sales": Table.from_rows("sales", [
            {"team": "ops", "n": 1}, {"team": "ops", "n": 2}, {"team": "dev", "n": 4.5}, {"team": None, "n": None},
        ])},
        "expect": expect(),
    }
    return {**base, **overrides}


def total_plan(name="total"):
    metric = {"as": name, "fn": "sum", "column": "n"}
    steps = [
        {"op": "aggregate", "group_by": ["team"], "metrics": [metric]},
        # Sum of an all-null group is 0; the expected table has null there.
        {"op": "derive", "column": name, "expr": f"null if is_null(team) else {name}"},
    ]
    return {"status": "plan", "steps": steps}


def test_compare_ignores_row_and_column_order():
    shuffled = [{"total": 4.5, "team": "dev"}, {"total": None, "team": None}, {"total": 3, "team": "ops"}]
    assert compare(expect(), ["total", "team"], shuffled) == "correct"


def test_compare_respects_row_order_when_the_task_is_ordered():
    assert compare(expect(ordered=True), ["team", "total"], ROWS) == "correct"
    assert compare(expect(ordered=True), ["team", "total"], ROWS[::-1]) == "wrong_result"


def test_compare_tolerates_float_noise_and_int_float_but_not_real_differences():
    noisy = [{"team": "ops", "total": 3.0000000000004}, {"team": "dev", "total": 4.5}, {"team": None, "total": None}]
    assert compare(expect(), ["team", "total"], noisy) == "correct"
    off = [{"team": "ops", "total": 3.01}, {"team": "dev", "total": 4.5}, {"team": None, "total": None}]
    assert compare(expect(), ["team", "total"], off) == "wrong_result"


def test_compare_keeps_types_apart():
    single = {"status": ["plan"], "columns": ["v"], "rows": [{"v": 1}]}
    assert compare(single, ["v"], [{"v": 1.0}]) == "correct"
    assert compare(single, ["v"], [{"v": True}]) == "wrong_result"
    assert compare(single, ["v"], [{"v": "1"}]) == "wrong_result"
    assert compare(single, ["v"], [{"v": None}]) == "wrong_result"


def test_compare_reports_right_values_under_other_names():
    renamed = [{"grp": row["team"], "sum": row["total"]} for row in ROWS]
    assert compare(expect(), ["grp", "sum"], renamed) == "values_only"
    assert compare(expect(), ["grp", "sum"], renamed[:2]) == "wrong_result"


def test_compare_rejects_extra_or_missing_columns():
    wide = [{**row, "extra": 1} for row in ROWS]
    assert compare(expect(), ["team", "total", "extra"], wide) == "wrong_result"
    assert compare(expect(), ["team"], [{"team": row["team"]} for row in ROWS]) == "wrong_result"


def test_score_runs_the_plan_and_judges_the_result():
    assert score(task(), total_plan())["outcome"] == "correct"
    assert score(task(), json.dumps(total_plan()))["outcome"] == "correct"
    assert score(task(), total_plan("sum_n"))["outcome"] == "values_only"
    wrong = {"status": "plan", "steps": [{"op": "aggregate", "group_by": ["team"],
                                          "metrics": [{"as": "total", "fn": "count"}]}]}
    assert score(task(), wrong)["outcome"] == "wrong_result"


def test_score_classifies_failures():
    bad_column = {"status": "plan", "steps": [{"op": "filter", "where": "amount > 1"}]}
    result = score(task(), bad_column)
    assert result["outcome"] == "plan_error" and result["detail"].startswith("unknown_column")
    assert score(task(), {"status": "plan", "steps": []})["outcome"] == "invalid_response"
    assert score(task(), "I think you should sum it")["outcome"] == "invalid_response"
    assert score(task(), {"status": "unsupported", "reason": "no"})["outcome"] == "wrong_status"
    assert score(task(), None)["outcome"] == "missing"


def test_score_accepts_any_listed_status_for_non_plan_tasks():
    ambiguous = task(expect={"status": ["needs_clarification", "unsupported"]})
    assert score(ambiguous, {"status": "needs_clarification", "question": "Which?"})["outcome"] == "correct"
    assert score(ambiguous, {"status": "unsupported", "reason": "no rate"})["outcome"] == "correct"
    assert score(ambiguous, total_plan())["outcome"] == "wrong_status"


def test_summarize_breaks_down_by_category_and_tag():
    results = [
        {"id": "a", "category": "x", "tags": ["steps:1"], "outcome": "correct"},
        {"id": "b", "category": "x", "tags": ["steps:1", "hinglish"], "outcome": "values_only"},
        {"id": "c", "category": "y", "tags": [], "outcome": "plan_error"},
        {"id": "d", "category": "y", "tags": ["hinglish"], "outcome": "correct"},
    ]
    summary = summarize(results)
    assert (summary["tasks"], summary["correct"], summary["accuracy"]) == (4, 2, 0.5)
    assert summary["accuracy_ignoring_column_names"] == 0.75
    assert summary["outcomes"] == {"correct": 2, "values_only": 1, "plan_error": 1}
    assert summary["by_category"]["x"] == {"tasks": 2, "correct": 1, "accuracy": 0.5}
    assert summary["by_tag"]["hinglish"] == {"tasks": 2, "correct": 1, "accuracy": 0.5}


def test_profile_shows_types_nulls_and_low_cardinality_values():
    table = Table.from_rows("t", [
        {"id": 1, "status": "open", "price": 2.5, "note": "a", "flag": True, "gone": None},
        {"id": 2, "status": "done", "price": 3, "note": "b", "flag": False, "gone": None},
        {"id": 3, "status": "open", "price": None, "note": "c", "flag": True, "gone": None},
    ])
    profile = profile_table(table, sample=1, max_values=2)
    assert profile["rows"] == 3 and profile["sample"] == table.rows[:1]
    assert profile["columns"] == [
        {"name": "id", "type": "int"},
        {"name": "status", "type": "str", "values": ["open", "done"]},
        {"name": "price", "type": "float", "nulls": 1},
        {"name": "note", "type": "str"},
        {"name": "flag", "type": "bool"},
        {"name": "gone", "type": "null", "nulls": 3},
    ]


def test_task_input_never_leaks_the_answer():
    shown = task_input(task(gold=total_plan()))
    assert set(shown) == {"id", "request", "tables"}
    assert "gold" not in json.dumps(shown) and "expect" not in json.dumps(shown)


@pytest.fixture
def eval_dir(tmp_path):
    rows = [{"team": "ops", "n": 1}, {"team": "ops", "n": 2}, {"team": "dev", "n": 4}]
    gold = {"status": "plan", "steps": [{"op": "filter", "where": "n > 1"}]}
    tasks = [
        {"id": "p-1", "category": "basic", "tags": [], "request": "n over 1", "tables": ["sales"], "gold": gold,
         "expect": {"status": ["plan"], "ordered": False, "columns": ["team", "n"], "rows": rows[1:]}},
        {"id": "q-1", "category": "clarify", "tags": [], "request": "recent ones", "tables": ["sales"],
         "expect": {"status": ["needs_clarification"]}},
    ]
    (tmp_path / "tables.json").write_text(json.dumps({"sales": rows}))
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    return tmp_path


def test_eval_command_scores_gold_and_predictions(capsys, eval_dir):
    assert main(["eval", str(eval_dir), "--gold"]) == 0
    assert json.loads(capsys.readouterr().out)["accuracy"] == 1.0

    predictions = eval_dir / "predictions.jsonl"
    predictions.write_text(json.dumps({"id": "p-1", "response": {"status": "unsupported", "reason": "x"}}) + "\n")
    assert main(["eval", str(eval_dir), str(predictions), "--details"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["accuracy"] == 0.0
    assert [f["outcome"] for f in report["failures"]] == ["wrong_status", "missing"]

    assert main(["eval", str(eval_dir)]) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "invalid_input"


def test_eval_inputs_command_prints_one_line_per_task(capsys, eval_dir):
    assert main(["eval-inputs", str(eval_dir)]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["id"] for line in lines] == ["p-1", "q-1"]
    assert lines[0]["tables"]["sales"]["rows"] == 3
