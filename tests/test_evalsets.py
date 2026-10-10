"""Integrity checks on the shipped eval sets themselves."""

from pathlib import Path

import pytest

from vasuki import Table, expr, run_plan
from vasuki.evals import compare, load_tasks, score
from vasuki.schema import validate_response

ROOT = Path(__file__).parent.parent / "evals"
SETS = {version: load_tasks(ROOT / version) for version in ("v1", "v2", "test1")}
PLAN_TASKS = [pytest.param(task, id=f"{version}/{task['id']}")
              for version, tasks in SETS.items() for task in tasks if "gold" in task]
ORDERED_TASKS = [param for param in PLAN_TASKS if param.values[0]["expect"]["ordered"]]


@pytest.mark.parametrize("tasks", SETS.values(), ids=SETS)
def test_the_set_is_well_formed(tasks):
    ids = [task["id"] for task in tasks]
    assert len(tasks) >= 150
    assert len(set(ids)) == len(ids)
    assert len({task["request"] for task in tasks}) == len(tasks)
    used = {name for task in tasks for name in task["tables"]}
    for task in tasks:
        assert task["request"].strip() and task["tables"], task["id"]
        assert ("gold" in task) == (task["expect"]["status"] == ["plan"]), task["id"]
    assert used == set().union(*(task["data"].keys() for task in tasks))


@pytest.mark.parametrize("task", PLAN_TASKS)
def test_gold_plan_reproduces_the_frozen_result(task):
    validate_response(task["gold"])
    assert score(task, task["gold"])["outcome"] == "correct"


@pytest.mark.parametrize("task", PLAN_TASKS)
def test_every_gold_step_changes_the_answer(task):
    """If dropping a step still scores as correct, the data cannot tell a right plan from a wrong one."""
    steps = task["gold"]["steps"]
    for i in range(len(steps)):
        remaining = steps[:i] + steps[i + 1:]
        # With no steps left, compare against the untouched first table.
        without = {"status": "plan", "steps": remaining or [{"op": "limit", "input": task["tables"][0], "n": 10**9}]}
        outcome = score(task, without)["outcome"]
        assert outcome != "correct", f"step {i} ({steps[i]['op']}) is redundant on this data"


@pytest.mark.parametrize("tasks", SETS.values(), ids=SETS)
def test_ordered_tasks_end_by_fixing_an_order(tasks):
    for task in tasks:
        if "gold" in task:
            ops = [step["op"] for step in task["gold"]["steps"]]
            assert task["expect"]["ordered"] == ("sort" in ops), task["id"]


def _rerun(task, tables):
    result = run_plan(task["gold"], tables)
    return compare(task["expect"], result.columns, result.rows)


@pytest.mark.parametrize("task", PLAN_TASKS)
def test_answers_do_not_depend_on_the_rounding_convention(task, monkeypatch):
    """A planner that writes code rounds half-to-even; it must not lose marks for that."""
    half_even = expr.FUNCTIONS["round"]._replace(call=lambda x, digits=0: round(x, digits))
    monkeypatch.setitem(expr.FUNCTIONS, "round", half_even)
    assert _rerun(task, task["data"]) == "correct"


@pytest.mark.parametrize("task", ORDERED_TASKS)
def test_ordered_answers_have_no_ties(task):
    """Reversing the input rows must not change an ordered answer, or any stable-sort detail would decide it."""
    reversed_tables = {name: Table(t.columns, t.rows[::-1]) for name, t in task["data"].items()}
    assert _rerun(task, reversed_tables) == "correct"


def test_the_sets_built_from_real_questions_load_and_stay_apart():
    """evals/real1 (test) and evals/real_dev (development) come from Spider; see evals/from_spider.py."""
    test, dev = load_tasks(ROOT / "real1"), load_tasks(ROOT / "real_dev")
    assert len(test) >= 500 and len(dev) >= 200
    for task in test + dev:
        assert task["request"].strip() and 1 <= len(task["data"]) <= 2
        assert task["expect"]["rows"] and set(task["expect"]["rows"][0]) == set(task["expect"]["columns"])
    databases = lambda tasks: {task["tags"][0] for task in tasks}
    assert not databases(test) & databases(dev)
    assert not {task["request"] for task in test} & {task["request"] for task in dev}
    # Spider words most queries twice; the development tasks that have a second wording say so.
    assert 0 < sum("paraphrased" in task["tags"] for task in dev) < len(dev)
    assert not any("paraphrased" in task["tags"] for task in test)


def test_the_set_built_from_a_second_source_loads_and_shares_nothing_with_the_spider_sets():
    """evals/real2 (test) comes from BIRD; see evals/from_bird.py."""
    second, spider = load_tasks(ROOT / "real2"), load_tasks(ROOT / "real1") + load_tasks(ROOT / "real_dev")
    assert len(second) >= 150
    for task in second:
        assert task["request"].strip() and 1 <= len(task["data"]) <= 2
        assert task["expect"]["rows"] and set(task["expect"]["rows"][0]) == set(task["expect"]["columns"])
    assert not {task["tags"][0] for task in second} & {task["tags"][0] for task in spider}
    assert not {task["request"] for task in second} & {task["request"] for task in spider}
