import json

import pytest

from vasuki import PlanError, Table, mcp_server
from vasuki.cli import main
from vasuki.planner import HTTPPlanner, ask, extract_json, full_system_prompt
from vasuki.request import VIC_SYSTEM

ORDERS = [
    {"order_id": 1, "region": "N", "amount": 10.0},
    {"order_id": 2, "region": "S", "amount": 4.5},
    {"order_id": 3, "region": "N", "amount": 2.5},
]
PLAN = {"status": "plan", "steps": [
    {"op": "aggregate", "group_by": ["region"], "metrics": [{"as": "total", "fn": "sum", "column": "amount"}]}]}


class Fake:
    """A planner that gives a canned reply and remembers what it was asked."""

    def __init__(self, reply):
        self.reply, self.seen = reply if isinstance(reply, str) else json.dumps(reply), None

    def complete(self, system, user):
        self.seen = (system, json.loads(user))
        return self.reply


def tables():
    return {"orders": Table.from_rows("orders", ORDERS)}


def test_ask_runs_the_plan_and_returns_a_compact_answer():
    planner = Fake(PLAN)
    answer = ask(planner, "Total amount per region, as total.", tables())
    assert answer == {"status": "ok", "columns": ["region", "total"], "row_count": 2,
                      "rows": [["N", 12.5], ["S", 4.5]], "plan": PLAN["steps"]}
    system, shown = planner.seen
    assert system == VIC_SYSTEM and shown["request"] == "Total amount per region, as total."
    assert shown["tables"]["orders"]["rows"] == 3


def test_ask_truncates_the_preview_and_can_save_everything(tmp_path):
    out = tmp_path / "all.csv"
    answer = ask(Fake({"status": "plan", "steps": [{"op": "sort", "by": [{"column": "amount"}]}]}), "by amount",
                 tables(), preview=1, output=str(out))
    assert answer["row_count"] == 3 and len(answer["rows"]) == 1 and "1 of 3" in answer["truncated"]
    assert answer["output_path"] == str(out) and out.read_text().splitlines()[1].startswith("3,N,2.5")


@pytest.mark.parametrize("reply", [
    {"status": "needs_clarification", "question": "Which rate?"},
    {"status": "unsupported", "reason": "I cannot send email."},
])
def test_ask_passes_a_question_or_refusal_through(reply):
    assert ask(Fake(reply), "anything", tables()) == reply


@pytest.mark.parametrize("reply, code", [
    ("I think you should sum it.", "invalid_response"),
    ({"status": "plan", "steps": [{"op": "filter", "where": "amont > 3"}]}, "unknown_column"),
    ({"status": "plan", "steps": [{"op": "explode"}]}, "unknown_op"),
])
def test_ask_reports_a_reply_that_does_not_run_as_failed(reply, code):
    answer = ask(Fake(reply), "anything", tables())
    assert answer["status"] == "failed" and answer["error"]["code"] == code and answer["reply"]


def test_extract_json_reads_through_fences_and_prose():
    assert extract_json('Sure:\n```json\n{"status": "unsupported", "reason": "x"}\n```') == {
        "status": "unsupported", "reason": "x"}
    assert extract_json("no object here") is None


def test_full_system_prompt_carries_the_format():
    prompt = full_system_prompt()
    assert "convert_unit" in prompt and "days_between" in prompt


def test_http_planner_builds_the_endpoint_and_reports_an_unreachable_server():
    assert HTTPPlanner("http://localhost:11434/v1/", "m").url == "http://localhost:11434/v1/chat/completions"
    assert HTTPPlanner("http://x.test/v1/chat/completions", "m").url == "http://x.test/v1/chat/completions"
    with pytest.raises(PlanError) as raised:
        HTTPPlanner("http://127.0.0.1:9/v1", "m", timeout=2).complete("s", "u")
    assert raised.value.code == "planner_unavailable"


def write_orders(tmp_path):
    path = tmp_path / "orders.csv"
    path.write_text("order_id,region,amount\n1,N,10.0\n2,S,4.5\n3,N,2.5\n")
    return str(path)


def test_cli_ask_prints_the_answer_and_hides_the_plan_unless_asked(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("vasuki.cli._planner", lambda args: (Fake(PLAN), VIC_SYSTEM))
    assert main(["ask", "Total per region, as total.", "-i", write_orders(tmp_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["rows"][0] == ["N", 12.5] and "plan" not in printed
    assert main(["ask", "Total per region, as total.", "-i", write_orders(tmp_path), "--show-plan"]) == 0
    assert json.loads(capsys.readouterr().out)["plan"] == PLAN["steps"]


def test_cli_ask_exit_codes_tell_a_question_from_a_failure(tmp_path, monkeypatch, capsys):
    question = {"status": "needs_clarification", "question": "Which rate?"}
    monkeypatch.setattr("vasuki.cli._planner", lambda args: (Fake(question), VIC_SYSTEM))
    assert main(["ask", "apply the rate", "-i", write_orders(tmp_path)]) == 3
    monkeypatch.setattr("vasuki.cli._planner", lambda args: (Fake("nonsense"), VIC_SYSTEM))
    assert main(["ask", "apply the rate", "-i", write_orders(tmp_path)]) == 1
    capsys.readouterr()
    assert main(["ask", "no inputs"]) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "invalid_input"


def test_mcp_compute_uses_the_planner_and_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "_planner", Fake(PLAN))
    path = write_orders(tmp_path)
    assert mcp_server.compute("Total per region, as total.", [path])["rows"][1] == ["S", 4.5]
    assert mcp_server.compute("x", [f"sales={path}"])["status"] == "ok"
    missing = mcp_server.compute("x", [str(tmp_path / "nope.csv")])
    assert missing["status"] == "error" and missing["code"] == "invalid_input"
    assert mcp_server.compute("x", [])["code"] == "invalid_input"


def test_mcp_run_plan_describe_tables_and_plan_format(tmp_path):
    path = write_orders(tmp_path)
    out = tmp_path / "out.json"
    ran = mcp_server.run_plan(PLAN["steps"], [path], output_path=str(out), preview_rows=1)
    assert ran["row_count"] == 2 and len(ran["rows"]) == 1 and json.loads(out.read_text())[1]["total"] == 4.5
    assert mcp_server.run_plan([{"op": "filter", "where": "nope > 1"}], [path])["code"] == "unknown_column"
    shown = mcp_server.describe_tables([path])["tables"]["orders"]
    assert shown["rows"] == 3 and [c["name"] for c in shown["columns"]] == ["order_id", "region", "amount"]
    assert "ops" in mcp_server.plan_format() or "operations" in mcp_server.plan_format()


def test_mcp_server_registers_its_tools():
    pytest.importorskip("mcp")
    assert [tool.__name__ for tool in mcp_server.TOOLS] == ["compute", "run_plan", "describe_tables", "plan_format"]
    assert all(tool.__doc__ for tool in mcp_server.TOOLS)


def test_csv_true_false_columns_load_as_booleans(tmp_path):
    from vasuki import load_table, run_plan

    path = tmp_path / "answers.csv"
    path.write_text("id,correct,note\n1,True,ok\n2,False,true story\n3,,\n4,true,x\n")
    table = load_table(path)
    assert [row["correct"] for row in table.rows] == [True, False, None, True]
    assert [row["note"] for row in table.rows][:2] == ["ok", "true story"]
    kept = run_plan({"status": "plan", "steps": [{"op": "filter", "where": "correct"}]}, {"answers": table})
    assert [row["id"] for row in kept.rows] == [1, 4]


def test_http_planner_turns_reasoning_off_for_a_vic_model_and_copes_with_a_server_that_refuses_the_field(monkeypatch):
    import io
    import urllib.error

    sent = []

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        sent.append(body)
        if "reasoning_effort" in body and refuse:
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(b""))
        return Reply(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    refuse = False
    assert HTTPPlanner("http://x.test/v1", "m", reasoning=False).complete("s", "u") == "{}"
    assert sent[-1]["reasoning_effort"] == "none"
    assert HTTPPlanner("http://x.test/v1", "m").complete("s", "u") == "{}" and "reasoning_effort" not in sent[-1]
    refuse = True
    assert HTTPPlanner("http://x.test/v1", "m", reasoning=False).complete("s", "u") == "{}"
    assert "reasoning_effort" in sent[-2] and "reasoning_effort" not in sent[-1]
