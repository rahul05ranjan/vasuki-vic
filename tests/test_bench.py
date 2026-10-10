import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pandas")
pytest.importorskip("anthropic")

from bench import providers, run as bench_run
from bench.prompts import CODE_SYSTEM, PLAN_SYSTEM, extract_code, extract_json, user_message
from bench.providers import AnthropicProvider, Completion, FatalProviderError
from bench.scoring import CodeError, judge, run_code
from vasuki.evals import load_tasks

EVALS = Path(__file__).parent.parent / "evals" / "v1"
TASKS = {task["id"]: task for task in load_tasks(EVALS)}


def by_request(start: str) -> dict:
    return next(task for task in TASKS.values() if task["request"].startswith(start))


CANCELLED = by_request("Keep only the cancelled orders")
PER_STATUS = by_request("How many orders are there per status")
RECENT = by_request("Show me the recent orders")
RANK = by_request("Add a column with each employee's salary rank")
FIRST_CLARIFY = by_request("Add the tax to each order")


def fenced(code: str) -> str:
    return f"Here you go:\n```python\n{code}\n```\n"


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"status": "plan", "steps": []}\n```') == {"status": "plan", "steps": []}
    assert extract_json('Sure. {"a": {"b": 1}} Hope that helps {') == {"a": {"b": 1}}
    assert extract_json("no json here {oops") is None


def test_extract_code_wants_a_solve_function():
    assert extract_code(fenced("def solve(tables):\n    return tables['t']")).startswith("def solve")
    assert extract_code("```python\nprint(1)\n```") is None
    assert extract_code("def solve(tables): ...") is None


def test_prompts_carry_the_format_and_never_the_answer():
    assert '"convert_unit"' in PLAN_SYSTEM and "def solve(tables)" in CODE_SYSTEM
    shown = json.loads(user_message(CANCELLED))
    assert set(shown) == {"request", "tables"}
    assert "gold" not in user_message(CANCELLED)


def test_plan_arm_reads_a_plan_out_of_chatty_text():
    reply = 'Plan:\n```json\n{"status":"plan","steps":[{"op":"filter","where":"status == \'cancelled\'"}]}\n```'
    assert judge("plan", CANCELLED, reply)["outcome"] == "correct"
    assert judge("plan", CANCELLED, "I would filter the rows.")["outcome"] == "invalid_response"
    assert judge("plan", CANCELLED, None)["outcome"] == "missing"


def test_code_arm_runs_pandas_and_compares_the_table():
    good = "def solve(tables):\n    df = tables['orders']\n    return df[df['status'] == 'cancelled']"
    assert judge("code", CANCELLED, fenced(good))["outcome"] == "correct"
    wrong = "def solve(tables):\n    return tables['orders']"
    assert judge("code", CANCELLED, fenced(wrong))["outcome"] == "wrong_result"


def test_code_arm_normalises_pandas_types():
    grouped = ("def solve(tables):\n"
               "    out = tables['orders'].groupby('status', sort=False).size().reset_index(name='n')\n"
               "    return out.rename(columns={'n': 'orders'})")
    assert judge("code", PER_STATUS, fenced(grouped))["outcome"] == "correct"
    series = "def solve(tables):\n    return tables['orders'].groupby('status').size().rename('orders')"
    # The index is discarded, so a grouped Series loses its key column.
    assert judge("code", PER_STATUS, fenced(series))["outcome"] == "wrong_result"


def test_code_arm_converts_missing_values_and_timestamps():
    table = TASKS[CANCELLED["id"]]["data"]["orders"]
    code = ("def solve(tables):\n"
            "    df = tables['orders'].head(2).copy()\n"
            "    df['placed_on'] = pd.to_datetime(df['placed_on'])\n"
            "    return df[['placed_on', 'discount_pct', 'items']]")
    columns, rows = run_code(code, {"orders": table})
    assert columns == ["placed_on", "discount_pct", "items"]
    assert rows == [{"placed_on": "2026-01-05", "discount_pct": 10.0, "items": 2},
                    {"placed_on": "2026-01-07", "discount_pct": None, "items": 1}]


def test_code_arm_reports_crashes_timeouts_and_wrong_shapes():
    data = CANCELLED["data"]
    with pytest.raises(CodeError, match="KeyError"):
        run_code("def solve(tables):\n    return tables['nope']", data)
    with pytest.raises(CodeError, match="not a DataFrame"):
        run_code("def solve(tables):\n    return 42", data)
    with pytest.raises(CodeError, match="timed out"):
        run_code("def solve(tables):\n    while True: pass", data, timeout=1)
    crash = "def solve(tables):\n    return tables['nope']"
    assert judge("code", CANCELLED, fenced(crash))["outcome"] == "plan_error"


def test_code_arm_status_replies():
    ask = '{"status": "needs_clarification", "question": "How recent?"}'
    assert judge("code", RECENT, ask)["outcome"] == "correct"
    assert judge("code", CANCELLED, ask)["outcome"] == "wrong_status"
    assert judge("code", RECENT, fenced("def solve(tables):\n    return tables['orders']"))["outcome"] == "wrong_status"
    assert judge("code", CANCELLED, "It depends.")["outcome"] == "invalid_response"
    assert judge("code", CANCELLED, '{"status": "plan", "steps": [{"op": "limit", "n": 1}]}')["outcome"] == "invalid_response"


def test_vic_arm_scores_like_the_plan_arm_with_a_short_prompt():
    from bench.prompts import SYSTEM
    reply = '{"status":"plan","steps":[{"op":"filter","where":"status == \'cancelled\'"}]}'
    assert judge("vic", CANCELLED, reply)["outcome"] == "correct"
    assert len(SYSTEM["vic"]) < 400 < len(SYSTEM["plan"])
    assert RANK in bench_run.select_tasks(list(TASKS.values()), "vic")


def test_code_arm_skips_unsupported_tasks():
    tasks = list(TASKS.values())
    assert RANK in bench_run.select_tasks(tasks, "plan")
    assert RANK not in bench_run.select_tasks(tasks, "code")
    sample = bench_run.select_tasks(tasks, "plan", per_category=2)
    assert len(sample) == 20 and {t["category"] for t in sample} == {t["category"] for t in tasks}


class FakeProvider:
    workers = 2

    def __init__(self, fail_on=()):
        self.calls, self.fail_on = [], set(fail_on)

    def complete(self, system, user):
        request = json.loads(user)["request"]
        self.calls.append(request)
        if request in self.fail_on:
            raise RuntimeError("overloaded")
        reply = {"status": "plan", "steps": [{"op": "filter", "where": "status == 'cancelled'"}]}
        return Completion(json.dumps(reply), "fake-1", "end_turn", 100, 900, 0, 20)


def args(tmp_path, **overrides):
    base = dict(tasks=str(EVALS), out=str(tmp_path), arm="plan", provider="fake", model="fake-1", effort="none",
                no_fallbacks=True, per_category=1, details=True, adapter=None, label=None)
    return SimpleNamespace(**{**base, **overrides})


def test_run_records_every_task_and_resumes_only_what_failed(tmp_path, monkeypatch):
    flaky = FakeProvider(fail_on={FIRST_CLARIFY["request"]})
    monkeypatch.setattr(bench_run, "make_provider", lambda *a, **k: flaky)
    first = bench_run.run(args(tmp_path))
    assert len(flaky.calls) == 10
    assert (first["answered"], first["errors_awaiting_retry"]) == (9, 1)
    # Only basic-001 really is "keep the cancelled orders"; every other canned reply is wrong.
    assert first["correct"] == 1 and first["tokens_per_task"] == {"input_including_system_prompt": 1000.0, "output": 20.0}

    healthy = FakeProvider()
    monkeypatch.setattr(bench_run, "make_provider", lambda *a, **k: healthy)
    second = bench_run.run(args(tmp_path))
    assert healthy.calls == [FIRST_CLARIFY["request"]]
    assert (second["answered"], second["errors_awaiting_retry"]) == (10, 0)


def test_a_fatal_error_stops_the_run(tmp_path, monkeypatch):
    class Broken(FakeProvider):
        def complete(self, system, user):
            self.calls.append(user)
            raise FatalProviderError("AuthenticationError: invalid x-api-key")

    broken = Broken()
    monkeypatch.setattr(bench_run, "make_provider", lambda *a, **k: broken)
    with pytest.raises(FatalProviderError):
        bench_run.run(args(tmp_path))
    assert len(broken.calls) == 1


def test_delegate_and_break_even(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench_run, "make_provider", lambda *a, **k: FakeProvider())
    bench_run.run(args(tmp_path, arm="code", per_category=1))
    assert bench_run.main(["--tasks", str(EVALS), "--out", str(tmp_path), "delegate"]) == 0
    sizes = json.loads(capsys.readouterr().out)
    assert sizes["tasks"] == 250 and sizes["counter"].startswith("estimate")

    assert bench_run.main(["--tasks", str(EVALS), "--out", str(tmp_path), "report"]) == 0
    versus = json.loads(capsys.readouterr().out)["runs"][0]["versus_delegating"]
    assert versus["tasks_compared"] == 9 and versus["unit"].startswith("relative_units")
    assert versus["caller_delegating"] < versus["caller_alone"]
    assert versus["break_even_planner_accuracy"] == pytest.approx(
        versus["caller_delegating"] / versus["caller_alone"], abs=0.01)


def test_anthropic_provider_builds_the_request_and_reads_usage(monkeypatch):
    seen = {}

    def create(**params):
        seen.update(params)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text='{"status":')],
            model="claude-opus-5-5", stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=310, cache_read_input_tokens=2000, cache_creation_input_tokens=None,
                                  output_tokens=140),
        )

    provider = AnthropicProvider("claude-opus-5-5", effort="medium", fallbacks=True)
    provider.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    done = provider.complete("SYSTEM", "USER")
    assert seen["betas"] == [providers.FALLBACK_BETA] and seen["fallbacks"] == "default"
    assert seen["output_config"] == {"effort": "medium"} and "thinking" not in seen
    assert seen["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert seen["messages"] == [{"role": "user", "content": "USER"}]
    assert (done.text, done.input_tokens, done.cache_read_tokens, done.cache_write_tokens, done.output_tokens) == (
        '{"status":', 310, 2000, 0, 140)


def test_anthropic_provider_without_credentials_fails_fast(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", "/nonexistent")
    provider = AnthropicProvider("claude-opus-5-5", fallbacks=False)
    provider.client = provider.client.with_options(max_retries=0)
    with pytest.raises(FatalProviderError):
        provider.count_tokens("hello")


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *args):
        return json.dumps(self.payload).encode()


def http_error(code, body='{"errors":[{"message":"nope"}]}'):
    import io
    import urllib.error
    return urllib.error.HTTPError("https://api.cloudflare.com", code, "error", {}, io.BytesIO(body.encode()))


@pytest.fixture
def cloudflare_env(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok456")
    monkeypatch.setattr(providers.time, "sleep", lambda seconds: None)


def test_cloudflare_provider_builds_the_request_and_reads_usage(monkeypatch, cloudflare_env):
    seen = {}

    def urlopen(request, timeout):
        seen.update(url=request.full_url, auth=request.get_header("Authorization"), body=json.loads(request.data))
        return FakeResponse({"model": "@cf/openai/gpt-oss-120b",
                             "choices": [{"message": {"content": '{"status":"plan"}'}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 2700, "completion_tokens": 90}})

    monkeypatch.setattr(providers.urllib.request, "urlopen", urlopen)
    done = providers.CloudflareProvider("@cf/openai/gpt-oss-120b").complete("SYSTEM", "USER")
    assert seen["url"] == "https://api.cloudflare.com/client/v4/accounts/acct123/ai/v1/chat/completions"
    assert seen["auth"] == "Bearer tok456"
    assert seen["body"]["model"] == "@cf/openai/gpt-oss-120b"
    assert seen["body"]["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]
    assert (done.text, done.stop_reason, done.input_tokens, done.output_tokens) == ('{"status":"plan"}', "stop", 2700, 90)


def test_cloudflare_provider_retries_rate_limits_and_stops_on_a_bad_token(monkeypatch, cloudflare_env):
    replies = [http_error(429), http_error(503), FakeResponse({"choices": [{"message": {"content": None}}]})]

    def flaky(request, timeout):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(providers.urllib.request, "urlopen", flaky)
    done = providers.CloudflareProvider("@cf/x/y").complete("S", "U")
    assert replies == [] and (done.text, done.input_tokens, done.model) == ("", 0, "@cf/x/y")

    def always(code):
        def urlopen(request, timeout):
            raise http_error(code)
        return urlopen

    monkeypatch.setattr(providers.urllib.request, "urlopen", always(401))
    with pytest.raises(FatalProviderError, match="401"):
        providers.CloudflareProvider("@cf/x/y").complete("S", "U")
    monkeypatch.setattr(providers.urllib.request, "urlopen", always(429))
    with pytest.raises(RuntimeError, match="429"):
        providers.CloudflareProvider("@cf/x/y").complete("S", "U")

    def out_of_allowance(request, timeout):
        raise http_error(429, '{"errors":[{"message":"you have used up your daily free allocation of 10,000 neurons"}]}')

    monkeypatch.setattr(providers.urllib.request, "urlopen", out_of_allowance)
    with pytest.raises(FatalProviderError, match="daily allowance"):
        providers.CloudflareProvider("@cf/x/y").complete("S", "U")


def test_cloudflare_provider_needs_credentials(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    with pytest.raises(FatalProviderError, match="CLOUDFLARE_ACCOUNT_ID"):
        providers.CloudflareProvider("@cf/x/y")


def test_env_file_supplies_credentials_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nCLOUDFLARE_ACCOUNT_ID=from_file\nexport CLOUDFLARE_API_TOKEN='quoted'\nKEPT=file\n")
    environ = {"KEPT": "shell"}
    monkeypatch.setattr(bench_run.os, "environ", environ)
    bench_run.load_env_file(env)
    assert environ == {"KEPT": "shell", "CLOUDFLARE_ACCOUNT_ID": "from_file", "CLOUDFLARE_API_TOKEN": "quoted"}
