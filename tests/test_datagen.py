import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pandas")

from datagen import run as datagen_run
from datagen.pipeline import (Rejected, build_world, extract_examples, make_world, rewrite_world, simplify_plan,
                              verify_example)
from datagen.prompts import AUTHOR_SYSTEM, FOCUS, recipes, world_prompt, world_spec
from datagen.topics import BANNED_TABLES, TOPICS
from vasuki import Table
from vasuki.evals import load_tasks

VISITS = [
    {"visit_id": 1, "member": "ana", "plan": "gold", "minutes": 50, "fee": 12.5, "day": "2026-03-01"},
    {"visit_id": 2, "member": "raj", "plan": "basic", "minutes": 35, "fee": 8.0, "day": "2026-03-02"},
    {"visit_id": 3, "member": "ana", "plan": "gold", "minutes": 70, "fee": 12.5, "day": "2026-03-04"},
    {"visit_id": 4, "member": "mei", "plan": "basic", "minutes": None, "fee": 8.0, "day": "2026-03-05"},
    {"visit_id": 5, "member": "raj", "plan": "basic", "minutes": 20, "fee": 9.5, "day": "2026-03-07"},
    {"visit_id": 6, "member": "zoe", "plan": "gold", "minutes": 90, "fee": 14.0, "day": "2026-03-09"},
]
SPEC = {"id": "w00000", "topic": "gym member visits", "traits": []}
WORLD = {**SPEC, "tables": {"visits": VISITS}}


def completion(text):
    return SimpleNamespace(text=text, model="scripted", input_tokens=100, cache_read_tokens=0, cache_write_tokens=0,
                           output_tokens=40)


class Scripted:
    """Answers with a fixed reply, or with whatever a function of the prompt returns."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, system, user):
        self.prompts.append((system, user))
        return completion(self.reply(user) if callable(self.reply) else self.reply)


def code(body):
    return f"```python\ndef solve(tables):\n    df = tables['visits']\n    {body}\n```"


GOLD_PLAN = {"status": "plan", "steps": [{"op": "filter", "input": "visits", "as": "gold", "where": "plan == 'gold'"},
                                         {"op": "select", "input": "gold", "as": "out", "columns": ["visit_id", "fee"]}]}
PLAN_RECIPE = {"n": 1, "kind": "plan", "topic": "basic", "about": "filter", "style": "plain"}
PLAN_EXAMPLE = {"n": 1, "request": "Gold plan visits: return visit_id and fee.", "response": GOLD_PLAN}


def test_specs_and_recipes_are_reproducible_and_fit_the_world():
    assert world_spec(1, 7) == world_spec(1, 7) != world_spec(2, 7)
    assert world_spec(1, 7)["topic"] in TOPICS
    plain = recipes(1, SPEC, 30)
    assert plain == recipes(1, SPEC, 30)
    assert {r["topic"] for r in plain}.isdisjoint({"join", "units", "text_numbers", "messy_case", "duplicates"})
    assert all(r["about"] in FOCUS[r["topic"]] for r in plain)
    assert {r["kind"] for r in plain} <= {"plan", "clarify", "unsupported"}
    quirky = recipes(1, {**SPEC, "traits": ["two_tables", "units", "messy_case", "duplicates"]}, 30)
    assert {"messy_case", "duplicates"} <= {r["topic"] for r in quirky}


def test_focus_profile_leans_on_joins_units_and_rare_operations():
    spec = world_spec(1, 3, "focus")
    assert spec["id"] == "f00003" and {"two_tables", "units"} <= set(spec["traits"])
    assert spec != world_spec(1, 3) and world_spec(1, 3)["id"] == "w00003"
    topics = [r["topic"] for i in range(40) for r in recipes(1, world_spec(1, i, "focus"), 10)]
    share = {topic: topics.count(topic) / len(topics) for topic in ("join", "units", "rare_ops")}
    assert share["join"] > 0.2 and share["units"] > 0.12 and share["rare_ops"] > 0.08


def test_target_profile_leans_on_what_the_last_model_got_wrong():
    specs = [world_spec(1, i, "target") for i in range(60)]
    assert specs[3]["id"] == "t00003" and specs[3] != world_spec(1, 3, "focus")
    traits = [trait for spec in specs for trait in spec["traits"]]
    assert traits.count("sparse_numbers") > 30 and traits.count("state_dates") > 30
    # The two new traits never appear in the earlier profiles, so those still draw the same worlds.
    assert not {"sparse_numbers", "state_dates"} & {t for i in range(60) for p in ("broad", "focus")
                                                    for t in world_spec(1, i, p)["traits"]}
    batch = [r for spec in specs for r in recipes(1, spec, 10)]
    assert all(r["about"] in FOCUS[r["topic"]] for r in batch)
    share = {topic: sum(r["topic"] == topic for r in batch) / len(batch) for topic in ("long", "null_zero", "clarify")}
    assert share["long"] > 0.12 and share["null_zero"] > 0.09 and share["clarify"] > 0.06


def test_world_prompt_states_traits_and_banned_names():
    prompt = world_prompt({**SPEC, "traits": ["two_tables", "text_numbers"]})
    assert "gym member visits" in prompt and "share one key column" in prompt and "as text" in prompt
    assert all(name in prompt for name in BANNED_TABLES)
    assert "Make one table." in world_prompt(SPEC)
    assert '"split_part' in AUTHOR_SYSTEM or "split_part" in AUTHOR_SYSTEM


def test_build_world_accepts_good_tables_and_rejects_bad_ones():
    good = json.dumps({"tables": {"visits": VISITS}})
    assert build_world(SPEC, f"```json\n{good}\n```")["tables"]["visits"] == VISITS
    bad = {
        "not json": "sorry",
        "wrong count": json.dumps({"tables": {"visits": VISITS, "members": VISITS}}),
        "banned name": json.dumps({"tables": {"orders": VISITS}}),
        "too few rows": json.dumps({"tables": {"visits": VISITS[:3]}}),
        "nested value": json.dumps({"tables": {"visits": [{**row, "tags": ["a"]} for row in VISITS]}}),
        "bad column": json.dumps({"tables": {"visits": [{**row, "Fee Paid": 1} for row in VISITS]}}),
    }
    for text in bad.values():
        with pytest.raises(Rejected):
            build_world(SPEC, text)
    with pytest.raises(Rejected, match="share no column"):
        other = [{"a": i, "b": "x", "c": 1.5, "d": "2026-01-01"} for i in range(6)]
        build_world({**SPEC, "traits": ["two_tables"]}, json.dumps({"tables": {"visits": VISITS, "other": other}}))


def test_simplify_plan_drops_needless_names_and_keeps_needed_ones():
    tables = {"visits": Table.from_rows("visits", VISITS)}
    assert simplify_plan(GOLD_PLAN, tables)["steps"] == [
        {"op": "filter", "where": "plan == 'gold'"}, {"op": "select", "columns": ["visit_id", "fee"]}]
    needed = {"status": "plan", "steps": [
        {"op": "aggregate", "input": "visits", "group_by": ["member"], "metrics": [{"as": "n", "fn": "count"}], "as": "counts"},
        {"op": "join", "input": "visits", "right": "counts", "on": ["member"], "as": "unused"}]}
    steps = simplify_plan(needed, tables)["steps"]
    assert steps[0]["as"] == "counts" and "input" not in steps[0]
    assert steps[1]["input"] == "visits" and "as" not in steps[1]


def test_a_plan_is_kept_only_when_independent_code_agrees():
    agrees = Scripted(code("return df[df['plan'] == 'gold'][['visit_id', 'fee']]"))
    kept = verify_example(WORLD, PLAN_RECIPE, PLAN_EXAMPLE, agrees)
    assert kept["verified_by"].startswith("second model's code")
    assert kept["response"]["steps"][0] == {"op": "filter", "where": "plan == 'gold'"}
    shown = json.loads(agrees.prompts[0][1])
    assert shown["request"] == PLAN_EXAMPLE["request"] and "response" not in agrees.prompts[0][1]

    for reply, why in ((code("return df[df['plan'] == 'basic'][['visit_id', 'fee']]"), "differs"),
                       (code("return df[df['plan'] == 'gold'][['visit_id', 'fee']].rename(columns={'fee': 'f'})"), "values_only"),
                       (code("return df['nope']"), "code failed"),
                       ("I would filter the rows.", "wrote no code")):
        with pytest.raises(Rejected, match=why):
            verify_example(WORLD, PLAN_RECIPE, PLAN_EXAMPLE, Scripted(reply))


def test_bad_authoring_is_rejected_before_any_verifier_call():
    never = Scripted("unused")
    cases = (
        ({**PLAN_EXAMPLE, "request": ""}, "no request"),
        ({**PLAN_EXAMPLE, "response": {"status": "plan", "steps": [{"op": "fliter", "where": "x"}]}}, "invalid response"),
        ({**PLAN_EXAMPLE, "response": {"status": "unsupported", "reason": "no"}}, "recipe wanted plan"),
        ({**PLAN_EXAMPLE, "response": {"status": "plan", "steps": [{"op": "filter", "where": "nope > 1"}]}}, "plan fails"),
        ({**PLAN_EXAMPLE, "response": {"status": "plan", "steps": [{"op": "filter", "where": "fee > 999"}]}}, "no rows"),
    )
    for example, why in cases:
        with pytest.raises(Rejected, match=why):
            verify_example(WORLD, PLAN_RECIPE, example, never)
    assert never.prompts == []


def test_clarifications_need_a_second_opinion_and_refusals_stand_by_construction():
    ask_recipe = {"n": 2, "kind": "clarify", "topic": "clarify", "about": "a rate", "style": "terse"}
    ask = {"n": 2, "request": "Add the service tax to each fee.",
           "response": {"status": "needs_clarification", "question": "What tax rate?"}}
    also_asks = Scripted('{"status": "needs_clarification", "question": "Which rate?"}')
    assert verify_example(WORLD, ask_recipe, ask, also_asks)["verified_by"] == "second model also asked"
    with pytest.raises(Rejected, match="instead of asking"):
        verify_example(WORLD, ask_recipe, ask, Scripted(code("return df")))

    no_recipe = {"n": 3, "kind": "unsupported", "topic": "unsupported", "about": "a chart", "style": "plain"}
    no = {"n": 3, "request": "Plot minutes per member as a bar chart.",
          "response": {"status": "unsupported", "reason": "Charts are not available."}}
    never = Scripted("unused")
    assert verify_example(WORLD, no_recipe, no, never)["verified_by"] == "construction"
    assert never.prompts == []


def test_one_malformed_example_does_not_cost_the_others():
    good = [{"n": 1, "request": "first", "response": GOLD_PLAN}, {"n": 3, "request": "third", "response": GOLD_PLAN}]
    broken = '{"n": 2, "request": "an "unescaped" quote", "response": {}}'
    text = '```json\n{"examples": [' + json.dumps(good[0]) + ", " + broken + ", " + json.dumps(good[1]) + "]}\n```"
    assert [e["n"] for e in extract_examples(text)] == [1, 3]
    assert extract_examples(json.dumps({"examples": good})) == good
    assert extract_examples("no examples here {\"n\": 5}") == []


def test_rewrites_are_kept_only_when_the_short_wording_still_gives_the_same_table():
    simple = {"status": "plan", "steps": [{"op": "filter", "where": "plan == 'gold'"}, {"op": "select", "columns": ["visit_id", "fee"]}]}
    examples = [
        {"id": "w00000-01", "world": "w00000", "n": 1, "topic": "basic", "style": "plain", "response": simple,
         "request": "Filter visits to the gold plan, then return visit_id and fee."},
        {"id": "w00000-02", "world": "w00000", "n": 2, "topic": "basic", "style": "plain", "response": simple,
         "request": "Filter visits to the gold plan and return the columns visit_id and fee."},
        {"id": "w00000-03", "world": "w00000", "n": 3, "topic": "unsupported", "style": "plain",
         "response": {"status": "unsupported", "reason": "no charts"}, "request": "Plot it."},
    ]
    rewriter = Scripted(json.dumps({"rewrites": [{"n": 1, "request": "Gold plan visits: visit_id and fee."},
                                                  {"n": 2, "request": "Basic plan visits please."}]}))

    def second_opinion(user):
        wanted = "gold" if "Gold" in json.loads(user)["request"] else "basic"
        return code(f"return df[df['plan'] == '{wanted}'][['visit_id', 'fee']]")

    kept, rejected = rewrite_world(WORLD, examples, rewriter, Scripted(second_opinion))
    assert [(e["id"], e["request"], e["rewritten_from"]) for e in kept] == [
        ("w00000-01-r", "Gold plan visits: visit_id and fee.", "w00000-01")]
    assert kept[0]["response"] == simple
    assert [(r["n"], r["why"][:29]) for r in rejected] == [(2, "second model's result differs")]
    assert "Plot it." not in rewriter.prompts[0][1]  # refusals are not rewritten

    natural = Scripted(json.dumps({"rewrites": [{"n": 1, "request": "Gold members' visits: the visit id and the fee."}]}))
    kept, _ = rewrite_world(WORLD, examples[:1], natural, Scripted(second_opinion), style="natural")
    assert [e["id"] for e in kept] == ["w00000-01-n"] and "the way people talk" in natural.prompts[0][0]


def scripted_models():
    inventor = Scripted(json.dumps({"tables": {"visits": VISITS}}))

    def write_examples(user):
        batch = json.loads(user)["recipes"]
        examples = []
        for recipe in batch:
            if recipe["kind"] == "plan":
                examples.append({"n": recipe["n"], "request": f"Gold plan visits, visit_id and fee (variant {recipe['n']}).",
                                 "response": GOLD_PLAN})
            elif recipe["kind"] == "clarify":
                examples.append({"n": recipe["n"], "request": "Add the usual surcharge to fee.",
                                 "response": {"status": "needs_clarification", "question": "How much?"}})
            else:
                examples.append({"n": recipe["n"], "request": "Chart the fees please.",
                                 "response": {"status": "unsupported", "reason": "No charts."}})
        examples.append({"n": 99, "request": "not in the batch", "response": GOLD_PLAN})
        return json.dumps({"examples": examples})

    def second_opinion(user):
        if "surcharge" in user:
            return '{"status": "needs_clarification", "question": "What surcharge?"}'
        return code("return df[df['plan'] == 'gold'][['visit_id', 'fee']]")

    return inventor, Scripted(write_examples), Scripted(second_opinion)


def test_make_world_builds_keeps_and_rejects():
    inventor, author, verifier = scripted_models()
    world, kept, rejected = make_world(1, SPEC, inventor, author, verifier, per_world=8)
    assert world["id"] == "w00000" and world["tables"]["visits"] == VISITS
    assert len(kept) == 8 and rejected == []
    assert [e["id"] for e in kept] == [f"w00000-{n:02d}" for n in range(1, 9)]
    assert '"rows"' in author.prompts[0][1] and "gold" not in json.loads(author.prompts[0][1])["recipes"][0]


def test_generate_resumes_and_export_splits_by_world(tmp_path, monkeypatch, capsys):
    inventor, author, verifier = scripted_models()
    by_model = {"inv": inventor, "auth-a": author, "auth-b": author, "ver-a": verifier, "ver-b": verifier}
    monkeypatch.setattr(datagen_run, "make_provider", lambda provider, model, *a: by_model[model])
    models = ["--inventor", "inv", "--author", "auth-a, auth-b", "--verifier", "ver-a,ver-b"]
    monkeypatch.setattr(datagen_run, "eval_requests", lambda: {"chart the fees please."})
    # The scripted inventor only ever returns one plain table, so give every world a spec that fits it.
    monkeypatch.setattr(datagen_run, "world_spec", lambda seed, i, profile: {**SPEC, "id": f"w{i:05d}"})
    out = str(tmp_path)
    assert datagen_run.main(["--out", out, "generate", "--worlds", "3", "--per-world", "6", "--workers", "2", *models]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["worlds"] == 3 and first["kept"] + first["rejected"] == 18
    assert first["examples_total"] == first["kept"]
    assert first["spend"][0] == {"model": "inv", "calls": 3, "input_tokens": 300, "output_tokens": 120}
    assert [spent["calls"] for spent in first["spend"][1:3]] == [2, 1]  # the two authors took turns

    assert datagen_run.main(["--out", out, "generate", "--worlds", "3", *models]) == 0
    again = json.loads(capsys.readouterr().out)
    assert "worlds" not in again and all(model["calls"] == 0 for model in again["spend"])
    assert datagen_run.main(["--out", out, "generate", "--worlds", "9", "--target", "1", *models]) == 0
    assert "worlds" not in json.loads(capsys.readouterr().out)  # the target is already met, so nothing new starts

    assert datagen_run.main(["--out", out, "stats"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["worlds"] == 3 and stats["examples_kept"] == first["kept"]
    assert set(stats["by_author_kept_of_authored"]) == {"auth-a", "auth-b"}
    assert sum(stats["kept_by_verifier"].values()) == stats["examples_kept"]
    assert set(stats["rejected_because"]) <= {"repeats an eval request"}

    assert datagen_run.main(["--out", out, "export", "--valid-percent", "50"]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["train"] + exported["valid"] == first["kept"]
    line = json.loads((tmp_path / "mlx" / "train.jsonl").read_text().splitlines()[0]
                      if exported["train"] else (tmp_path / "mlx" / "valid.jsonl").read_text().splitlines()[0])
    roles_seen = [m["role"] for m in line["messages"]]
    assert roles_seen == ["system", "user", "assistant"]
    assert set(json.loads(line["messages"][1]["content"])) == {"request", "tables"}
    assert json.loads(line["messages"][2]["content"])["status"] in ("plan", "needs_clarification", "unsupported")


def test_training_topics_avoid_the_eval_sets():
    eval_tables = {name for version in ("v1", "v2", "test1") for task in load_tasks(f"evals/{version}") for name in task["tables"]}
    assert eval_tables <= BANNED_TABLES
    assert len(set(TOPICS)) == len(TOPICS) >= 100


SPIDER_TRAIN = Path(__file__).parent.parent / "data" / "spider" / "train.jsonl"


@pytest.mark.skipif(not SPIDER_TRAIN.exists(), reason="needs the raw Spider files")
def test_spider_training_questions_share_no_wording_and_no_sql_with_the_development_set():
    from datagen.spider import candidates, load_builder

    key = load_builder().sql_key
    dev = load_tasks(SPIDER_TRAIN.parent.parent.parent / "evals" / "real_dev")
    usable = candidates(None)[0]
    assert len(usable) > 3000
    assert not {task["request"] for task in usable} & {task["request"] for task in dev}
    assert not {key(task["tags"][0], task["sql"]) for task in usable} & {key(task["tags"][0], task["sql"]) for task in dev}
