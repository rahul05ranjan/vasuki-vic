"""The two ways a model is asked to solve an eval task, and how its reply is read."""

from __future__ import annotations

import json
import re

from vasuki import describe
from vasuki.evals import task_input
from vasuki.request import VIC_SYSTEM

_SHARED = """\
You are given a request and a profile of each input table: column names and types, null counts, a few \
sample rows, and the distinct values of text columns that have only a few.

Name output columns exactly as the request names them. Return only the columns the request asks for; \
when it does not say, keep whatever columns the work naturally produces. Dates are ISO text and stay text."""

_STATUS_FORMS = """\
- {"status": "needs_clarification", "question": "..."} when the request leaves out something you would \
have to guess, such as a rate, a threshold, or which of several columns is meant.
- {"status": "unsupported", "reason": "..."} when %s."""

PLAN_SYSTEM = f"""\
You turn a data request into a plan for the Vasuki executor. You never compute results yourself.

{_SHARED}

Reply with exactly one JSON object and nothing else, in one of three forms:
- {{"status": "plan", "steps": [...]}} when the operations below can do what is asked.
{_STATUS_FORMS % "the operations below cannot express the request"}

Reference for the plan format:
{json.dumps(describe(), separators=(",", ":"))}"""

CODE_SYSTEM = f"""\
You turn a data request into pandas code.

{_SHARED}

Reply in exactly one of three forms and nothing else:
- One ```python code block defining `def solve(tables)`. `tables` maps each table name to a pandas \
DataFrame, and `solve` returns the result as a DataFrame. `pd` and `np` are already imported. The index \
is discarded, so everything you want returned must be a column. Missing values are NaN or None.
{_STATUS_FORMS % "code alone cannot do it, for example because it needs outside data or a human decision"}"""

# "vic" is the plan arm for a fine-tuned planner: same task and scoring, but no format description.
SYSTEM = {"plan": PLAN_SYSTEM, "code": CODE_SYSTEM, "vic": VIC_SYSTEM}

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def user_message(task: dict) -> str:
    shown = task_input(task)
    del shown["id"]
    return json.dumps(shown, separators=(",", ":"), ensure_ascii=False)


def extract_json(text: str):
    """The first JSON object in `text`, tolerating code fences and surrounding prose; None if there is none."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            return decoder.raw_decode(text, match.start())[0]
        except json.JSONDecodeError:
            continue
    return None


def extract_code(text: str) -> str | None:
    blocks = [block for block in _FENCE.findall(text) if "def solve" in block]
    return blocks[0] if blocks else None
