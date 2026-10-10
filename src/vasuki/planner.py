"""Ask a planner model for a plan and run it: the path from a request in words to a result table.

A planner is anything with `complete(system, user) -> str`. Two are provided: `MLXPlanner` runs a VIC model on
Apple silicon, and `HTTPPlanner` talks to any server with an OpenAI-style `/chat/completions` endpoint (Ollama,
LM Studio, llama.cpp, vLLM, a hosted model). Neither is imported until it is used, so the executor keeps working
without them.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

from .errors import PlanError
from .executor import Table, run_plan
from .io import write_table
from .request import VIC_SYSTEM, planner_input
from .schema import describe, validate_response

DEFAULT_MODEL = "rahul05ranjan/VIC-2B"
# The same model as a GGUF file, which Ollama fetches by this name and runs on macOS, Linux and Windows.
OLLAMA_MODEL = "hf.co/rahul05ranjan/VIC-2B-GGUF:Q8_0"
OLLAMA_URL = "http://localhost:11434"


def full_system_prompt() -> str:
    """For a model that was not fine-tuned on the plan format: the short instruction plus the whole format."""
    return (
        "You are a planner for the Vasuki executor. Turn the request into one JSON object: a plan, a "
        "needs_clarification question, or an unsupported reason. Reply with the JSON only. If only part of the "
        "request can be done with these operations, reply unsupported and say why.\n\nThe format:\n"
        + json.dumps(describe(), separators=(",", ":"))
    )


class MLXPlanner:
    """A VIC model on this machine, through MLX-LM. Loading takes a few seconds; keep the object and reuse it."""

    def __init__(self, model: str | None = None, max_tokens: int = 700):
        try:
            from mlx_lm import generate, load
        except ImportError:
            raise PlanError("planner_unavailable", "this machine cannot run the model itself (MLX-LM needs a Mac "
                            "with Apple silicon); install Ollama and pass --ollama, or point --url at a "
                            "server that has the model") from None
        self.model = model or os.environ.get("VASUKI_MODEL") or DEFAULT_MODEL
        self.max_tokens, self._generate = max_tokens, generate
        try:
            self._net, self._tokenizer = load(self.model)
        except Exception as e:
            raise PlanError("planner_unavailable", f"cannot load {self.model}: {type(e).__name__}: {e}") from None

    def complete(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            # Qwen's template reasons out loud unless told not to; VIC was trained to reply with the JSON alone.
            prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                         enable_thinking=False)
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return self._generate(self._net, self._tokenizer, prompt=prompt, max_tokens=self.max_tokens, verbose=False)


class HTTPPlanner:
    """A model behind an OpenAI-style chat endpoint, e.g. http://localhost:11434/v1 for Ollama."""

    def __init__(self, url: str, model: str, api_key: str | None = None, max_tokens: int = 2000,
                 timeout: float = 120, reasoning: bool = True):
        self.url = url.rstrip("/") + ("" if url.rstrip("/").endswith("/chat/completions") else "/chat/completions")
        self.model, self.api_key, self.max_tokens, self.timeout = model, api_key, max_tokens, timeout
        # False for a VIC model. It was trained to answer with no reasoning, and a server that turns
        # reasoning on by default (Ollama does, for Qwen bases) gets an argument from it and no plan.
        self.reasoning = reasoning

    def complete(self, system: str, user: str) -> str:
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if not self.reasoning:
            body["reasoning_effort"] = "none"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            try:
                reply = self._post(body, headers)
            except urllib.error.HTTPError as e:
                if e.code != 400 or "reasoning_effort" not in body:
                    raise
                del body["reasoning_effort"]        # a server that does not know the field
                reply = self._post(body, headers)
            return reply["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            raise PlanError("planner_unavailable", f"{self.url} answered {e.code}: {e.reason}") from None
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            raise PlanError("planner_unavailable", f"no usable reply from {self.url}: {type(e).__name__}: {e}") from None


    def _post(self, body: dict, headers: dict) -> dict:
        request = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)


def ollama_planner(model: str | None = None, url: str | None = None, reasoning: bool = False) -> HTTPPlanner:
    """A planner on the Ollama running on this machine. A model Ollama does not have yet is fetched first."""
    model = model or OLLAMA_MODEL
    base = (url or os.environ.get("OLLAMA_HOST") or OLLAMA_URL).rstrip("/")
    if "://" not in base:
        base = "http://" + base
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=10) as response:
            have = {entry.get("name") for entry in json.load(response).get("models", [])}
        if not have & {model, model + ":latest"}:
            print(f"fetching {model} into Ollama; this happens once", file=sys.stderr)
            pull = urllib.request.Request(base + "/api/pull", json.dumps({"model": model, "stream": False}).encode(),
                                          {"Content-Type": "application/json"})
            with urllib.request.urlopen(pull, timeout=3600) as response:
                status = json.load(response)
            if status.get("status") != "success":
                raise PlanError("planner_unavailable", f"Ollama could not fetch {model}: {status.get('error') or status}")
    except urllib.error.HTTPError as e:
        raise PlanError("planner_unavailable", f"Ollama at {base} answered {e.code}: {e.reason}") from None
    except (OSError, ValueError) as e:
        raise PlanError("planner_unavailable", f"no Ollama at {base} ({type(e).__name__}); install it from "
                        "https://ollama.com and start it") from None
    return HTTPPlanner(base + "/v1", model, reasoning=reasoning)


def extract_json(text: str):
    """The first JSON object in `text`, tolerating code fences and surrounding prose; None if there is none."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            return decoder.raw_decode(text, match.start())[0]
        except json.JSONDecodeError:
            continue
    return None


def ask(planner, request: str, tables: dict[str, Table], *, system: str = VIC_SYSTEM, preview: int = 20,
        output: str | None = None) -> dict:
    """Plan `request` with `planner`, run the plan, and return a compact answer.

    The answer's `status` is one of:
      ok                   the plan ran; `columns`, `row_count`, the first `preview` `rows` (each a list of
                           values in column order, which costs a reader half the tokens of named fields), and
                           the `plan`
      needs_clarification  the planner needs something the request did not say; `question`
      unsupported          the request is outside the operations; `reason`
      failed               the planner's reply was not a plan that runs; `error`, and the `reply` it gave
    A planner that cannot be reached raises PlanError instead, since that is not an answer to the request.
    """
    text = planner.complete(system, json.dumps(planner_input(request, tables), separators=(",", ":"),
                                               ensure_ascii=False))
    reply = extract_json(text)
    try:
        if reply is None:
            raise PlanError("invalid_response", "the planner's reply holds no JSON object")
        validate_response(reply)
        if reply["status"] != "plan":
            return reply
        result = run_plan(reply, tables)
    except PlanError as e:
        error = {key: value for key, value in e.to_dict().items() if key != "status"}
        return {"status": "failed", "error": error, "reply": reply or text}
    answer = {"status": "ok", "columns": result.columns, "row_count": len(result.rows),
              "rows": [[row[column] for column in result.columns] for row in result.rows[:preview]],
              "plan": reply["steps"]}
    if len(result.rows) > preview:
        answer["truncated"] = f"showing {preview} of {len(result.rows)} rows; pass an output path for all of them"
    if output:
        write_table(output, result.columns, result.rows)
        answer["output_path"] = output
    return answer
