"""Model backends for the benchmark: one call in, text and token usage out."""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens: input, output, cache read, 5-minute cache write.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25),
    # Workers AI, from the account's model catalog on 2026-10-07. No separate cache-write price is listed.
    "@cf/openai/gpt-oss-120b": (0.35, 0.75, 0.35, 0.35),
    "@cf/zai-org/glm-5.3": (1.40, 4.40, 0.26, 1.40),
    "@cf/qwen/qwen3.8-27b": (0.45, 3.20, 0.05, 0.45),
    "@cf/deepseek-ai/deepseek-v4-pro-0813": (1.32, 3.96, 0.044, 1.32),
    "@cf/ibm-granite/granite-4.0-h-micro": (0.017, 0.112, 0.017, 0.017),
    "@cf/meta/llama-3.2-3b-instruct": (0.0509, 0.335, 0.0509, 0.0509),
    "@cf/meta/llama-3.2-1b-instruct": (0.027, 0.201, 0.027, 0.027),
}


class FatalProviderError(Exception):
    """The provider cannot serve any request as configured; stop the run."""


@dataclass
class Completion:
    text: str
    model: str
    stop_reason: str | None
    input_tokens: int  # input not served from cache and not written to it
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int

    def record(self) -> dict:
        usage = asdict(self)
        return {"text": usage.pop("text"), "model": usage.pop("model"), "stop_reason": usage.pop("stop_reason"),
                "usage": usage}


class AnthropicProvider:
    name = "anthropic"
    workers = 4

    def __init__(self, model: str, effort: str | None = "medium", fallbacks: bool = True, max_tokens: int = 16000):
        import anthropic

        self._anthropic = anthropic
        self.model, self.effort, self.fallbacks, self.max_tokens = model, effort, fallbacks, max_tokens
        try:
            self.client = anthropic.Anthropic(max_retries=5)
        except anthropic.AnthropicError as e:
            raise FatalProviderError(f"no Anthropic credentials: {e}") from e

    def _fatal(self):
        a = self._anthropic
        return (a.AuthenticationError, a.PermissionDeniedError, a.NotFoundError, a.BadRequestError, TypeError)

    def complete(self, system: str, user: str) -> Completion:
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # The system prompt is identical for every task in an arm, so it is the cached prefix.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        try:
            if self.fallbacks:
                # A declined request is re-run on Anthropic's recommended substitute. The record keeps
                # `model`, so the report can show any task that a different model answered.
                response = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **params)
            else:
                response = self.client.messages.create(**params)
        except self._fatal() as e:
            raise FatalProviderError(f"{type(e).__name__}: {e}") from e
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return Completion(
            text=text,
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=usage.input_tokens,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            output_tokens=usage.output_tokens,
        )

    def count_tokens(self, text: str) -> int:
        try:
            counted = self.client.messages.count_tokens(model=self.model, messages=[{"role": "user", "content": text}])
        except self._fatal() as e:
            raise FatalProviderError(f"{type(e).__name__}: {e}") from e
        return counted.input_tokens


class OllamaProvider:
    """A local model served by Ollama. Runs one request at a time."""

    name = "ollama"
    workers = 1

    def __init__(self, model: str, host: str = "http://localhost:11434", context: int = 8192,
                 max_tokens: int = 4000, timeout: float = 400, think: bool | None = None):
        self.model, self.host, self.context, self.timeout = model, host.rstrip("/"), context, timeout
        self.think = think    # None leaves reasoning to the model's default; False is what a VIC model needs
        # The cap stops a model that repeats itself without end. It counts hidden reasoning too, so it
        # has to be far above the size of a plan: a 1500 cap cut Gemma 4 off before it answered.
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> Completion:
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.context, "num_predict": self.max_tokens},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.think is not None:
            body["think"] = self.think
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (400, 404):
                raise FatalProviderError(f"Ollama rejected the request ({e.code}): {detail}") from e
            raise RuntimeError(f"Ollama error {e.code}: {detail}") from e
        except TimeoutError as e:
            raise RuntimeError(f"no reply within {self.timeout:g}s") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise RuntimeError(f"no reply within {self.timeout:g}s") from e
            raise FatalProviderError(f"cannot reach Ollama at {self.host}: {e.reason}") from e
        return Completion(
            text=data.get("message", {}).get("content", ""),
            model=data.get("model", self.model),
            stop_reason=data.get("done_reason"),
            input_tokens=data.get("prompt_eval_count", 0),
            cache_read_tokens=0,
            cache_write_tokens=0,
            output_tokens=data.get("eval_count", 0),
        )


class CloudflareProvider:
    """Cloudflare's OpenAI-compatible endpoint.

    One endpoint serves two kinds of model, chosen by name: Workers AI models (`@cf/openai/gpt-oss-120b`)
    and, with Unified Billing credits on the account, third-party models (`anthropic/claude-opus-5-5`).
    """

    name = "cloudflare"
    # Cloudflare rate-limits per account (error 971), so several runs at once starve each other.
    workers = 4
    retries = 6

    def __init__(self, model: str, max_tokens: int = 8000, timeout: float = 300):
        account, token = os.environ.get("CLOUDFLARE_ACCOUNT_ID"), os.environ.get("CLOUDFLARE_API_TOKEN")
        if not account or not token:
            raise FatalProviderError("set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (see README.md, Benchmarking a model)")
        self.model, self.max_tokens, self.timeout = model, max_tokens, timeout
        # Extra request fields, such as {"reasoning_effort": "low"} for models that reason before answering.
        self.extra: dict = {}
        # Attempts that had to be retried (rate limit, timeout, server error); a rough measure of throttling.
        self.retried = 0
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1/chat/completions"
        self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=self._headers)
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode(errors="replace")[:400]
                except (OSError, http.client.HTTPException):
                    detail = "(connection dropped while reading the error)"
                if e.code in (400, 401, 403, 404):
                    raise FatalProviderError(f"Cloudflare rejected the request ({e.code}): {detail}") from e
                if e.code == 429 and ("allocation" in detail.lower() or "neurons" in detail.lower()):
                    # The free daily allowance is spent; retrying today cannot succeed.
                    raise FatalProviderError(f"Cloudflare daily allowance used up: {detail}") from e
                failure = f"Cloudflare error {e.code}: {detail}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
                # Includes connections the server drops mid-request, which happen under heavy parallel load.
                failure = f"cannot reach Cloudflare: {getattr(e, 'reason', None) or type(e).__name__}"
            self.retried += 1
            if attempt < self.retries - 1:
                time.sleep(min(2 ** attempt * 3, 45))
        raise RuntimeError(failure)

    def complete(self, system: str, user: str) -> Completion:
        data = self._post({
            "model": self.model,
            "max_tokens": self.max_tokens,
            **self.extra,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return Completion(
            text=(choice.get("message") or {}).get("content") or "",
            model=data.get("model", self.model),
            stop_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens", 0) - cached,
            cache_read_tokens=cached,
            cache_write_tokens=0,
            output_tokens=usage.get("completion_tokens", 0),
        )


class MLXProvider:
    """A model run on this Mac with MLX-LM, optionally with a fine-tuned LoRA adapter on top."""

    name = "mlx"
    workers = 1

    def __init__(self, model: str, adapter: str | None = None, max_tokens: int = 700):
        try:
            from mlx_lm import generate, load
        except ImportError as e:
            raise FatalProviderError("MLX-LM is not installed; run `uv sync --group train`") from e
        self.model, self.max_tokens, self._generate = model, max_tokens, generate
        try:
            self._net, self._tokenizer = load(model, adapter_path=adapter)
        except Exception as e:
            raise FatalProviderError(f"cannot load {model}: {type(e).__name__}: {e}") from e

    def complete(self, system: str, user: str) -> Completion:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            # Qwen's template thinks by default; a planner's reply is the JSON itself.
            prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                         enable_thinking=False)
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text = self._generate(self._net, self._tokenizer, prompt=prompt, max_tokens=self.max_tokens, verbose=False)
        output_tokens = len(self._tokenizer.encode(text))
        return Completion(
            text=text,
            model=self.model,
            stop_reason="length" if output_tokens >= self.max_tokens else "stop",
            input_tokens=len(self._tokenizer.encode(prompt)),
            cache_read_tokens=0,
            cache_write_tokens=0,
            output_tokens=output_tokens,
        )


def make_provider(name: str, model: str, effort: str | None, fallbacks: bool, adapter: str | None = None):
    if name == "anthropic":
        return AnthropicProvider(model, effort=effort, fallbacks=fallbacks)
    if name == "ollama":
        return OllamaProvider(model)
    if name == "cloudflare":
        return CloudflareProvider(model)
    if name == "mlx":
        return MLXProvider(model, adapter)
    raise FatalProviderError(f"unknown provider {name!r}")
