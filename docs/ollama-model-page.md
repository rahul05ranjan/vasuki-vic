# VIC-2B

**A small model that turns a question about a data file into a typed plan, which the [vasuki](https://github.com/rahul05ranjan/vasuki-vic) engine checks and runs on your machine.**

Ask in plain words, get the answer as a table. The model never does arithmetic, never writes code, and never sees the whole file: it is shown column names, types and a few sample rows, and it writes a short plan of steps such as filter, aggregate and join. The engine does the computing.

## Use

VIC-2B is not a chat model. Use it through `vasuki`, which sends it the right input and runs the plan it writes. With [uv](https://docs.astral.sh/uv/) installed, one line:

```sh
uvx --from git+https://github.com/rahul05ranjan/vasuki-vic vasuki ask "Total amount per vendor, highest first." -i invoices.csv --ollama --model rahuul/vic-2b
```

```json
{
  "status": "ok",
  "columns": ["vendor", "total_amount"],
  "row_count": 3,
  "rows": [["Acme", 2500.0], ["Bolt", 500.5], ["Cora", 0]]
}
```

Add `--show-plan` to see the plan the model wrote. Inputs can be `.csv`, `.tsv`, `.json` or `.jsonl`; give two `-i` for a join.

## Calling it yourself

Two things to know if you call the model without vasuki:

- **Turn reasoning off.** Ollama turns it on by default for this model family, and VIC asked to reason returns no plan. Send `"think": false` on Ollama's API, or `"reasoning_effort": "none"` on the OpenAI-style endpoint. With `ollama run`, pass `--think=false`.
- **The user message is JSON.** The request plus a profile of each table, as `vasuki.request.planner_input` builds it. The system prompt is stored with this model.

The reply is one JSON object: a plan, a `needs_clarification` question, or an `unsupported` reason.

## How good it is

Scored by running the model's plan and comparing the resulting table with the expected one. The reference is `gpt-oss-120b`, a 120B-parameter model, given the full format description.

| Test set | Tasks | VIC-2B | Reference model |
|---|---:|---:|---:|
| Questions real people wrote, from Spider's validation split | 681 | 83.0% | 81.6% |
| The same kind from a second source, BIRD's dev split | 195 | 66.7% | 65.6% |
| Hand-written requests that spell out their output | 173 | 82.1% | 93.6% |

These figures were measured on the 6-bit MLX build. This 8-bit GGUF build scores the same as that build on the development set (210 of 245).

**About one plain-language request in seven gets a plan that runs and returns the wrong table, with no warning.** Read the plan when the answer matters.

## Details

| | |
|---|---|
| Base model | Qwen3.5-2B, text part, fine-tuned with LoRA |
| This build | GGUF, 8-bit (`Q8_0`), 2.0 GB |
| Languages | English; Hinglish to a lesser degree |
| Licence | Apache 2.0 |
| Code, eval sets and docs | https://github.com/rahul05ranjan/vasuki-vic |
| Full model card | https://huggingface.co/rahul05ranjan/VIC-2B |
