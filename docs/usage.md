# Usage

How to run vasuki from the command line and from Python, and how to choose where the model runs.

- [Install](#install)
- [Ask in words](#ask-in-words)
- [Where the model runs](#where-the-model-runs)
- [Ollama](#ollama)
- [Run a plan you wrote](#run-a-plan-you-wrote)
- [Commands](#commands)
- [From Python](#from-python)

## Install

Nothing needs installing by hand except [uv](https://docs.astral.sh/uv/). This line fetches the program and runs it:

```sh
uvx --from git+https://github.com/rahul05ranjan/vasuki-vic vasuki ask "Total amount per vendor, highest first." -i invoices.csv
```

To work in a checkout instead:

```sh
git clone https://github.com/rahul05ranjan/vasuki-vic && cd vasuki-vic
uv sync
uv run vasuki ask "Total amount per vendor, highest first." -i examples/invoices.csv
```

Python 3.11 to 3.14; `uvx` fetches one if the machine has none. The engine imports nothing outside the standard library. Running the model on the machine itself uses MLX-LM, which is installed with the package on Apple silicon.

## Ask in words

```sh
vasuki ask "Total amount and number of invoices per vendor, as amount and invoices, biggest amount first." \
  -i examples/invoices.csv
```

```json
{
  "status": "ok",
  "columns": ["vendor", "amount", "invoices"],
  "row_count": 3,
  "rows": [["Acme", 2500.0, 3], ["Bolt", 500.5, 2], ["Cora", 0, 1]]
}
```

The model sees the request and a profile of each table: the row count, column names and types, null counts, three sample rows, and the distinct values of text columns that have twenty or fewer. It never sees the whole file. Loading the model takes about three seconds; a request then takes well under a second.

| Option | What it does |
|---|---|
| `-i [NAME=]PATH` | An input table: `.csv`, `.tsv`, `.json` (an array of objects) or `.jsonl`. Repeat it for a join. The table's name defaults to the file stem |
| `-o OUT` | Write every row of the result to a file |
| `--preview N` | Rows to print (default 20) |
| `--show-plan` | Include the plan the model wrote |
| `--model NAME` | A local folder or a Hugging Face name. Default: `$VASUKI_MODEL`, else `rahul05ranjan/VIC-2B` |
| `--ollama` | Use the Ollama on this machine |
| `--url URL` | Use any server with an OpenAI-style chat endpoint; needs `--model` |
| `--describe` | Send the full plan format with the request, for a model that was not trained on it |

The answer's `status` is `ok`, `needs_clarification` (with a `question`), `unsupported` (with a `reason`), or `failed` when the model's reply was not a plan that runs.

Write the request as one sentence that says what to compute, per what, over which rows and in what order. About one request in seven gets a plan that runs and returns the wrong table, so read the plan when the answer matters.

## Where the model runs

| Machine | What to pass | What runs the model |
|---|---|---|
| Mac with Apple silicon | nothing | MLX, inside the `vasuki` process. 1.55 GB download on first use |
| macOS, Linux or Windows with Ollama | `--ollama` | Ollama. 2.0 GB download on first use |
| Anything else | `--url URL --model NAME` | The server you name: LM Studio, llama.cpp, vLLM or a hosted model |

With `--url`, `$VASUKI_API_KEY` is sent as a bearer token if set. For a VIC model, vasuki turns the model's reasoning off itself; VIC was trained to reply with the plan alone. A model that was not trained on the plan format needs `--describe`.

## Ollama

`--ollama` talks to the Ollama on this machine (`$OLLAMA_HOST`, else `http://localhost:11434`) and uses `hf.co/rahul05ranjan/VIC-2B-GGUF:Q8_0`, the same model as an 8-bit GGUF file. If Ollama does not have it yet, vasuki asks Ollama to fetch it first, once.

```sh
vasuki ask "Total amount per vendor, highest first." -i invoices.csv --ollama
```

To fetch it yourself, or to use it from another program:

```sh
ollama pull hf.co/rahul05ranjan/VIC-2B-GGUF:Q8_0
```

Two things to know when calling the model without vasuki:

- **Turn reasoning off.** Ollama turns it on by default for this model family, and a VIC model asked to reason argues with the request and returns no plan. Send `"think": false` on Ollama's own API, or `"reasoning_effort": "none"` on the OpenAI-style endpoint.
- **It is not a chat model.** It expects the system prompt in `vasuki.request.VIC_SYSTEM` and, as the user message, the request with the table profiles as JSON (`vasuki.request.planner_input`). `vasuki ask` sends both.

On the development set the GGUF build in Ollama scores the same as the MLX build (210 of 245 against 210). This path was tested on macOS; Linux and Windows run the same GGUF file through Ollama, and reports from those platforms are welcome.

## Run a plan you wrote

```sh
vasuki run examples/invoice_totals.json -i examples/invoices.csv -o totals.csv
```

```json
{
  "status": "ok",
  "rows": 3,
  "columns": ["vendor", "invoices", "amount", "gst"],
  "steps": [
    {"op": "deduplicate", "rows_in": 6, "rows_out": 5},
    {"op": "derive", "rows_in": 5, "rows_out": 5},
    {"op": "aggregate", "rows_in": 5, "rows_out": 3},
    {"op": "sort", "rows_in": 3, "rows_out": 3}
  ],
  "preview": [...],
  "output_path": "totals.csv"
}
```

No model is involved. The plan format is in [plan-format.md](plan-format.md).

## Commands

| Command | What it does |
|---|---|
| `vasuki ask REQUEST -i [NAME=]PATH ... [-o OUT]` | Have a planner model write the plan, then run it |
| `vasuki run PLAN -i [NAME=]PATH ... [-o OUT]` | Check the plan, run it, print the report |
| `vasuki check PLAN -i [NAME=]PATH ...` | Check the plan against the input columns only |
| `vasuki schema` | Print the plan format, operations, expression functions and units as JSON |
| `vasuki eval-inputs DIR` | Print what a planner sees for each eval task, as JSONL |
| `vasuki eval DIR PREDICTIONS [--details]` | Score planner responses against an eval set |

`PLAN` is a JSON file, or `-` for stdin.

Exit codes: `0` ran, `1` error (for `ask`, also a reply that was not a plan that runs), `3` the response was a clarification request or a refusal.

## From Python

Run a plan:

```python
from vasuki import run_plan, check_plan, load_table, PlanError

result = run_plan(plan, {"invoices": load_table("invoices.csv")})
result.rows, result.columns, result.report()
```

Ask in words:

```python
from vasuki import load_table
from vasuki.planner import MLXPlanner, ask

planner = MLXPlanner()        # load once and reuse
answer = ask(planner, "Unpaid invoices over 500, newest first.", {"invoices": load_table("invoices.csv")})
answer["status"], answer.get("rows"), answer.get("plan")
```

`ollama_planner()` gives a planner on the local Ollama, and `HTTPPlanner(url, model, reasoning=False)` one on any OpenAI-style server. A planner is anything with `complete(system, user) -> str`, so your own is a few lines.
