# vasuki

Ask a question about a CSV file in plain words and get the answer as a table, computed on your own machine.

```sh
uvx --from git+https://github.com/rahul05ranjan/vasuki-vic vasuki ask "Total amount per vendor, highest first." -i invoices.csv
```

That one line is the whole setup on a Mac with Apple silicon, given [uv](https://docs.astral.sh/uv/). The first run installs the program and downloads the VIC model (1.55 GB, about two minutes); later runs take a few seconds. Nothing leaves the machine.

To have a coding agent (Claude Code, Codex, Cursor) use it by itself, install the skill: `npx skills add rahul05ranjan/vasuki-vic` (see "For agents: the skill").

Under it are two parts: typed operation plans for tabular work with a deterministic executor that runs them (`vasuki`), and a small model that writes those plans from a request ([VIC](https://huggingface.co/rahul05ranjan/VIC-2B-preview)).

A planner (any model, or a person) describes a computation as a JSON plan: a list of steps such as `filter`, `derive`, `aggregate` and `join`. Vasuki checks the whole plan against the input columns, runs it locally, and returns a compact report in place of the data. The planner never does arithmetic and never writes code.

The executor imports nothing outside the standard library. Python 3.11 to 3.14; `uvx` fetches one if the machine has none. Running a VIC model locally uses MLX-LM, which is installed with the package on Apple silicon; on any other machine, point `--url` at a server that has the model (see "Ask in words").

## How good the model is

Scored by running the model's plan and comparing the resulting table with the expected one. `VIC-2B-preview` is run `k5`, 1.55 GB at 6 bits; the reference is `gpt-oss-120b` given the full format description.

| Set | Tasks | VIC-2B-preview | Reference model |
|---|---:|---:|---:|
| Questions real people wrote, from Spider's validation split (`evals/real1`) | 681 | 83.0% | 81.6% |
| The same kind from a second source, BIRD's dev split (`evals/real2`) | 195 | 66.7% | 65.6% |
| Hand-written test set, requests that spell out their output (`evals/test1`) | 173 | 82.1% | 93.6% |
| Development set (`evals/v2`) | 245 | 85.7% | 97.1% |

The first two rows ignore column names, because those questions name none; on both, the two models cannot be told apart. The model was trained on Spider's train split, so the second row is the check that the first is not familiarity with Spider's wording.

**What that leaves.** About one request in seven gets a plan that runs and returns the wrong table, with no warning. Read the plan (`--show-plan`) when the answer matters; it is short and says exactly which rows and columns were used. This is a preview, not something to rely on unattended.

## Names

- **vasuki** is this package: the plan format, the executor and the eval harness. It works with any planner.
- **VIC** (Vasuki Indicus Compute) is the planner model trained to emit this format. Checkpoints are named `VIC-<size>` by parameter count, for example `VIC-2B` or `VIC-100M`.

## Try it

```sh
uv sync
uv run vasuki run examples/invoice_totals.json -i examples/invoices.csv -o totals.csv
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

## Ask in words

With a VIC model, the plan is written for you:

```sh
uv run vasuki ask "Total amount and number of invoices per vendor, as amount and invoices, biggest amount first." \
  -i examples/invoices.csv
```

In a checkout that is the whole setup too; from any other folder use the `uvx` line at the top of this page.

```json
{
  "status": "ok",
  "columns": ["vendor", "amount", "invoices"],
  "row_count": 3,
  "rows": [["Acme", 2500.0, 3], ["Bolt", 500.5, 2], ["Cora", 0, 1]]
}
```

The model sees the request and a profile of each table (column names, types, three sample rows), never the whole file. Loading it takes about three seconds; a request then takes well under a second. Add `--show-plan` to see the plan it wrote, and `-o out.csv` to save every row.

The answer's `status` says what happened: `ok`; `needs_clarification` with a `question`; `unsupported` with a `reason`; or `failed` when the model's reply was not a plan that runs. The model is right on about four requests in five (see "How good the model is"), so read the plan when the answer matters.

- `--model` takes a local folder or a Hugging Face name. The default is `$VASUKI_MODEL`, else the published preview.
- A VIC model in Ollama, for a model you have trained and merged yourself: `uv run python kaggle_job/to_ollama.py --label k5` installs `models/VIC-k5-merged` as `vic-k5`, and `--url http://localhost:11434/v1 --model vic-k5` uses it. `vasuki` turns the model's reasoning off for you; with `ollama run`, pass `--think=false`.
- `--url http://localhost:11434/v1 --model NAME` uses any server with an OpenAI-style chat endpoint (Ollama, LM Studio, llama.cpp, a hosted model) in place of MLX. Add `--describe` to send the full plan format with the request, which a model not trained on the format needs.

## For agents: the skill

A coding agent that can run a shell command needs no server. The skill in `skills/vasuki` tells it when to hand a question to `vasuki ask`, how to read the answer, and how to check the plan before trusting the rows:

```sh
npx skills add rahul05ranjan/vasuki-vic
```

That is the only step. It installs the skill for Claude Code, Codex, Cursor and the other agents the [skills](https://skills.sh) installer knows, and needs Node.js for `npx`.

What it puts on the machine is one file of instructions, `SKILL.md`. The program and the model are not installed by that command: the skill has the agent run the `uvx` line at the top of this page, which fetches both the first time a question is asked (1.55 GB, about two minutes) and takes a few seconds after that. If `uv` is missing, the skill has the agent offer to install it. The model itself still needs a Mac with Apple silicon, or a server given with `--url`.

Then ask the agent in plain words, for example "which vendor has the highest total in invoices.csv?". The agent decides when the skill applies; about one answer in seven is wrong without warning (see "How good the model is"), so the skill has it check the plan before it trusts the rows.

## For agents: the MCP server

`vasuki-mcp` lets an agent (Claude Code, Codex, or any MCP client) hand tabular work to VIC instead of reading data files into its own context. The agent sends a request and file paths; only the first rows of the result come back.

```sh
claude mcp add vasuki -- uv run --directory /path/to/vasuki vasuki-mcp     # Claude Code
```

For another client, register the same command as a stdio server:

```json
{"mcpServers": {"vasuki": {"command": "uv", "args": ["run", "--directory", "/path/to/vasuki", "vasuki-mcp"],
                           "env": {"VASUKI_MODEL": "rahul05ranjan/VIC-2B-preview"}}}}
```

| Tool | What it does |
|---|---|
| `compute(request, files, output_path?, preview_rows?)` | Plans the request with VIC and runs it. Returns the same answer as `vasuki ask`, with the plan |
| `describe_tables(files)` | Row count, columns, types and three sample rows per file |
| `run_plan(steps, files, output_path?, preview_rows?)` | Runs a plan the agent wrote itself, for when `compute` gets it wrong |
| `plan_format()` | The plan format, for writing plans by hand (about 2,000 tokens) |

The model loads on the first `compute` and stays in memory. `VASUKI_MODEL`, `VASUKI_URL`, `VASUKI_API_KEY` and `VASUKI_DESCRIBE=1` choose the planner, as the flags of `vasuki ask` do. The server reads any file path it is given and writes `output_path`, with the permissions of whoever starts it.

## Commands

| Command | What it does |
|---|---|
| `vasuki ask REQUEST -i [NAME=]PATH ... [-o OUT]` | Have a planner model write the plan, then run it |
| `vasuki run PLAN -i [NAME=]PATH ... [-o OUT]` | Check the plan, run it, print the report |
| `vasuki check PLAN -i [NAME=]PATH ...` | Check the plan against the input columns only |
| `vasuki schema` | Print the plan format, ops, expression functions and units as JSON |
| `vasuki eval-inputs DIR` | Print what a planner sees for each eval task, as JSONL |
| `vasuki eval DIR PREDICTIONS [--details]` | Score planner responses against an eval set |

`PLAN` is a JSON file, or `-` for stdin. Inputs and outputs can be `.csv`, `.tsv`, `.json` (an array of objects) or `.jsonl`. A table's name defaults to its file stem.

Exit codes: `0` ran, `1` error (for `ask`, also a reply that was not a plan that runs), `3` the response was a clarification request or a refusal.

## The response format

A planner returns one of three things:

```json
{"status": "plan", "steps": [{"op": "filter", "where": "amount > 100"}]}
{"status": "needs_clarification", "question": "Is GST included in amount?"}
{"status": "unsupported", "reason": "needs a currency exchange rate"}
```

Steps run in order. A step reads the previous step's result unless `input` names a table, and `as` names a step's result so a later step or join can use it. The last step's result is the output.

`vasuki schema` is the reference for the ops and their fields. Its `response_schema` is a JSON Schema, so it can drive constrained decoding.

## Behaviour worth knowing

- **Step names can be reused.** `as` may name a result after an existing table; later steps see the newest one.
- **Checked before run.** Every column, table, unit and expression is resolved against the input columns before any step runs. A typo in step 5 fails immediately and names the closest match.
- **Expressions are not `eval`.** `where` and `expr` use Python syntax, parsed to an AST and interpreted against a whitelist. There is no indexing, slicing or `**`. A few Python habits are accepted as other spellings: `round(x, digits=2)`, `x.lower()`, `x.startswith(y)`, `x is null`.
- **Strict types.** Arithmetic on text is an error, not a coercion. Use `cast`, or `int()` / `float()` in an expression. The one exception is a join key: when it is a number in one table and text in the other, the text is read as the number it spells, as a database would.
- **Nulls.** They propagate through arithmetic and functions, compare as false, sort last, never match in a join, and are skipped by aggregates.
- **Rounding** is half-up on the decimal as written: `round(2.675, 2)` is `2.68`.
- **CSV typing** is per column: all-integer columns become ints, all-numeric become floats, anything else stays text. Values with leading zeros (`01234`) stay text.
- **No currency units.** Exchange rates are data, so a plan must multiply by an explicit rate.

## Evaluating a planner

`evals/v2` holds 245 held-out tasks over 16 small tables. Each task is a request, its input tables, and a frozen expected result. (`evals/v1` is the first version, kept frozen because earlier model runs were scored on it.)

`evals/test1` is the test set: 173 tasks over 17 tables on subjects that appear in neither `evals/v2` nor the training data. `evals/v2` is where failures are read and acted on; `evals/test1` is only scored, so its number stays an honest estimate. Pass `--tasks evals/test1` to the commands below.

`evals/real1` (681 tasks) and `evals/real_dev` (250 tasks) hold questions written by people who wanted an answer, taken from the Spider text-to-SQL dataset (Yu et al., 2018, CC BY-SA 4.0) with each expected result computed by Spider's own SQL; `evals/from_spider.py` builds them. These questions name no output columns, so read `accuracy_ignoring_column_names` in the report. `real1` is a test set, scored and never steered by; `real_dev` is for reading failures.

`evals/real2` (195 tasks) is a second test set of the same kind from another source, the BIRD benchmark (Li et al., 2023, CC BY-SA 4.0), built by `evals/from_bird.py` under the same rules. Its questions were written by other people about other databases, and each carries BIRD's line of evidence after "Note:". It exists because VIC is trained on Spider's train split, which makes `real1` a measure of Spider-style questions. Scored and never steered by.

```sh
uv run vasuki eval-inputs evals/v2 > inputs.jsonl      # what the planner sees, one task per line
# ...run your planner, writing {"id": ..., "response": {...}} per line to predictions.jsonl...
uv run vasuki eval evals/v2 predictions.jsonl --details
```

A planner sees the request plus a profile of each table: column names and types, null counts, three sample rows, and the distinct values of text columns with eight or fewer of them. It also needs the format description from `vasuki schema`.

Scoring runs the response and compares the resulting table with the expected one, so any plan that produces the right table is correct. Row order matters only where the request asks for an order, column order never matters, and numbers match within 1e-9. The report gives accuracy overall, by category and by tag, and a second figure that forgives different column names.

| Category | Tasks | What it covers |
|---|---:|---|
| `basic` | 46 | One or two simple steps |
| `derive` | 27 | Row-level arithmetic, conditions, null handling |
| `aggregate` | 35 | A single grouping step |
| `multi` | 43 | Several steps on one table, up to seven |
| `join` | 22 | Two tables, including anti-joins, self-joins, a rank and a running total |
| `units` | 17 | Conversions, including ones the unit table does not cover |
| `text_date` | 15 | Text and date functions |
| `tricky` | 16 | Data where the obvious plan is wrong: text-typed numbers, duplicates, messy casing |
| `clarify` | 14 | Requests that cannot be planned without more information |
| `unsupported` | 10 | Requests the operations cannot express |

14 tasks are written in Hinglish and tagged `hinglish`.

`evals/v2/build.py` is the authoring source; run it to regenerate `tables.json` and `tasks.jsonl`. The test suite checks that every gold plan reproduces its frozen result and that removing any single step from a gold plan changes the answer.

**Keep this set held out.** Do not generate training data from these tables or requests.

## Benchmarking a model

`bench/` runs a model over the eval set and reports accuracy and token use. It needs the `bench` dependency group (`anthropic`, `pandas`), which `uv sync` installs by default.

```sh
uv run python -m bench.run run --arm plan --provider anthropic --model claude-opus-5-5
uv run python -m bench.run run --arm code --provider anthropic --model claude-opus-5-5
uv run python -m bench.run delegate --provider anthropic --model claude-opus-5-5
uv run python -m bench.run report --details
```

| Piece | What it measures |
|---|---|
| `--arm plan` | The model writes vasuki plans. Its accuracy is the ceiling for a planner using this format. |
| `--arm code` | The model writes pandas itself. This is what a caller spends without a planner. Skips the `unsupported` category. |
| `delegate` | No model call. Sizes the call a caller would write and the report it would read when delegating. |
| `report` | Scores every results file and, for a code arm, gives the planner accuracy needed for delegating to break even. |

Providers:

- `cloudflare` needs `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`. It serves Workers AI models (`--model @cf/openai/gpt-oss-120b`) and, with Unified Billing credits on the account, third-party models (`--model anthropic/claude-opus-5-5`).
- `anthropic` needs `ANTHROPIC_API_KEY`.
- `ollama` runs a local model, for example `--model gemma4:12b-mlx`.

Credentials can go in a `.env` file in this folder as `KEY=value` lines; the runner reads it and `.gitignore` excludes it.

It runs `evals/v2` unless `--tasks` says otherwise. Results go to `bench/results/<eval set>/<arm>__<model>.jsonl`. A run resumes from that file, so stopping and restarting never repeats a finished task. `--per-category N` runs a small sample first.

The code arm executes model-written Python in a subprocess with a time limit, in an empty temporary directory. It is not isolated from the network or the rest of the file system.

## From Python

```python
from vasuki import run_plan, check_plan, load_table, PlanError

result = run_plan(plan, {"invoices": load_table("invoices.csv")})
result.rows, result.columns, result.report()
```

With a planner model:

```python
from vasuki.planner import MLXPlanner, ask

planner = MLXPlanner()        # load once, reuse; HTTPPlanner(url, model) talks to an OpenAI-style server instead
answer = ask(planner, "Unpaid invoices over 500, newest first.", {"invoices": load_table("invoices.csv")})
answer["status"], answer.get("rows"), answer.get("plan")
```

## Tests

```sh
uv run pytest
```

## Licence

The code and the VIC model are released under the Apache License 2.0 (`LICENSE`). Three eval sets are derived from other people's datasets and keep those datasets' licence, CC BY-SA 4.0: `evals/real1` and `evals/real_dev` from Spider (Yu et al., 2018), and `evals/real2` from BIRD (Li et al., 2023). `NOTICE` has the details, including what the model was trained on.
