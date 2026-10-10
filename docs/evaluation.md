# Evaluation

How VIC-2B was measured, what the eval sets contain, and how to score a planner of your own.

## Results

VIC-2B is measured as shipped: 6-bit, 1.55 GB, on a Mac. The reference is `gpt-oss-120b`, given the full format description with every request.

| Set | What it is | Tasks | VIC-2B | Reference model |
|---|---|---:|---:|---:|
| `evals/real1` | Questions real people wrote, from Spider's validation split | 681 | 83.0% | 81.6% |
| `evals/real2` | The same kind from a second source, BIRD's dev split | 195 | 66.7% | 65.6% |
| `evals/test1` | Hand-written test set, requests that spell out their output | 173 | 82.1% | 93.6% |
| `evals/v2` | Development set | 245 | 85.7% | 97.1% |

How to read it:

- `real1` and `real2` are scored ignoring column names, because their questions name none.
- On `real1`, 52 tasks are right only in VIC-2B and 43 only in the reference model (p = 0.41). On `real2` it is 25 against 23 (p = 0.89). Neither set can tell the two models apart.
- VIC-2B was trained on questions from Spider's train split. No question or database of `real1` is among them, but the wording is familiar. `real2` is the check on that: other authors, other databases, and nothing from BIRD was trained on. Both models fall by about the same amount there.
- `test1` and `v2` state exactly which columns to return and what to call them. The large model is ahead on that style.
- 173 tasks give a margin of about 6 points either way; 195 give about 7.

The stored answers of every model are in [`bench/results/`](../bench/results), one file per model and set, so each number can be recomputed without running a model. VIC-2B's files are named `vic__VIC-k5-mlx-6bit.jsonl`, after `k5`, the training run that became VIC-2B; the reference model's are `plan__-cf-openai-gpt-oss-120b.jsonl`.

```sh
uv run python -m bench.run --tasks evals/real1 report
```

## The eval sets

| Set | Tasks | Role |
|---|---:|---|
| `evals/v2` | 245 over 16 tables | Development: failures are read and acted on |
| `evals/test1` | 173 over 17 tables | Test: scored only. Its subjects appear in neither `v2` nor the training data |
| `evals/real_dev` | 250 | Development twin of `real1`, from Spider's train-split databases |
| `evals/real1` | 681 over 20 databases | Test: scored only |
| `evals/real2` | 195 over 6 databases | Test: scored only. It has no development twin |
| `evals/v1` | 250 | The first version, kept frozen because early runs were scored on it |

`real1` and `real_dev` come from the Spider text-to-SQL dataset (Yu et al., 2018, CC BY-SA 4.0); `evals/from_spider.py` builds them. `real2` comes from the BIRD benchmark (Li et al., 2023, CC BY-SA 4.0); `evals/from_bird.py` builds it under the same rules, and each request carries BIRD's line of evidence after "Note:". In all three the requests are the dataset's questions word for word and each expected result is what the dataset's own SQL returns. A question is kept when its SQL is one SELECT over one or two tables of at most 1,000 rows, and the answer does not depend on the order the rows are stored in.

Each task is a request, its input tables, and a frozen expected result. A test set is scored and never steered by: its failures are not read to decide what to change. `evals/real_dev` has a known weakness for VIC-2B: most of its questions have a second wording with the same SQL in Spider's train split, which the model was trained on, so its score there is flattering.

`evals/v2` by category:

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

**Keep these sets held out.** Do not generate training data from their tables or requests.

## How scoring works

Scoring runs the response and compares the resulting table with the expected one, so any plan that produces the right table is correct. Row order matters only where the request asks for an order, column order never matters, and numbers match within 1e-9. The report gives accuracy overall, by category and by tag, and a second figure, `accuracy_ignoring_column_names`, that forgives different column names.

## Score your own planner

A planner sees the request plus a profile of each table: the row count, column names and types, null counts, three sample rows, and the distinct values of text columns that have twenty or fewer. It also needs the format description from `vasuki schema`.

```sh
uv run vasuki eval-inputs evals/v2 > inputs.jsonl      # what the planner sees, one task per line
# ...run your planner, writing {"id": ..., "response": {...}} per line to predictions.jsonl...
uv run vasuki eval evals/v2 predictions.jsonl --details
```

## Benchmark a model

`bench/` runs a model over an eval set and reports accuracy and token use. It needs the `bench` dependency group (`anthropic`, `pandas`), which `uv sync` installs by default.

```sh
uv run python -m bench.run run --arm plan --provider anthropic --model claude-opus-5-5
uv run python -m bench.run run --arm code --provider anthropic --model claude-opus-5-5
uv run python -m bench.run run --arm vic --provider mlx --model rahul05ranjan/VIC-2B --label VIC-2B
uv run python -m bench.run delegate --provider anthropic --model claude-opus-5-5
uv run python -m bench.run report --details
```

| Piece | What it measures |
|---|---|
| `--arm vic` | A VIC model with its short system prompt, as `vasuki ask` uses it |
| `--arm plan` | A general model writes plans from the full format description. Its accuracy is the ceiling for a planner using this format |
| `--arm code` | The model writes pandas itself. Skips the `unsupported` category |
| `delegate` | No model call. Sizes the call a caller would write and the report it would read when delegating |
| `report` | Scores every results file |

Providers:

- `mlx` runs a local model on Apple silicon.
- `ollama` runs a model in the local Ollama, for example `--model hf.co/rahul05ranjan/VIC-2B-GGUF:Q8_0`.
- `cloudflare` needs `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`. It serves Workers AI models (`--model @cf/openai/gpt-oss-120b`).
- `anthropic` needs `ANTHROPIC_API_KEY`.

Credentials can go in a `.env` file in the checkout as `KEY=value` lines; the runner reads it and `.gitignore` excludes it.

It runs `evals/v2` unless `--tasks` says otherwise. Results go to `bench/results/<eval set>/<arm>__<model>.jsonl`. A run resumes from that file, so stopping and restarting never repeats a finished task. `--per-category N` runs a small sample first.

The code arm executes model-written Python in a subprocess with a time limit, in an empty temporary directory. It is not isolated from the network or the rest of the file system.
