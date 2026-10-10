---
name: vasuki
description: Answer a question about a CSV, TSV, JSON or JSONL table with one local command, in place of reading the file or writing a script. Use when the user wants totals, counts, averages, a filter, a ranking or a join from data files and the answer is a table.
---

# vasuki

`vasuki ask` hands a question about data files to VIC, a small model on this machine. VIC sees each file's column names, types and three sample rows, and writes a plan of typed steps; a deterministic executor runs the plan. The rows of the result come back and the file stays out of your context.

Reach for it when the file is larger than you would read. A file of a few dozen rows is cheaper to read.

## Ask

```sh
uvx --from git+https://github.com/rahul05ranjan/vasuki-vic vasuki ask "REQUEST" -i FILE --show-plan
```

- Write `REQUEST` as one sentence that spells out the output: what to compute, per what, over which rows, in what order. "Total amount per vendor for invoices from 2025, highest first."
- Give one `-i` per file, two for a join. A table is named after its file stem; `-i orders=exports/2025.csv` names it yourself.
- The answer carries the first 20 rows (`--preview N` changes that). `-o out.csv` writes every row to a file.
- Installing this skill installed only these instructions. The command above fetches the program and the VIC model (1.55 GB) the first time it runs, about two minutes: tell the user and allow the command five minutes. Later runs take a few seconds.
- If `uvx` is not found, install uv with the user's yes (`curl -LsSf https://astral.sh/uv/install.sh | sh`, then `~/.local/bin/uvx` until the shell restarts).
- The model runs on a Mac with Apple silicon. On another machine add `--url http://localhost:11434/v1 --model NAME` for any OpenAI-style server that holds a VIC model. With neither, do the work yourself.

## Read the status

| `status` | Exit code | What to do |
|---|---|---|
| `ok` | 0 | Check the plan (below), then use `rows` |
| `needs_clarification` | 3 | Answer its `question` from what you know, or put it to the user, and ask again with the detail in the request |
| `unsupported` | 3 | The operations cannot do this; `reason` says why. Write the code yourself |
| `failed` | 1 | The model's reply was not a plan that runs. Write the code yourself |

## Check the plan

About one request in seven comes back `ok` with the wrong table, and nothing in `rows` shows it. The `plan` is the evidence. It passes when every part of the request has its step:

- each condition in the request is a `filter` on the right column and value
- the `sort` order matches the words ("highest first" is `desc`)
- a question about two files has a `join`
- each metric uses the function and column asked for (a count is not a sum)
- `row_count` fits the question (one named customer gives a few rows, not the whole file)

When a check fails, correct the plan and run it, which costs less than starting over:

```sh
echo '{"status": "plan", "steps": [...]}' | uvx --from git+https://github.com/rahul05ranjan/vasuki-vic vasuki run - -i FILE
```

`vasuki schema` prints every operation and expression function. If the corrected plan does not run, write the code yourself.

A request that compares one group of rows with another ("tasks that model A got right and model B got wrong") is beyond VIC today. Write the code for those from the start.
