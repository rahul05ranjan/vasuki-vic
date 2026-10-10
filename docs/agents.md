# Using vasuki from an agent

Two ways to give an agent vasuki: a skill for coding agents that can run a shell command, and an MCP server for any MCP client.

## The skill

The skill in [`skills/vasuki`](../skills/vasuki/SKILL.md) tells a coding agent when to hand a question to `vasuki ask`, how to read the answer, and how to check the plan before trusting the rows.

```sh
npx skills add rahul05ranjan/vasuki-vic
```

That is the only step. It installs the skill for Claude Code, Codex, Cursor and the other agents the [skills](https://skills.sh) installer knows, and needs Node.js for `npx`.

**What it installs.** One file of instructions, `SKILL.md`. The program and the model are not installed by that command: the skill has the agent run the `uvx` line from the README, which fetches both the first time a question is asked (1.55 GB, about two minutes) and takes a few seconds after that. If `uv` is missing, the skill has the agent offer to install it. The model itself needs a Mac with Apple silicon, or a server given with `--ollama` or `--url`.

**How to use it.** Ask for it by name: "use the vasuki skill: which vendor has the highest total in invoices.csv?". An agent that can already run Python will usually write its own script unless asked. The skill is worth asking for when the file is large and you want its rows kept out of the agent's context.

**What the agent is told to check.** About one of the model's answers in seven is wrong without warning, so the skill has the agent read the plan before it trusts the rows: every condition in the request has its filter, the sort order matches the words, a question about two files has a join, and each metric uses the function and column asked for. When a check fails, the agent corrects the plan and runs it with `vasuki run -`.

## MCP server

`vasuki-mcp` is a stdio MCP server. An agent sends a request and file paths; only the first rows of the result come back, so the data stays out of the agent's context.

```sh
claude mcp add vasuki -- uv run --directory /path/to/vasuki-vic vasuki-mcp     # Claude Code
```

For another client, register the same command as a stdio server:

```json
{"mcpServers": {"vasuki": {"command": "uv", "args": ["run", "--directory", "/path/to/vasuki-vic", "vasuki-mcp"],
                           "env": {"VASUKI_MODEL": "rahul05ranjan/VIC-2B"}}}}
```

| Tool | What it does |
|---|---|
| `compute(request, files, output_path?, preview_rows?)` | Plans the request with VIC and runs it. Returns the same answer as `vasuki ask`, with the plan |
| `describe_tables(files)` | Row count, columns, types and three sample rows per file |
| `run_plan(steps, files, output_path?, preview_rows?)` | Runs a plan the agent wrote itself, for when `compute` gets it wrong |
| `plan_format()` | The plan format, for writing plans by hand (about 2,000 tokens) |

The model loads on the first `compute` and stays in memory.

| Variable | What it sets |
|---|---|
| `VASUKI_MODEL` | The VIC model: a local folder or a Hugging Face name. Default `rahul05ranjan/VIC-2B` |
| `VASUKI_URL` | Use an OpenAI-style chat endpoint instead of the local MLX model. `VASUKI_MODEL` then names the model on that server. For Ollama: `http://localhost:11434/v1` with `VASUKI_MODEL=hf.co/rahul05ranjan/VIC-2B-GGUF:Q8_0` |
| `VASUKI_API_KEY` | Sent as the bearer token when `VASUKI_URL` is set |
| `VASUKI_DESCRIBE` | Set to `1` to send the full plan format with each request, for a model not trained on it |

**Permissions.** The server reads any file path it is given and writes `output_path`, with the permissions of whoever starts it. Run it as a user who may read only what the agent should read.
