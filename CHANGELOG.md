# Changelog

All notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-10

The first stable release.

### Added

- `vasuki ask --ollama`: run the model in the Ollama on this machine, on macOS, Linux or Windows. The model is fetched into Ollama the first time.
- VIC-2B as an 8-bit GGUF file, [`rahul05ranjan/VIC-2B-GGUF`](https://huggingface.co/rahul05ranjan/VIC-2B-GGUF), for Ollama and llama.cpp.
- Documentation in `docs/`: usage, the plan format, agents, and evaluation.
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue and pull request templates, and continuous integration on Linux and macOS.

### Changed

- The model is now [`rahul05ranjan/VIC-2B`](https://huggingface.co/rahul05ranjan/VIC-2B). It was published as `VIC-2B-preview`; the old name still resolves. The weights are unchanged.
- A machine that cannot run the model itself is now pointed to `--ollama`.

## [0.1.0] - 2026-10-10

The first public release.

### Added

- The plan format: 12 operations, an expression language and unit conversion, described by `vasuki schema`.
- The engine: a plan is checked in full against the input columns, then run. Standard library only.
- `vasuki ask`: a planner model writes the plan and the engine runs it. Works with a VIC model through MLX on Apple silicon, or with any OpenAI-style server through `--url`.
- `vasuki-mcp`, a stdio MCP server with `compute`, `describe_tables`, `run_plan` and `plan_format`.
- An agent skill, installed with `npx skills add rahul05ranjan/vasuki-vic`.
- Five eval sets and the benchmark harness in `bench/`, with the stored answers of every model scored.
- The VIC-2B model: Qwen3.5-2B fine-tuned to write plans, 1.55 GB at 6 bits.

[1.0.0]: https://github.com/rahul05ranjan/vasuki-vic/releases/tag/v1.0.0
[0.1.0]: https://github.com/rahul05ranjan/vasuki-vic/commit/5a1f7db
