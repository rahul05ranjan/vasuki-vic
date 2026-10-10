# Contributing

Thank you for considering a contribution. This page says how to set up, what a change needs before it is merged, and the few rules that protect the measurements.

## Set up

You need [uv](https://docs.astral.sh/uv/) and Python 3.11 to 3.14.

```sh
git clone https://github.com/rahul05ranjan/vasuki-vic && cd vasuki-vic
uv sync
uv run pytest
```

The tests take a few seconds and need no model and no network.

## What a change needs

- **Tests.** A new operation, expression function or command comes with tests in `tests/`. A bug fix comes with a test that failed before it.
- **A passing suite.** `uv run pytest` must pass. It also checks the integrity of the eval sets: every gold plan reproduces its frozen result, and removing any step from a gold plan changes the answer.
- **Documentation.** If behaviour changes, update the page in `docs/` that describes it, and add a line to `CHANGELOG.md`.
- **No new dependencies in the engine.** The `vasuki` package's engine imports nothing outside the standard library. Keep it that way; a planner or benchmark dependency belongs in an optional group.

## The rules that protect the measurements

- **Do not edit `evals/v1`, `evals/test1`, `evals/real1` or `evals/real2`.** They are frozen. Models have been scored on them, and a change would make old and new numbers incomparable.
- **Do not build training data from anything under `evals/`.** The sets are held out.
- **Report scores with the stored answers.** A model's results file goes in `bench/results/<eval set>/`, so anyone can recompute the number.

## Proposing a change

1. For anything larger than a small fix, open an issue first and describe the problem. It saves you writing something that cannot be merged.
2. Fork, branch, and keep the change to one subject.
3. Open a pull request that says what changed and how you checked it.

## Reporting bugs

Open an issue with the command you ran, the plan if there is one (`--show-plan`), what you expected and what you got. A few rows of data that reproduce the problem help most; remove anything private first.

A wrong answer from the model is worth reporting too. Include the request, the plan it wrote and the table's column names.

## Security

Do not report a vulnerability in a public issue. See [SECURITY.md](SECURITY.md).

## Licence

By contributing you agree that your contribution is licensed under the Apache License 2.0, the licence of this repository.
