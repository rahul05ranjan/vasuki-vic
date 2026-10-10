# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 1.x | Yes |
| Earlier | No |

## Reporting a vulnerability

Please report it privately, not in a public issue.

Use GitHub's private reporting: open the repository's **Security** tab and choose **Report a vulnerability**, or go to https://github.com/rahul05ranjan/vasuki-vic/security/advisories/new.

Include what you found, how to reproduce it, and what an attacker could do with it. You will get a reply as soon as the report has been read. Please give a fix a reasonable time before disclosing publicly.

## What is in scope

vasuki runs plans written by a model against files on the user's machine, so these matter most:

- **Escaping the expression language.** `where` and `expr` are parsed to a syntax tree and interpreted against a whitelist. Anything that makes an expression run arbitrary code, read a file, or reach the network is a vulnerability.
- **Reading or writing outside what was asked.** The engine should read only the input files it is given and write only the output path it is given.
- **A plan that exhausts the machine.** A small plan that makes the engine use unbounded memory or time.
- **The MCP server.** It reads any path a client hands it and writes `output_path`, with the permissions of whoever started it. That is by design and documented; a way to exceed those permissions is a vulnerability.

## What is not a vulnerability

- **A wrong answer.** The model returns a wrong table for about one request in seven. That is a documented limit of the model, not a security defect. Report it as an ordinary issue.
- **The benchmark's code arm.** `bench/` runs model-written Python in a subprocess with a time limit. It is a research tool, it is documented as not isolated, and it is not part of the installed program.

## How the engine limits what a plan can do

- A plan is validated in full against the input columns before any step runs.
- Expressions have no indexing, no slicing, no `**`, and no access to Python names or builtins.
- The engine imports nothing outside the Python standard library.
- vasuki sends no telemetry. The only network use is downloading the model on first run, and talking to the server named with `--url` or `--ollama`.
