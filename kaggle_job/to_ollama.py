"""Install a run's model in Ollama on this machine.

    uv run python kaggle_job/to_ollama.py --label k5        # makes the Ollama model vic-k5

Needs models/VIC-<label>-merged, which kaggle_job/to_mlx.py writes, and a running Ollama new enough to
import safetensors (0.19 or later; tried on 0.40). Ollama quantises the merged model itself. Then:

    uv run vasuki ask "..." -i data.csv --url http://localhost:11434/v1 --model vic-k5

Two things matter, and both were found by trying:
- 8 bits. Ollama offers 4 or 8, and 4 bits cost a VIC model almost 6 points
  (run k2 on evals/v2: 82.9% at 6 and at 8 bits, 77.1% at 4).
- Reasoning must be off. Ollama turns it on by default for these bases, and nothing in a Modelfile
  changes that. A VIC model was trained with an empty reasoning block: asked to reason, it argues with
  the request and returns no plan. `vasuki ask --url` and `bench.run --provider ollama --arm vic` turn
  it off themselves; anything else must send `"think": false` (or `"reasoning_effort": "none"` on the
  OpenAI-style endpoint), and `ollama run` needs `--think=false`. Swapping in the `qwen3-coder`
  renderer looks like a way round this and is not: the model then reasons on some requests anyway.
A VIC model is not a chat model: it expects the request and a profile of the tables as JSON, which is
what `vasuki ask` sends. The system prompt is stored with the model, so a hand-made request needs only
that JSON as its message.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vasuki.request import VIC_SYSTEM  # noqa: E402


def modelfile(merged: Path) -> str:
    assert '"""' not in VIC_SYSTEM
    # Named outright because Ollama picks `qwen3-coder` for a Qwen3 base, which has no reasoning switch at
    # all; `qwen3.5` writes the prompt both bases were trained on once reasoning is turned off.
    return (f"FROM {merged}\n"
            "RENDERER qwen3.5\nPARSER qwen3.5\n"
            f'SYSTEM """{VIC_SYSTEM}"""\n'
            "PARAMETER temperature 0\nPARAMETER num_predict 400\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="run label whose merged model is in models/, e.g. k5")
    parser.add_argument("--bits", choices=("int8", "int4", "mxfp8", "mxfp4", "nvfp4"), default="int8",
                        help="Ollama's quantisation level (default int8)")
    parser.add_argument("--name", help="Ollama model name (default vic-<label>)")
    args = parser.parse_args()
    merged = ROOT / "models" / f"VIC-{args.label}-merged"
    if not (merged / "model.safetensors").exists():
        sys.exit(f"{merged} is missing: run kaggle_job/to_mlx.py --label {args.label} first")
    name = args.name or f"vic-{args.label}"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "Modelfile"
        path.write_text(modelfile(merged), encoding="utf-8")
        done = subprocess.run(["ollama", "create", name, "-f", str(path), "-q", args.bits])
    if done.returncode:
        sys.exit(f"ollama create failed for {name}")
    print(f"installed {name}: uv run vasuki ask \"...\" -i FILE --url http://localhost:11434/v1 --model {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
