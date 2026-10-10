"""Build the public copy of this repository in a folder.

    uv run python tools/publish.py OUT_FOLDER

The working repository holds a private diary (TRACKER.md), instructions for coding agents, run logs and
scripts with local paths. The public copy is the tracked files named by INCLUDE and nothing else, and the
build stops if any of them still carries something private. Pushing the folder anywhere is a separate,
deliberate step.
"""

from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = [
    "README.md", "LICENSE", "NOTICE", "pyproject.toml", "uv.lock", ".gitignore",
    "src/*", "tests/*", "examples/*", "evals/*", "skills/*", "datagen/*.py", "tools/*.py",
    "bench/*.py", "bench/results/*.jsonl", "bench/results/*.json",
    "kaggle_job/*.py", "adapters/*.py",
]
# Words that must not appear in the public copy: the diary's name, this machine's paths and accounts.
PRIVATE = re.compile(r"TRACKER|AGENTS\.md|CLAUDE\.md|/Users/|rivansh|rahulswt7|rahulsingh|hf_[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9]{8,}")
ALLOWED = {"tools/publish.py"}      # this file names those words in order to refuse them


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1]).resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty")
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    chosen = [name for name in tracked if name and any(fnmatch.fnmatch(name, pattern) for pattern in INCLUDE)]
    found = []
    for name in chosen:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        if name in ALLOWED:
            continue
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found += [f"{name}: {match.group(0)}" for match in PRIVATE.finditer(text)]
    if found:
        shutil.rmtree(out)
        sys.exit("private content in files meant for the public copy:\n  " + "\n  ".join(sorted(set(found))[:40]))
    left_out = sorted({name.split("/")[0] for name in tracked if name and name not in chosen})
    print(f"{len(chosen)} files in {out}; left out, wholly or partly: {', '.join(left_out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
