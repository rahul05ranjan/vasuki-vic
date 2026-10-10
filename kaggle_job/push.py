"""Run VIC training on Kaggle from this machine.

    uv run python kaggle_job/push.py bundle --label e1 --train data/train_mix_e --eval evals/v2 --eval data/train_all/valid_eval
    uv run python kaggle_job/push.py push --label e1     # upload the bundle as a private dataset and start the kernel
    uv run python kaggle_job/push.py status --label e1
    uv run python kaggle_job/push.py fetch --label e1    # download the adapter, scores and predictions

Needs KAGGLE_USERNAME plus KAGGLE_API_TOKEN (or the legacy KAGGLE_KEY) in the environment or in .env.
The bundle is flat on purpose: Kaggle unpacks uploaded archives in ways that are easy to get wrong, so
package files and eval sets travel as `vasuki_src__<file>` and `eval__<set>__<file>`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLES = ROOT / "kaggle_job" / "_bundle"


def run_names(label: str) -> tuple[Path, str, str]:
    """Each run label gets its own bundle folder, dataset and kernel, so runs never overwrite each other."""
    return BUNDLES / label, f"vic-data-{label}", f"vic-train-{label}"


def load_env(path: Path = ROOT / ".env") -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    # Kaggle's newer tokens start with KGAT_ and are read from KAGGLE_API_TOKEN, whatever name they were saved under.
    if os.environ.get("KAGGLE_KEY", "").startswith("KGAT_"):
        os.environ.setdefault("KAGGLE_API_TOKEN", os.environ["KAGGLE_KEY"])


def user() -> str:
    name = os.environ.get("KAGGLE_USERNAME")
    if not name or not (os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")):
        sys.exit("add KAGGLE_USERNAME and KAGGLE_API_TOKEN (or the legacy KAGGLE_KEY) to .env first")
    return name


def kaggle(*args: str, check: bool = True) -> str:
    done = subprocess.run([str(Path(sys.executable).parent / "kaggle"), *args], capture_output=True, text=True)
    output = (done.stdout + done.stderr).strip()
    if check and done.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{output}")
    return output


def eval_name(folder: Path) -> str:
    """evals/v2 -> evals_v2; data/train_all/valid_eval -> train_all_valid."""
    parts = [p for p in folder.parts if p not in ("data", "valid_eval")]
    return "_".join(parts) + ("_valid" if folder.name == "valid_eval" else "")


def bundle(args) -> dict:
    """Collect everything the kernel needs into one flat folder."""
    folder, _, _ = run_names(args.label)
    shutil.rmtree(folder, ignore_errors=True)
    data = folder / "data"
    data.mkdir(parents=True)
    train = ROOT / args.train / "mlx"
    for name in ("train.jsonl", "valid.jsonl"):
        shutil.copy(train / name, data / name)
    for source in (ROOT / "src" / "vasuki").glob("*.py"):
        shutil.copy(source, data / f"vasuki_src__{source.name}")
    names = []
    for folder in map(Path, args.eval):
        name = eval_name(folder)
        names.append(name)
        for file in ("tasks.jsonl", "tables.json"):
            shutil.copy(ROOT / folder / file, data / f"eval__{name}__{file}")
    if args.init_adapter:
        # The kernel continues from this adapter instead of starting LoRA from zero.
        for file in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy(ROOT / args.init_adapter / file, data / f"adapter__{file}")
    config = {"base": args.base, "epochs": args.epochs, "learning_rate": args.learning_rate,
              "lora_rank": args.rank, "lora_alpha": 2 * args.rank, "batch_size": args.batch_size,
              "accumulate": args.accumulate, "eval_sets": names, "init_adapter": bool(args.init_adapter)}
    if args.smoke:
        config.update(train_limit=48, eval_limit=12, epochs=1.0)
    for item in args.set:
        key, _, value = item.partition("=")
        config[key] = json.loads(value)
    (data / "config.json").write_text(json.dumps(config, indent=1))
    size = sum(f.stat().st_size for f in data.iterdir()) / 1e6
    return {"bundle": str(data.relative_to(ROOT)), "megabytes": round(size, 1), "config": config,
            "train_examples": sum(1 for _ in (data / "train.jsonl").open())}


def push(args) -> dict:
    name = user()
    folder, DATASET, KERNEL = run_names(args.label)
    data = folder / "data"
    if not (data / "config.json").exists():
        sys.exit("no bundle yet: run `push.py bundle` first")
    (data / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET, "id": f"{name}/{DATASET}", "licenses": [{"name": "CC0-1.0"}]}))
    # Datasets made through the API are private unless --public is passed.
    exists = kaggle("datasets", "status", f"{name}/{DATASET}", check=False)
    if "ready" in exists.lower():
        uploaded = kaggle("datasets", "version", "-p", str(data), "-m", args.message)
    else:
        uploaded = kaggle("datasets", "create", "-p", str(data))

    # A kernel started while Kaggle is still processing the upload runs without the dataset attached and
    # fails in its first minute (run k3, 2026-10-08). Wait until the dataset reports ready.
    deadline = time.time() + 900
    while "ready" not in kaggle("datasets", "status", f"{name}/{DATASET}", check=False).lower():
        if time.time() > deadline:
            sys.exit(f"dataset {name}/{DATASET} was not ready after 15 minutes; run `push` again later")
        time.sleep(10)
    time.sleep(20)  # ready is reported a little before a new kernel can mount it

    kernel = folder / "kernel"
    kernel.mkdir(exist_ok=True)
    shutil.copy(ROOT / "kaggle_job" / "train_vic.py", kernel / "train_vic.py")
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{name}/{KERNEL}", "title": KERNEL, "code_file": "train_vic.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "machine_shape": args.accelerator, "dataset_sources": [f"{name}/{DATASET}"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=1))
    started = kaggle("kernels", "push", "-p", str(kernel), "--accelerator", args.accelerator)
    return {"dataset": uploaded.splitlines()[-1] if uploaded else "", "kernel": started.splitlines()[-1] if started else ""}


def status(args) -> dict:
    return {"status": kaggle("kernels", "status", f"{user()}/{run_names(args.label)[2]}")}


def fetch(args) -> dict:
    out = ROOT / "kaggle_job" / "output" / args.label
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{user()}/{run_names(args.label)[2]}", "-p", str(out))
    summary = out / "summary.json"
    return {"saved_to": str(out.relative_to(ROOT)), "files": sorted(p.name for p in out.iterdir()),
            "evals": json.loads(summary.read_text())["evals"] if summary.exists() else None}


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(prog="kaggle_job/push.py", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    bundling = commands.add_parser("bundle", help="collect the training data, eval sets and package into one folder")
    bundling.add_argument("--train", required=True, help="data folder that has been exported, e.g. data/train_mix_e")
    bundling.add_argument("--eval", action="append", default=[], help="eval set folder; repeat for several")
    bundling.add_argument("--base", default="Qwen/Qwen3-1.7B", help="Hugging Face model to fine-tune")
    bundling.add_argument("--epochs", type=float, default=2.0)
    bundling.add_argument("--learning-rate", type=float, default=1e-4)
    bundling.add_argument("--rank", type=int, default=16)
    bundling.add_argument("--batch-size", type=int, default=4)
    bundling.add_argument("--accumulate", type=int, default=2)
    bundling.add_argument("--init-adapter", metavar="FOLDER",
                          help="continue from an earlier run's adapter, e.g. kaggle_job/output/k4/adapter")
    bundling.add_argument("--smoke", action="store_true", help="a few examples and steps, to test the script")
    bundling.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                          help="any other train_vic.py setting, e.g. --set gradient_checkpointing=false")
    pushing = commands.add_parser("push", help="upload the bundle and start the kernel")
    pushing.add_argument("--accelerator", default="NvidiaTeslaT4")
    pushing.add_argument("--message", default="update")
    commands.add_parser("status", help="show the kernel's run status")
    fetching = commands.add_parser("fetch", help="download the kernel's output")
    for command in (bundling, pushing, commands.choices["status"], fetching):
        command.add_argument("--label", required=True, help="short name for this run, e.g. e1; lowercase letters, digits, dashes")
    args = parser.parse_args()
    print(json.dumps({"bundle": bundle, "push": push, "status": status, "fetch": fetch}[args.command](args), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
