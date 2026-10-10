"""Turn a Kaggle run's PEFT adapter into models that run on this Mac.

    uv run python kaggle_job/to_mlx.py --label k2

Merges the adapter into the full-precision base with PyTorch, saves that as a standard Transformers
model in models/VIC-<label>-merged, then converts it to 4-bit MLX in models/VIC-<label>-mlx-4bit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def mlx_source(merged_dir: Path, base: str) -> Path:
    """The folder to hand MLX-LM's converter.

    A base such as Qwen3.5 is a vision-language model whose text part Transformers loads alone. The merged
    model then carries the text-only config (`qwen3_5_text`), which MLX-LM does not know, although the weights
    keep the full model's names. For such a model the converter gets the base's own config beside links to
    the merged files.
    """
    config = json.loads((merged_dir / "config.json").read_text())
    if not config.get("model_type", "").endswith("_text"):
        return merged_dir
    from huggingface_hub import hf_hub_download

    source = merged_dir.with_name(merged_dir.name + "-mlxsrc")
    shutil.rmtree(source, ignore_errors=True)
    source.mkdir()
    for file in merged_dir.iterdir():
        if file.name != "config.json":
            (source / file.name).symlink_to(file)
    shutil.copy(hf_hub_download(base, "config.json"), source / "config.json")
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="run label whose output was fetched, e.g. k2")
    parser.add_argument("--bits", type=int, default=4, help="MLX quantisation bits (default 4)")
    parser.add_argument("--convert-only", action="store_true", help="reuse models/VIC-<label>-merged; skip the merge")
    args = parser.parse_args()

    adapter = ROOT / "kaggle_job" / "output" / args.label / "adapter"
    base = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
    merged_dir = ROOT / "models" / f"VIC-{args.label}-merged"
    mlx_dir = ROOT / "models" / f"VIC-{args.label}-mlx-{args.bits}bit"

    if not args.convert_only:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
        merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        shutil.rmtree(merged_dir, ignore_errors=True)
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)

    shutil.rmtree(mlx_dir, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(mlx_source(merged_dir, base)),
                    "--mlx-path", str(mlx_dir), "-q", "--q-bits", str(args.bits)], check=True)
    size = lambda d: round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9, 2)
    print(json.dumps({"base": base, "merged": str(merged_dir.relative_to(ROOT)), "merged_gb": size(merged_dir),
                      "mlx": str(mlx_dir.relative_to(ROOT)), "mlx_gb": size(mlx_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
