"""Build data/train_mix: every intent-level rewrite, every clarification and refusal, and a tenth of the
original procedural plans so the model keeps what it already does well."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path("data")
out = root / "train_mix"
out.mkdir(exist_ok=True)
read = lambda p: [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
originals = read(root / "train_all" / "examples.jsonl")
rewrites = read(root / "train_intent" / "examples.jsonl")
keep = lambda e: e["response"]["status"] != "plan" or int(hashlib.md5(e["id"].encode()).hexdigest(), 16) % 10 == 0
mix = rewrites + [e for e in originals if keep(e)]
(out / "examples.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in mix), encoding="utf-8")
shutil.copy(root / "train_all" / "worlds.jsonl", out / "worlds.jsonl")
(out / "rejects.jsonl").write_text("", encoding="utf-8")
print(json.dumps({"rewrites": len(rewrites), "originals_kept": len(mix) - len(rewrites), "total": len(mix)}))
