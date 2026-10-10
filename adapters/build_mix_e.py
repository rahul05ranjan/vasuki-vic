"""Build data/train_mix_e for run E: every natural-reference rewrite, 40% of the intent rewrites, every
clarification and refusal, and 5% of the original procedural plans."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path("data")
out = root / "train_mix_e"
out.mkdir(exist_ok=True)
read = lambda p: [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
bucket = lambda e: int(hashlib.md5(e["id"].encode()).hexdigest(), 16) % 100
originals, intent, natural = (read(root / d / "examples.jsonl") for d in ("train_all", "train_intent", "train_natural"))
mix = (natural + [e for e in intent if bucket(e) < 40]
       + [e for e in originals if e["response"]["status"] != "plan" or bucket(e) < 5])
(out / "examples.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in mix), encoding="utf-8")
shutil.copy(root / "train_all" / "worlds.jsonl", out / "worlds.jsonl")
(out / "rejects.jsonl").write_text("", encoding="utf-8")
print(json.dumps({"natural": len(natural), "intent_total": len(intent), "mix_total": len(mix)}))
