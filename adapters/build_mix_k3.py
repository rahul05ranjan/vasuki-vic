"""Build data/train_mix_k3 for Kaggle run k3: everything in data/train_mix_k, plus the targeted batch under the
same rule (every natural-reference rewrite, every intent rewrite, every clarification and refusal, and 30% of
the original long requests)."""
import hashlib
import json
from pathlib import Path

root = Path("data")
out = root / "train_mix_k3"
out.mkdir(exist_ok=True)
read = lambda p: [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
bucket = lambda e: int(hashlib.md5(e["id"].encode()).hexdigest(), 16) % 100
base = read(root / "train_mix_k" / "examples.jsonl")
originals, intent, natural = (read(root / d / "examples.jsonl")
                              for d in ("train_target", "train_target_intent", "train_target_natural"))
kept = [e for e in originals if e["response"]["status"] != "plan" or bucket(e) < 30]
# An author sometimes numbers two examples alike, so an id can repeat. Exact copies are dropped from the new
# batch; two different examples under one id are both kept. data/train_mix_k is left exactly as k2 saw it.
seen, added = set(), []
for example in natural + intent + kept:
    key = (example["id"], example["request"])
    if key not in seen:
        seen.add(key)
        added.append(example)
mix = base + added
worlds = read(root / "train_mix_k" / "worlds.jsonl") + read(root / "train_target" / "worlds.jsonl")
(out / "examples.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in mix), encoding="utf-8")
(out / "worlds.jsonl").write_text("".join(json.dumps(w, ensure_ascii=False) + "\n" for w in worlds), encoding="utf-8")
(out / "rejects.jsonl").write_text("", encoding="utf-8")
print(json.dumps({"from_mix_k": len(base), "target_natural": len(natural), "target_intent": len(intent),
                  "target_long_and_statuses": len(kept), "target_added": len(added), "total": len(mix)}))
