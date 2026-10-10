"""Build data/train_mix_g1 for a run that starts a new base from zero: everything in data/train_mix_k3 plus
every example made from Spider's human-written questions."""
import json
from pathlib import Path

root = Path("data")
out = root / "train_mix_g1"
out.mkdir(exist_ok=True)
read = lambda p: [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
pinned, spider = read(root / "train_mix_k3" / "examples.jsonl"), read(root / "train_spider" / "examples.jsonl")
worlds = read(root / "train_mix_k3" / "worlds.jsonl") + read(root / "train_spider" / "worlds.jsonl")
(out / "examples.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in pinned + spider), encoding="utf-8")
(out / "worlds.jsonl").write_text("".join(json.dumps(w, ensure_ascii=False) + "\n" for w in worlds), encoding="utf-8")
(out / "rejects.jsonl").write_text("", encoding="utf-8")
print(json.dumps({"from_mix_k3": len(pinned), "spider": len(spider), "total": len(pinned) + len(spider)}))
