"""Build data/train_mix_k5 for Kaggle run k5, which continues from k4's adapter: every example made from
Spider's human-written questions, plus 30% of data/train_mix_k3 as a reminder of what k4 already knows."""
import hashlib
import json
from pathlib import Path

root = Path("data")
out = root / "train_mix_k5"
out.mkdir(exist_ok=True)
read = lambda p: [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
bucket = lambda e: int(hashlib.md5(e["id"].encode()).hexdigest(), 16) % 100
spider = read(root / "train_spider" / "examples.jsonl")
replay = [e for e in read(root / "train_mix_k3" / "examples.jsonl") if bucket(e) < 30]
mix = spider + replay
worlds = read(root / "train_spider" / "worlds.jsonl") + read(root / "train_mix_k3" / "worlds.jsonl")
(out / "examples.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in mix), encoding="utf-8")
(out / "worlds.jsonl").write_text("".join(json.dumps(w, ensure_ascii=False) + "\n" for w in worlds), encoding="utf-8")
(out / "rejects.jsonl").write_text("", encoding="utf-8")
print(json.dumps({"spider": len(spider), "replay_of_mix_k3": len(replay), "total": len(mix)}))
