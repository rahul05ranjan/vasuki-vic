"""Fine-tune a VIC planner with LoRA on an NVIDIA GPU, then score it. Runs as a Kaggle script kernel.

Everything it needs arrives in one attached dataset, built by kaggle_job/push.py: config.json, the
training examples, the eval sets, and the vasuki package source. It writes the adapter, per-set scores
and raw predictions to the kernel's output folder.

To try it away from Kaggle, point VIC_DATA at a bundle folder and VIC_OUT at an output folder.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if importlib.util.find_spec("torchao") is not None and "VIC_DATA" not in os.environ:
    # Kaggle's image ships a torchao too old for its PEFT, and PEFT refuses to build LoRA layers while
    # that version is installed. Nothing here uses torchao, so it is removed before PEFT is imported.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)



def find_data() -> Path:
    if os.environ.get("VIC_DATA"):
        return Path(os.environ["VIC_DATA"])
    found = glob.glob("/kaggle/input/**/config.json", recursive=True)
    if not found:
        sys.exit("no config.json under /kaggle/input: is the vic-data dataset attached?")
    return Path(found[0]).parent


DATA = find_data()
REQUESTED = json.loads((DATA / "config.json").read_text())
if REQUESTED.get("pip") and "VIC_DATA" not in os.environ:
    # A newer base may need a newer library than Kaggle's image has. It must be installed before the import.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *REQUESTED["pip"]], check=True)

import peft  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402

DEFAULTS = {
    "base": "Qwen/Qwen3-1.7B",
    "epochs": 1.0,
    "learning_rate": 1e-4,
    "batch_size": 4,
    "accumulate": 2,
    "max_tokens": 1280,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "gradient_checkpointing": True,   # a T4 has 14.5GB; without this, 4 long examples do not fit
    "max_train_hours": 7.0,   # a Kaggle GPU session ends at 9 hours; leave room for the evals
    "eval_batch_size": 8,
    "max_new_tokens": 400,
    "eval_sets": [],
    "train_limit": None,      # smoke tests only
    "eval_limit": None,       # smoke tests only
    "seed": 1,
    "target_modules": None,   # layer names LoRA attaches to; None means TARGET_MODULES
    "pip": [],                # packages to upgrade before anything is imported, e.g. ["transformers>=4.57"]
    "init_adapter": False,    # True: the bundle carries an earlier run's adapter to continue from
    "drop_modules": [],       # parts of a multimodal base to discard before training, e.g. ["vision_tower"]
    "gpus": 1,                # 2: train on both of a Kaggle session's T4s, one copy of the model on each
}
# Qwen3's projections. Another architecture names its layers differently; give its list as `target_modules`,
# or one regular expression (a string) when the names also occur in parts that must be left alone.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


OUT = Path(os.environ.get("VIC_OUT", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)
CONFIG = {**DEFAULTS, **REQUESTED}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def unpack() -> Path:
    """Rebuild the folders the bundle flattened: the vasuki package and one folder per eval set."""
    work = OUT / "_bundle"
    for path in DATA.iterdir():
        # Split on the first separator only: the package has a file called __init__.py.
        kind, _, rest = path.name.partition("__")
        if kind == "vasuki_src":
            target = work / "vasuki" / rest
        elif kind == "adapter":
            target = work / "init_adapter" / rest
        elif kind == "eval":
            name, _, file = rest.partition("__")
            target = work / "evals" / name / file
        else:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)
    sys.path.insert(0, str(work))
    return work


WORK = unpack()
from vasuki.evals import load_tasks, score, summarize  # noqa: E402
from vasuki.request import VIC_SYSTEM, planner_input  # noqa: E402


def render(tokenizer, messages: list[dict], generation: bool) -> str:
    try:
        # Qwen's template reasons by default; a planner's reply is the JSON itself.
        return tokenizer.apply_chat_template(messages, add_generation_prompt=generation, tokenize=False,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=generation, tokenize=False)


def encode_examples(tokenizer, path: Path, limit: int | None) -> list[tuple[list[int], int]]:
    """Each example as (token ids, number of leading prompt tokens that take no loss)."""
    encoded, too_long = [], 0
    for line in path.read_text(encoding="utf-8").splitlines()[:limit]:
        messages = json.loads(line)["messages"]
        prompt = render(tokenizer, messages[:-1], generation=True)
        full = render(tokenizer, messages, generation=False)
        if not full.startswith(prompt):
            full = prompt + messages[-1]["content"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        ids = tokenizer(full, add_special_tokens=False).input_ids
        if len(ids) > CONFIG["max_tokens"] or ids[:len(prompt_ids)] != prompt_ids:
            too_long += 1
            continue
        encoded.append((ids, len(prompt_ids)))
    log(f"{path.name}: {len(encoded)} examples, {too_long} skipped as too long or misaligned")
    return encoded


def batches(examples: list, size: int, rng: random.Random) -> list[list]:
    """Batches of similar length, so little is wasted on padding, in a shuffled order."""
    ordered = sorted(examples, key=lambda e: len(e[0]))
    chunks = [ordered[i:i + size] for i in range(0, len(ordered), size)]
    rng.shuffle(chunks)
    return chunks


def collate(batch: list, pad: int, device) -> dict:
    width = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, (ids, prompt_len) in enumerate(batch):
        input_ids[row, :len(ids)] = torch.tensor(ids)
        mask[row, :len(ids)] = 1
        labels[row, prompt_len:len(ids)] = torch.tensor(ids[prompt_len:])
    return {"input_ids": input_ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}


def load_model(device: str):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(CONFIG["base"], dtype=dtype)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(CONFIG["base"], torch_dtype=dtype)
    for name in CONFIG["drop_modules"]:
        # A vision-language base loaded whole: its picture and sound parts take memory and are never used here.
        if getattr(getattr(model, "model", None), name, None) is not None:
            setattr(model.model, name, None)
    model.to(device)
    if CONFIG["gradient_checkpointing"]:
        # Trades some speed for a large cut in memory: activations are recomputed in the backward pass.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    lora = LoraConfig(r=CONFIG["lora_rank"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=CONFIG["lora_dropout"],
                      target_modules=CONFIG["target_modules"] or TARGET_MODULES, task_type="CAUSAL_LM")
    # The frozen base stays in half precision; PEFT keeps the trainable adapter weights in float32.
    if CONFIG["init_adapter"]:
        # Continue an earlier run: its rank and target layers come with it, and `lora` above is not used.
        model = PeftModel.from_pretrained(model, str(WORK / "init_adapter"), is_trainable=True)
    else:
        model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"{CONFIG['base']} on {device}: {trainable / 1e6:.1f}M trainable parameters, rank {CONFIG['lora_rank']}")
    return tokenizer, model


def train(tokenizer, replicas: list, devices: list[str]) -> dict:
    """Train the adapter of `replicas[0]`.

    With more than one replica, each holds the same model on its own device. Every round gives each of
    them one batch at the same time; the gradients of the others are added to the first one's, which
    takes the optimizer step, and the others then copy its adapter weights. An update still covers
    `accumulate` batches, so the arithmetic is the same as on one device, only done side by side.
    """
    model, device = replicas[0], devices[0]
    adapters = [[p for p in replica.parameters() if p.requires_grad] for replica in replicas]
    rng = random.Random(CONFIG["seed"])
    examples = encode_examples(tokenizer, DATA / "train.jsonl", CONFIG["train_limit"])
    size, accumulate = CONFIG["batch_size"], CONFIG["accumulate"]
    updates = math.ceil(len(examples) * CONFIG["epochs"] / (size * accumulate))
    optimizer = torch.optim.AdamW(adapters[0], lr=CONFIG["learning_rate"])
    schedule = get_cosine_schedule_with_warmup(optimizer, max(1, updates // 30), updates)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler.scale(torch.zeros((), device=device))    # sets the scaler up here, not in two threads at once
    deadline = time.time() + CONFIG["max_train_hours"] * 3600
    log(f"training: {len(examples)} examples, {CONFIG['epochs']} passes, batches of {size}, an update every "
        f"{accumulate}, on {len(replicas)} device(s)")

    def share_weights() -> None:
        with torch.no_grad():
            for others in adapters[1:]:
                for source, target in zip(adapters[0], others):
                    target.copy_(source)

    def gather_gradients() -> None:
        for others in adapters[1:]:
            for target, source in zip(adapters[0], others):
                if source.grad is not None:
                    moved = source.grad.to(target.device)
                    target.grad = moved if target.grad is None else target.grad.add_(moved)
                    source.grad = None

    def forward_backward(replica, where: str, chunk: list, share: int) -> float:
        batch = collate(chunk, tokenizer.pad_token_id, where)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = replica(**batch).loss
        scaler.scale(loss / share).backward()
        return loss.item()

    for replica in replicas:
        replica.train()
    share_weights()
    pool = ThreadPoolExecutor(len(replicas))
    start, step, seen, recent, history, report_at = time.time(), 0, 0, [], [], 25
    planned = int(len(examples) * CONFIG["epochs"])
    queue: list = []
    pending = 0  # micro-batches whose gradients are waiting for an optimizer update

    def update() -> None:
        nonlocal pending
        gather_gradients()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(adapters[0], 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        schedule.step()
        share_weights()
        pending = 0

    while seen < planned and time.time() < deadline:
        if not queue:
            queue = batches(examples, size, rng)
        # One batch per device, but never more than the current update still has room for.
        chunks = [queue.pop() for _ in range(min(len(replicas), len(queue), accumulate - pending))]
        running = [pool.submit(forward_backward, replica, where, chunk, accumulate)
                   for replica, where, chunk in zip(replicas, devices, chunks)]
        losses, out_of_memory = [], None
        for future in running:
            try:
                losses.append(future.result())
            except torch.OutOfMemoryError as e:
                out_of_memory = e
        if out_of_memory:
            if size == 1:
                raise out_of_memory
            # Halve the batch and keep the same number of examples per update, then carry on.
            out_of_memory = None
            for group in adapters:
                for parameter in group:
                    parameter.grad = None
            torch.cuda.empty_cache()
            size, accumulate, pending = max(1, size // 2), accumulate * 2, 0
            queue = batches(examples, size, rng)
            log(f"out of GPU memory: continuing with batches of {size}, an update every {accumulate}")
            continue
        step += len(chunks)
        seen += sum(len(chunk) for chunk in chunks)
        pending += len(chunks)
        recent.extend(losses)
        if pending >= accumulate or seen >= planned:
            update()
        if step >= report_at or seen >= planned:
            report_at = step + 25
            elapsed = time.time() - start
            mean = sum(recent) / len(recent)
            rate = seen / elapsed
            history.append({"step": step, "examples": seen, "loss": round(mean, 4), "minutes": round(elapsed / 60, 1)})
            log(f"{seen}/{planned} examples  loss {mean:.4f}  {rate:.2f} examples/s  "
                f"about {(planned - seen) / rate / 60:.0f} min left")
            recent = []
    total = planned
    step = seen
    pool.shutdown()
    if step < total:
        log(f"stopped after {step} of {total} examples: the time budget ran out")
    model.save_pretrained(OUT / "adapter")
    (OUT / "train_log.json").write_text(json.dumps(history))
    return {"examples_seen": step, "examples_planned": total, "final_batch_size": size, "train_minutes": round((time.time() - start) / 60, 1),
            "final_loss": history[-1]["loss"] if history else None, "examples": len(examples)}


def extract_json(text: str):
    """The first JSON object in `text`, tolerating code fences and surrounding prose; None if there is none."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            return decoder.raw_decode(text, match.start())[0]
        except json.JSONDecodeError:
            continue
    return None


@torch.no_grad()
def evaluate(tokenizer, model, device: str, name: str) -> dict:
    tasks = load_tasks(WORK / "evals" / name)[:CONFIG["eval_limit"]]
    prompts = []
    for task in tasks:
        shown = json.dumps(planner_input(task["request"], task["data"]), separators=(",", ":"), ensure_ascii=False)
        prompts.append(render(tokenizer, [{"role": "system", "content": VIC_SYSTEM},
                                          {"role": "user", "content": shown}], generation=True))
    tokenizer.padding_side = "left"
    order = sorted(range(len(tasks)), key=lambda i: len(prompts[i]))
    replies = [""] * len(tasks)
    model.eval()
    start = time.time()
    for at in range(0, len(order), CONFIG["eval_batch_size"]):
        chosen = order[at:at + CONFIG["eval_batch_size"]]
        enc = tokenizer([prompts[i] for i in chosen], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model.generate(**enc, max_new_tokens=CONFIG["max_new_tokens"], do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for i, row in zip(chosen, out):
            replies[i] = tokenizer.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    results = [score(task, extract_json(reply) or reply) for task, reply in zip(tasks, replies)]
    with (OUT / f"pred_{name}.jsonl").open("w", encoding="utf-8") as f:
        for task, reply in zip(tasks, replies):
            f.write(json.dumps({"id": task["id"], "text": reply}, ensure_ascii=False) + "\n")
    summary = {**summarize(results), "seconds_per_task": round((time.time() - start) / max(1, len(tasks)), 2)}
    (OUT / f"eval_{name}.json").write_text(json.dumps(summary, indent=1))
    log(f"eval {name}: {summary['correct']}/{summary['tasks']} = {100 * summary['accuracy']:.1f}%")
    # Sets of questions that name no output columns are read by the second figure.
    return {"tasks": summary["tasks"], "correct": summary["correct"], "accuracy": summary["accuracy"],
            "accuracy_ignoring_column_names": summary["accuracy_ignoring_column_names"]}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.device_count()} available")
    else:
        log("no CUDA GPU: running on CPU, which is only sensible for a smoke test")
    log(f"torch {torch.__version__}, transformers {transformers.__version__}, peft {peft.__version__}")
    torch.manual_seed(CONFIG["seed"])
    if device == "cuda":
        devices = [f"cuda:{i}" for i in range(min(CONFIG["gpus"], torch.cuda.device_count()))]
    else:
        devices = ["cpu"] * CONFIG["gpus"]      # two copies on a CPU only prove the two-device path
    loaded = [load_model(where) for where in devices]
    tokenizer, model = loaded[0]
    replicas = [replica for _, replica in loaded]
    summary = {"config": {k: v for k, v in CONFIG.items()}, "device": device, "devices": devices,
               "train": train(tokenizer, replicas, devices)}
    del loaded, replicas    # the evals need only the first copy
    summary["evals"] = {name: evaluate(tokenizer, model, device, name) for name in CONFIG["eval_sets"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    shutil.rmtree(OUT / "_bundle", ignore_errors=True)
    log("done: " + json.dumps(summary["evals"]))


if __name__ == "__main__":
    main()
