"""Training run of RAGnarok Judge (TRAINING.md): LoRA on one base, three-logit loss at the last
position, batch of 1 with gradient accumulation, four checkpoints scored on the dev set.

    python training/train.py --base pleias --lr 1e-4 --train TRAIN.jsonl --dev DEV.jsonl --out RUN_DIR
    python training/train.py --base pleias --lr 1e-4 --train TRAIN.jsonl --out PILOT_DIR --pilot

The same command resumes an interrupted run from the latest state found in RUN_DIR. With
``--pilot`` the run stops after 300 updates, writes resume states only and never reads the dev
set. RUN_DIR holds adapters, optimiser states, logs and per-sentence dev scores: it stays
outside the repository. The test set is never read.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

import judge
import sampler

PILOT_UPDATES = 300
SWAP_STOP = 75  # exit code when the run stops itself because the machine swaps


@dataclass(frozen=True)
class Config:
    base: str
    learning_rate: float
    epochs: int = 2
    accumulation: int = 8
    warmup_share: float = 0.05
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    seed: int = sampler.SEED
    supported_per_epoch: int = sampler.SUPPORTED_PER_EPOCH
    resume_every: int = 200
    grad_checkpoint: bool = True

    def updates_per_epoch(self, epoch_size: int) -> int:
        if epoch_size % self.accumulation:
            raise ValueError(f"an epoch of {epoch_size} examples is not a whole number of updates")
        return epoch_size // self.accumulation

    def total_updates(self, epoch_size: int) -> int:
        return self.epochs * self.updates_per_epoch(epoch_size)

    def checkpoints(self, epoch_size: int) -> list[int]:
        """Updates after which a checkpoint is saved: every half epoch, rounded up."""
        per_epoch = self.updates_per_epoch(epoch_size)
        return [math.ceil(k * per_epoch / 2) for k in range(1, 2 * self.epochs + 1)]

    def warmup(self, epoch_size: int) -> int:
        return round(self.warmup_share * self.total_updates(epoch_size))


def learning_rate(update: int, peak: float, total: int, warmup: int) -> float:
    """Rate of an update numbered from 1: linear warm-up to the peak, then cosine decay to 0."""
    if update <= warmup:
        return peak * update / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * (update - warmup) / (total - warmup)))


def swap_used_mb() -> float | None:
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False).stdout
        return float(re.search(r"used = ([\d.]+)M", out).group(1))  # type: ignore[union-attr]
    except (OSError, AttributeError, ValueError):
        return None


# ----------------------------------------------------------------------------- states


def save_state(folder: Path, model: nn.Module, optimizer: optim.Optimizer, meta: dict[str, Any]) -> None:
    """Adapter weights, optimiser state and progress, written apart then moved into place."""
    tmp = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(str(tmp / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    (tmp / "state.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(folder, ignore_errors=True)
    tmp.rename(folder)


def load_state(folder: Path, model: nn.Module, optimizer: optim.Optimizer) -> dict[str, Any]:
    model.load_weights(list(mx.load(str(folder / "adapters.safetensors")).items()), strict=False)
    optimizer.state = tree_unflatten(list(mx.load(str(folder / "optimizer.safetensors")).items()))
    mx.eval(model.parameters(), optimizer.state)
    return json.loads((folder / "state.json").read_text(encoding="utf-8"))


def latest_state(out: Path) -> Path | None:
    """The state with the most updates among the checkpoints and the resume state."""
    best: tuple[int, Path] | None = None
    for folder in [*sorted(out.glob("checkpoint-*")), out / "resume"]:
        file = folder / "state.json"
        if folder.name.endswith(".tmp") or not file.exists():
            continue
        done = json.loads(file.read_text(encoding="utf-8"))["updates_done"]
        if best is None or done > best[0]:
            best = (done, folder)
    return None if best is None else best[1]


def checkpoint_dir(out: Path, update: int) -> Path:
    return out / f"checkpoint-{update:04d}"


# ----------------------------------------------------------------------------- run


def run(
    model: nn.Module,
    labels: dict[str, str],
    example: Callable[[str], tuple[list[int], int]],
    label_tokens: list[int],
    config: Config,
    out: Path,
    *,
    score_dev: Callable[[Path], None] | None = None,
    stop_after: int | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    swap_margin_mb: float | None = 500.0,
) -> dict[str, Any]:
    """Trains from the latest state of ``out`` to the end of the run, or to ``stop_after`` updates.

    ``labels``: id -> label of the training file. ``example(id)``: token ids of the prompt and
    index of the label. ``score_dev(folder)`` scores the checkpoint just saved in ``folder``; it
    is called once per checkpoint, and again on resumption only if its output is missing.
    """
    size = sampler.epoch_size(labels, config.supported_per_epoch)
    total, warmup = config.total_updates(size), config.warmup(size)
    checkpoints = config.checkpoints(size) if score_dev is not None or stop_after is None else []
    out.mkdir(parents=True, exist_ok=True)

    optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.init(model.trainable_parameters())
    meta = {"config": asdict(config), "updates_done": 0, "resumed_from": []}
    state = latest_state(out)
    if state is not None:
        meta = load_state(state, model, optimizer)
        if meta["config"] != asdict(config):
            raise SystemExit(f"{state}: saved with another configuration; use another --out")
        meta["resumed_from"] = [*meta["resumed_from"], meta["updates_done"]]
    done = meta["updates_done"]
    if score_dev is not None:  # a checkpoint whose dev pass was interrupted
        for update in checkpoints:
            folder = checkpoint_dir(out, update)
            if update <= done and (folder / "state.json").exists() and not (folder / "dev.json").exists():
                score_dev(folder)

    if config.grad_checkpoint:
        from mlx_lm.tuner.trainer import grad_checkpoint

        grad_checkpoint(model.layers[0])
    value_and_grad = nn.value_and_grad(model, judge.loss)
    label_array = mx.array(label_tokens)
    model.train()
    swap_start = swap_used_mb() if swap_margin_mb is not None else None
    last = total if stop_after is None else min(stop_after, total)
    stream = sampler.examples_from(labels, done * config.accumulation, config.epochs, config.seed,
                                   config.supported_per_epoch)
    log = (out / "train_log.jsonl").open("a", encoding="utf-8")
    status = "completed" if last == total else f"stopped after {last} updates"

    for update in range(done + 1, last + 1):
        tic = time.perf_counter()
        grads, losses, tokens, first = None, [], 0, None
        for _ in range(config.accumulation):
            position, example_id = next(stream)
            first = position if first is None else first
            ids, target = example(example_id)
            value, grad = value_and_grad(model, mx.array([ids]), mx.array([target]), label_array)
            grads = grad if grads is None else tree_map(lambda a, b: a + b, grads, grad)
            mx.eval(value, grads)
            losses.append(float(value.item()))
            tokens += len(ids)
            mx.clear_cache()
        grads = tree_map(lambda g: g / config.accumulation, grads)
        grads, norm = optim.clip_grad_norm(grads, config.clip_norm)
        rate = learning_rate(update, config.learning_rate, total, warmup)
        optimizer.learning_rate = rate
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        del grads
        meta["updates_done"] = update
        row = {"update": update, "first_position": first, "loss": sum(losses) / len(losses), "grad_norm": float(norm),
               "learning_rate": rate, "tokens": tokens, "seconds": time.perf_counter() - tic,
               "mlx_peak_gb": mx.get_peak_memory() / 1e9}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if on_update is not None:
            on_update(row)

        if update in checkpoints:
            folder = checkpoint_dir(out, update)
            save_state(folder, model, optimizer, meta)
            if score_dev is not None:
                score_dev(folder)
                model.train()
        elif update % config.resume_every == 0 or update == last:
            save_state(out / "resume", model, optimizer, meta)
        if swap_start is not None and update < last:
            swap = swap_used_mb()
            if swap is not None and swap > swap_start + swap_margin_mb:
                save_state(out / "resume", model, optimizer, meta)
                status = f"stopped at update {update}: swap grew from {swap_start:.0f} to {swap:.0f} MB"
                break
    log.close()
    return {"status": status, **meta}


# ----------------------------------------------------------------------------- command line


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", choices=("pleias", "qwen3"), required=True)
    parser.add_argument("--lr", type=float, required=True, help="peak learning rate")
    parser.add_argument("--train", type=Path, required=True, help="train-rule5-rule1.jsonl")
    parser.add_argument("--dev", type=Path, help="dev.jsonl; required unless --pilot")
    parser.add_argument("--out", type=Path, required=True, help="run folder, outside the repository")
    parser.add_argument("--pilot", action="store_true", help=f"{PILOT_UPDATES} updates, no dev evaluation")
    args = parser.parse_args()
    if not args.pilot and args.dev is None:
        parser.error("--dev is required for a run")
    if args.pilot and args.dev is not None:
        parser.error("a pilot does not read the dev set")

    config = Config(base=args.base, learning_rate=args.lr)
    records = {record["id"]: record for record in read_jsonl(args.train)}
    labels = {i: record["label"] for i, record in records.items()}
    model, tokenizer = judge.load_base(args.base)
    trainable = judge.add_lora(model, config.seed)
    label_tokens = judge.label_ids(args.base, tokenizer)
    print(f"{args.base}: {trainable} trainable parameters, label tokens {label_tokens}", flush=True)

    def example(example_id: str) -> tuple[list[int], int]:
        record = records[example_id]
        return judge.prompt_ids(args.base, tokenizer, record), judge.TARGET[record["label"]]

    score_dev = None
    if not args.pilot:
        import dev_eval

        dev = dev_eval.ra.read_split(args.dev)
        dev_records = [dev[i] for i in sorted(dev)]

        def score_dev(folder: Path) -> None:
            tic = time.perf_counter()
            scores = judge.score_records(model, args.base, tokenizer, dev_records)
            (folder / "dev_scores.json").write_text(json.dumps(scores) + "\n", encoding="utf-8")
            result = {**dev_eval.point(dev, scores), "sentences": len(scores), "seconds": time.perf_counter() - tic}
            (folder / "dev.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
            print(f"{folder.name}: dev criterion {result['criterion']:.4f} (UNSUPPORTED {result['pauc_U']:.4f}, "
                  f"CONTRADICTED {result['pauc_C']:.4f})", flush=True)

    def report(row: dict[str, Any]) -> None:
        if row["update"] % 10 == 0:
            print(f"update {row['update']:4d}  loss {row['loss']:.4f}  grad norm {row['grad_norm']:.2f}  "
                  f"lr {row['learning_rate']:.2e}  {row['seconds']:.1f} s", flush=True)

    result = run(model, labels, example, label_tokens, config, args.out, score_dev=score_dev,
                 stop_after=PILOT_UPDATES if args.pilot else None, on_update=report)
    print(result["status"], "| resumed from updates:", result["resumed_from"], flush=True)
    return SWAP_STOP if "swap" in result["status"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
