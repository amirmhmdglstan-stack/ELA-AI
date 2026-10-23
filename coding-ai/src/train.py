#!/usr/bin/env python3
"""Pretraining / fine-tuning loop for the from-scratch coding language model.

Features: AdamW, warmup + cosine decay, gradient accumulation and clipping,
resumable checkpoints, validation loss/perplexity, throughput logging, NaN
detection and a wall-clock budget option.

    python src/train.py --config configs/pretrain.yaml
    python src/train.py --config configs/pretrain.yaml --resume checkpoints/pretrain/latest.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, ModelConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class TokenDataset:
    """Memory-mapped uint16 token stream; samples random contiguous windows."""

    def __init__(self, paths: list[str], seq_len: int, seed: int = 0,
                 weights: list[float] | None = None):
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.sizes = [len(a) for a in self.arrays]
        self.total = sum(self.sizes)
        if self.total < seq_len + 1:
            raise ValueError(f"dataset too small: {self.total} tokens")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        if weights is None:
            weights = [s / self.total for s in self.sizes]
        self.p_choose = np.array(weights) / sum(weights)
        self.epochs_done = 0

    def get_batch(self, batch_size: int, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for b in range(batch_size):
            i = int(self.rng.choice(len(self.arrays), p=self.p_choose))
            arr = self.arrays[i]
            start = int(self.rng.integers(0, self.sizes[i] - self.seq_len - 1))
            chunk = np.asarray(arr[start:start + self.seq_len + 1])
            x[b] = chunk[:-1]
            y[b] = chunk[1:]
        return torch.from_numpy(x), torch.from_numpy(y)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    run_name: str = "pretrain"
    model: ModelConfig = field(default_factory=ModelConfig)
    data_files: list[str] = field(default_factory=list)
    train_seq_len: int = 0          # 0 = use model.max_seq_len
    val_files: list[str] = field(default_factory=list)
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 2000
    eval_every: int = 100
    eval_batches: int = 20
    save_every: int = 250
    keep_last: int = 3
    log_every: int = 10
    seed: int = 1337
    num_threads: int = 2
    compile: bool = False
    max_minutes: float | None = None
    resume: str | None = None
    out_dir: str = "checkpoints/pretrain"
    log_dir: str = "logs"


def load_config(path: str) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text())
    model_keys = {k: v for k, v in raw.get("model", {}).items()}
    cfg = TrainConfig(**{k: v for k, v in raw.items() if k not in ("model", "data")})
    cfg.model = ModelConfig(**model_keys)
    data = raw.get("data", {})
    cfg.data_files = data.get("train_files", cfg.data_files)
    cfg.val_files = data.get("val_files", cfg.val_files)
    # resolve relative paths against the repository root
    cfg.data_files = [str((ROOT / f) if not os.path.isabs(f) else f) for f in cfg.data_files]
    cfg.val_files = [str((ROOT / f) if not os.path.isabs(f) else f) for f in cfg.val_files]
    return cfg


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


# --------------------------------------------------------------------------- #
# Evaluation / checkpointing
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model: GPT, dataset: TokenDataset, cfg: TrainConfig, device) -> tuple[float, float]:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = dataset.get_batch(cfg.batch_size)
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        if torch.isfinite(loss):
            losses.append(loss.item())
    model.train()
    if not losses:
        return float("nan"), float("nan")
    mean = float(np.mean(losses))
    return mean, math.exp(min(20.0, mean))


def save_checkpoint(path: Path, model, optimizer, cfg: TrainConfig, step: int,
                    best_val: float, tokens_seen: int, tokens_history: list[int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "best_val": best_val,
        "tokens_seen": tokens_seen,
        "tokens_history": tokens_history[-64:],
        "model_config": cfg.model.to_dict(),
        "train_config": {k: v for k, v in cfg.__dict__.items() if k != "model"},
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "format_version": 1,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

class TimeUp(Exception):
    pass


def train(cfg: TrainConfig):
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = ROOT / cfg.out_dir
    log_dir = ROOT / cfg.log_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.data_files:
        raise SystemExit("no training files configured (data.train_files)")

    seq_len = cfg.train_seq_len or cfg.model.max_seq_len
    if seq_len > cfg.model.max_seq_len:
        raise SystemExit(f"train_seq_len {seq_len} exceeds model max_seq_len {cfg.model.max_seq_len}")
    train_ds = TokenDataset(cfg.data_files, seq_len, seed=cfg.seed)
    val_ds = None
    if cfg.val_files:
        val_ds = TokenDataset(cfg.val_files, seq_len, seed=cfg.seed + 1)

    model = GPT(cfg.model).to(device)
    n_params = model.num_params()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay, eps=1e-8,
    )

    step, best_val, tokens_seen, tokens_history = 0, float("inf"), 0, []
    if cfg.resume:
        ckpt = load_checkpoint(Path(cfg.resume), model, optimizer, device)
        step = ckpt["step"]
        best_val = ckpt.get("best_val", float("inf"))
        tokens_seen = ckpt.get("tokens_seen", 0)
        tokens_history = ckpt.get("tokens_history", [])
        if "numpy_rng" in ckpt:
            np.random.set_state(ckpt["numpy_rng"])
        print(f"[resume] step={step} tokens={tokens_seen/1e6:.2f}M best_val={best_val:.4f}")

    tokens_per_step = cfg.batch_size * seq_len * cfg.grad_accum
    print(f"[init] model={cfg.run_name} params={n_params/1e6:.2f}M "
          f"layers={cfg.model.n_layers} d_model={cfg.model.d_model} "
          f"vocab={cfg.model.vocab_size} seq={seq_len} "
          f"tokens/step={tokens_per_step} device={device}")
    print(f"[init] train_tokens={train_ds.total/1e6:.2f}M "
          f"val_tokens={(val_ds.total/1e6 if val_ds else 0):.2f}M "
          f"planned_steps={cfg.max_steps}")

    log_path = log_dir / f"{cfg.run_name}.jsonl"
    log_file = log_path.open("a", encoding="utf-8")
    t_start = time.time()
    t_last = t_start
    tokens_at_last = tokens_seen
    model.train()

    stop_requested = {"flag": False}

    def handler(signum, frame):
        stop_requested["flag"] = True
        print("[signal] graceful stop requested; finishing current step")

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    while step < cfg.max_steps:
        if stop_requested["flag"]:
            break
        if cfg.max_minutes and (time.time() - t_start) / 60 > cfg.max_minutes:
            print(f"[time] budget of {cfg.max_minutes} min reached at step {step}")
            break

        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        skipped = False
        for _ in range(cfg.grad_accum):
            x, y = train_ds.get_batch(cfg.batch_size)
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, targets=y)
            if not torch.isfinite(loss):
                skipped = True
                optimizer.zero_grad(set_to_none=True)
                break
            (loss / cfg.grad_accum).backward()
            loss_sum += loss.item()
        if skipped:
            print(f"[warn] non-finite loss at step {step}; skipping step")
            step += 1
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not torch.isfinite(grad_norm):
            print(f"[warn] non-finite grad norm at step {step}; skipping step")
            optimizer.zero_grad(set_to_none=True)
            step += 1
            continue
        optimizer.step()

        step += 1
        tokens_seen += tokens_per_step
        tokens_history.append(tokens_per_step)
        train_loss = loss_sum / cfg.grad_accum

        if step % cfg.log_every == 0 or step == 1:
            now = time.time()
            dt = now - t_last
            tps = (tokens_seen - tokens_at_last) / dt if dt > 0 else 0.0
            t_last, tokens_at_last = now, tokens_seen
            elapsed = now - t_start
            rec = {"step": step, "tokens": tokens_seen, "loss": round(train_loss, 4),
                   "lr": round(lr, 6), "grad_norm": round(float(grad_norm), 3),
                   "tok_per_s": round(tps, 1), "elapsed_s": round(elapsed, 1),
                   "eta_min": round((cfg.max_steps - step) * tokens_per_step / max(tps, 1e-9) / 60, 1)}
            print(f"step {step:5d}/{cfg.max_steps} loss {train_loss:.4f} "
                  f"lr {lr:.2e} gnorm {float(grad_norm):.2f} "
                  f"{tps:6.0f} tok/s elapsed {elapsed/60:5.1f}m "
                  f"eta {rec['eta_min']:6.1f}m", flush=True)
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()

        if val_ds is not None and (step % cfg.eval_every == 0 or step == cfg.max_steps):
            t_eval = time.time()
            val_loss, ppl = evaluate(model, val_ds, cfg, device)
            print(f"  [val] step {step} loss {val_loss:.4f} ppl {ppl:.2f} "
                  f"({time.time()-t_eval:.1f}s)", flush=True)
            log_file.write(json.dumps({"step": step, "val_loss": round(val_loss, 4),
                                       "val_ppl": round(ppl, 3),
                                       "tokens": tokens_seen}) + "\n")
            log_file.flush()
            if val_loss < best_val and math.isfinite(val_loss):
                best_val = val_loss
                save_checkpoint(out_dir / "best.pt", model, optimizer, cfg, step,
                                best_val, tokens_seen, tokens_history)
                print(f"  [ckpt] new best val {best_val:.4f} -> best.pt")

        if step % cfg.save_every == 0:
            save_checkpoint(out_dir / "latest.pt", model, optimizer, cfg, step,
                            best_val, tokens_seen, tokens_history)
            save_checkpoint(out_dir / f"step_{step:06d}.pt", model, optimizer, cfg, step,
                            best_val, tokens_seen, tokens_history)
            # prune old step checkpoints
            steps = sorted(out_dir.glob("step_*.pt"))
            for old in steps[:-cfg.keep_last]:
                old.unlink(missing_ok=True)

    save_checkpoint(out_dir / "latest.pt", model, optimizer, cfg, step,
                    best_val, tokens_seen, tokens_history)
    elapsed = time.time() - t_start
    summary = {"run": cfg.run_name, "steps": step, "tokens": tokens_seen,
               "elapsed_min": round(elapsed / 60, 1),
               "tokens_per_s": round(tokens_seen / max(elapsed, 1e-9), 1),
               "params": n_params, "best_val": best_val}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[done] {json.dumps(summary)}")
    log_file.close()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seq-len", type=int, default=None,
                    help="override the training sequence length (context-extension stage)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.resume is not None:
        cfg.resume = args.resume
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.max_minutes is not None:
        cfg.max_minutes = args.max_minutes
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.lr is not None:
        cfg.lr = args.lr
    if args.seq_len is not None:
        if args.seq_len > cfg.model.max_seq_len:
            raise SystemExit(f"--seq-len {args.seq_len} exceeds model max_seq_len "
                             f"{cfg.model.max_seq_len}; edit the config's max_seq_len first")
        cfg.train_seq_len = args.seq_len
    train(cfg)


if __name__ == "__main__":
    main()
