#!/usr/bin/env python3
"""Supervised instruction tuning on top of the pretrained base checkpoint.

The base checkpoint is copied, never modified.  Loss is computed only on the
assistant response tokens (prompt tokens are masked to -100), which is what
makes a small model follow the chat template instead of continuing the prompt.

    python src/finetune.py --config configs/sft.yaml \
        --base checkpoints/pretrain/best.pt --out checkpoints/sft
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chatfmt  # noqa: E402
from model import GPT, ModelConfig  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402
from train import load_checkpoint, save_checkpoint, TrainConfig, lr_at  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class SFTDataset:
    """Instruction examples with prompt-masked labels."""

    def __init__(self, path: str, tok: BPETokenizer, max_seq_len: int, seed: int = 0):
        self.tok = tok
        self.max_seq_len = max_seq_len
        self.examples: list[tuple[list[int], list[int]]] = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                try:
                    prompt, response = chatfmt.render_example(rec["messages"],
                                                              system=rec.get("system", chatfmt.DEFAULT_SYSTEM))
                except (KeyError, ValueError):
                    skipped += 1
                    continue
                p_ids = tok.encode(prompt, allow_special=True)
                r_ids = tok.encode(response, allow_special=True)
                if len(p_ids) + len(r_ids) <= 8:
                    skipped += 1
                    continue
                if len(p_ids) + len(r_ids) > max_seq_len:
                    # keep the response, truncate the prompt
                    keep = max(8, max_seq_len - len(r_ids))
                    p_ids = p_ids[-keep:]
                    r_ids = r_ids[: max_seq_len - len(p_ids)]
                self.examples.append((p_ids, r_ids))
        self.rng = random.Random(seed)
        self.skipped = skipped
        if not self.examples:
            raise SystemExit(f"no usable examples in {path}")

    def __len__(self):
        return len(self.examples)

    def batch(self, batch_size: int):
        samples = [self.examples[self.rng.randrange(len(self.examples))]
                   for _ in range(batch_size)]
        maxlen = max(len(p) + len(r) for p, r in samples)
        x = torch.zeros((batch_size, maxlen), dtype=torch.long)
        y = torch.full((batch_size, maxlen), -100, dtype=torch.long)
        mask = torch.zeros((batch_size, maxlen), dtype=torch.long)  # 1 = real token
        for i, (p, r) in enumerate(samples):
            seq = p + r
            x[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            y[i, len(p) - 1:len(seq) - 1] = torch.tensor(seq[len(p) - 1:-1], dtype=torch.long)
            mask[i, :len(seq)] = 1
        # do not predict padding
        y[mask == 0] = -100
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", default=str(ROOT / "checkpoints/pretrain/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "data/sft/train.jsonl"))
    ap.add_argument("--val-data", default=str(ROOT / "data/sft/val.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints/sft"))
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    tok_path = ROOT / raw.get("tokenizer", "data/tokenizer.json")
    tok = BPETokenizer.load(tok_path)

    base = Path(args.base)
    if not base.exists():
        raise SystemExit(f"base checkpoint missing: {base}\n"
                         f"run pretraining first (src/train.py --config configs/pretrain.yaml)")
    ckpt = torch.load(base, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    print(f"[sft] base={base} step={ckpt.get('step')} params={model.num_params()/1e6:.2f}M "
          f"val={ckpt.get('best_val')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(int(raw.get("num_threads", 2)))
    model.to(device)

    train_ds = SFTDataset(args.data, tok, cfg.max_seq_len, seed=int(raw.get("seed", 7)))
    val_ds = None
    if Path(args.val_data).exists():
        val_ds = SFTDataset(args.val_data, tok, cfg.max_seq_len, seed=99)
    print(f"[sft] train examples={len(train_ds)} (skipped {train_ds.skipped}) "
          f"val examples={len(val_ds) if val_ds else 0}")

    tcfg = TrainConfig(run_name=raw.get("run_name", "sft"), model=cfg,
                       lr=float(raw.get("lr", 1e-4)), min_lr=float(raw.get("min_lr", 1e-5)),
                       max_steps=int(raw.get("max_steps", 400)),
                       warmup_steps=int(raw.get("warmup_steps", 20)),
                       weight_decay=float(raw.get("weight_decay", 0.05)),
                       batch_size=int(raw.get("batch_size", 8)),
                       grad_accum=int(raw.get("grad_accum", 2)),
                       grad_clip=float(raw.get("grad_clip", 1.0)),
                       eval_every=int(raw.get("eval_every", 50)),
                       save_every=int(raw.get("save_every", 100)),
                       log_every=int(raw.get("log_every", 10)),
                       out_dir=args.out, log_dir="logs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                                  betas=(0.9, 0.95), weight_decay=tcfg.weight_decay)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / "sft.jsonl"
    log_file = log_path.open("a", encoding="utf-8")

    step, best_val = 0, float("inf")
    if args.resume:
        c = load_checkpoint(Path(args.resume), model, optimizer, device)
        step, best_val = c["step"], c.get("best_val", float("inf"))

    model.train()
    t0 = time.time()
    while step < tcfg.max_steps:
        lr = lr_at(step, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(tcfg.grad_accum):
            x, y = train_ds.batch(tcfg.batch_size)
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, targets=y)
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at step {step}")
                optimizer.zero_grad(set_to_none=True)
                break
            (loss / tcfg.grad_accum).backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()
        step += 1
        if step % tcfg.log_every == 0:
            tps = (step * tcfg.batch_size * tcfg.grad_accum) / (time.time() - t0)
            print(f"sft step {step}/{tcfg.max_steps} loss {total/tcfg.grad_accum:.4f} lr {lr:.2e} "
                  f"{tps:.2f} ex/s", flush=True)
            log_file.write(json.dumps({"step": step, "loss": round(total / tcfg.grad_accum, 4),
                                       "lr": lr}) + "\n")
            log_file.flush()
        if val_ds and (step % tcfg.eval_every == 0 or step == tcfg.max_steps):
            model.eval()
            losses = []
            with torch.no_grad():
                for _ in range(min(20, len(val_ds))):
                    x, y = val_ds.batch(tcfg.batch_size)
                    _, loss, _ = model(x, targets=y.to(device))
                    losses.append(loss.item())
            model.train()
            vl = float(np.mean(losses))
            print(f"  [sft val] step {step} loss {vl:.4f}", flush=True)
            log_file.write(json.dumps({"step": step, "val_loss": round(vl, 4)}) + "\n")
            log_file.flush()
            if vl < best_val:
                best_val = vl
                save_checkpoint(out_dir / "best.pt", model, optimizer, tcfg, step, best_val,
                                step * tcfg.batch_size * tcfg.grad_accum * 32, [])
        if step % tcfg.save_every == 0:
            save_checkpoint(out_dir / "latest.pt", model, optimizer, tcfg, step, best_val,
                            step * tcfg.batch_size * tcfg.grad_accum * 32, [])
    save_checkpoint(out_dir / "latest.pt", model, optimizer, tcfg, step, best_val,
                    step * tcfg.batch_size * tcfg.grad_accum * 32, [])
    print(f"[sft done] steps={step} best_val={best_val:.4f} elapsed={(time.time()-t0)/60:.1f}m")
    log_file.close()


if __name__ == "__main__":
    main()
