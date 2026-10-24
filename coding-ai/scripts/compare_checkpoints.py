#!/usr/bin/env python3
"""Evaluate several checkpoints of the same run on the identical validation windows.

Random validation windows make 100-step-apart checkpoints hard to compare (the
sampled slices differ), so this script fixes the window list and reports a
per-checkpoint loss / perplexity table plus a paired comparison against the
final checkpoint.

    python scripts/compare_checkpoints.py --checkpoints checkpoints/pretrain/*.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import GPT, ModelConfig  # noqa: E402
from train import TokenDataset  # noqa: E402


@torch.no_grad()
def eval_ckpt(path: Path, dataset: TokenDataset, windows, batch_size: int = 4) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(ModelConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    losses = []
    for b in range(len(windows) // batch_size):
        x, y = dataset.batch_from_windows(windows, batch_size, b)
        _, loss, _ = model(x, targets=y)
        losses.extend([loss.item()])
    del model
    mean = float(np.mean(losses))
    return {"path": str(path), "step": ckpt.get("step"), "val_loss": mean,
            "val_ppl": float(np.exp(min(20, mean))), "best_val_at_save": ckpt.get("best_val")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--windows", type=int, default=120)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    files = sorted((str(ROOT / "data/tokens/val.bin"),))
    ds = TokenDataset(files, args.seq_len, seed=0)
    windows = ds.make_windows(args.windows, seed=12345)
    print(f"{len(windows)} fixed validation windows x {args.seq_len} tokens "
          f"({len(windows)*args.seq_len/1e6:.2f}M tokens)\n")
    print("| checkpoint | step | fixed val loss | ppl |")
    print("|---|---|---|---|")
    results = []
    for p in args.checkpoints:
        r = eval_ckpt(Path(p), ds, windows, args.batch_size)
        results.append(r)
        print(f"| {Path(p).name} | {r['step']} | {r['val_loss']:.4f} | {r['val_ppl']:.2f} |")
    best = min(results, key=lambda r: r["val_loss"])
    print(f"\nbest on the fixed suite: {Path(best['path']).name} "
          f"(step {best['step']}, loss {best['val_loss']:.4f})")


if __name__ == "__main__":
    main()
