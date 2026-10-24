#!/usr/bin/env python3
"""Turn a training JSONL log into a compact markdown table (for the report).

    python scripts/summarize_run.py logs/pretrain.jsonl --every 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="logs/pretrain.jsonl")
    ap.add_argument("--every", type=int, default=100, help="sampling interval in steps")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.log).read_text().splitlines() if l.strip()]
    train = {r["step"]: r for r in rows if "loss" in r and "tokens" in r}
    val = {r["step"]: r for r in rows if "val_loss" in r}

    print("| step | tokens (M) | train loss | val loss | val ppl | lr | tok/s | elapsed (min) |")
    print("|---|---|---|---|---|---|---|---|")
    for step in sorted(train):
        if step % args.every and step not in val:
            continue
        r = train[step]
        v = val.get(step)
        print(f"| {step} | {r['tokens']/1e6:.2f} | {r['loss']:.4f} | "
              f"{v['val_loss']:.4f} | {v['val_ppl']:.1f} | {r['lr']:.2e} | "
              f"{r['tok_per_s']:.0f} | {r['elapsed_s']/60:.1f} |" if v else
              f"| {step} | {r['tokens']/1e6:.2f} | {r['loss']:.4f} | – | – | "
              f"{r['lr']:.2e} | {r['tok_per_s']:.0f} | {r['elapsed_s']/60:.1f} |")
    if val:
        print(f"\nvalidation points: {len(val)}; "
              f"best val loss {min(v['val_loss'] for v in val.values()):.4f} "
              f"at step {min(val.values(), key=lambda v: v['val_loss'])['step']}")


if __name__ == "__main__":
    main()
