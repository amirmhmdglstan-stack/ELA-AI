#!/usr/bin/env python3
"""Export a compact inference checkpoint from a training checkpoint.

Training checkpoints contain the AdamW state (2x the weights) so that runs can be
resumed.  Release artefacts do not need it:

    python scripts/export_checkpoint.py --src checkpoints/pretrain/best.pt \
        --dst release/coding-ai-17m-base.pt [--fp16]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--fp16", action="store_true", help="store fp16 weights (GPU inference)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    state = dict(ckpt["model"])
    if args.fp16:
        state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
    # tied embeddings appear under two keys; re-link them so torch.save stores the
    # storage once instead of twice (halving the file size)
    if "lm_head.weight" in state and "tok_emb.weight" in state:
        state["lm_head.weight"] = state["tok_emb.weight"]
    out = {
        "model": state,
        "model_config": ckpt["model_config"],
        "step": ckpt.get("step"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "best_val": ckpt.get("best_val"),
        "dtype": "float16" if args.fp16 else "float32",
        "note": args.note,
        "format_version": 1,
    }
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    n = sum(v.numel() for v in state.values())
    print(f"{dst}: {dst.stat().st_size/1e6:.1f} MB, {n/1e6:.2f}M tensors-elements, "
          f"step={out['step']} val={out['best_val']}")


if __name__ == "__main__":
    main()
