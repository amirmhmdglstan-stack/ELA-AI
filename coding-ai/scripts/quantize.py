#!/usr/bin/env python3
"""Create an int8-dynamic-quantised copy of a checkpoint (CPU inference).

Dynamic quantisation converts the weights of nn.Linear layers to int8 and
quantises activations on the fly.  It is a torch.quantization feature that has
existed for years, needs no CUDA and works on any CPU, so it is safe for the
target machine.  The original fp32 checkpoint is never modified.

    python scripts/quantize.py --src checkpoints/pretrain/best.pt \
        --dst checkpoints/pretrain/best_int8.pt
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import GPT, ModelConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "checkpoints/pretrain/best.pt"))
    ap.add_argument("--dst", default=str(ROOT / "checkpoints/pretrain/best_int8.pt"))
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Reference loss on random tokens before/after quantisation.
    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (2, args.seq_len))
    with torch.no_grad():
        _, loss_fp32, _ = model(x, targets=x)

    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    with torch.no_grad():
        _, loss_int8, _ = quantized(x, targets=x)

    torch.save({"model": quantized.state_dict(), "model_config": ckpt["model_config"],
                "step": ckpt.get("step"), "best_val": ckpt.get("best_val"),
                "tokens_seen": ckpt.get("tokens_seen"), "quantized": "int8-dynamic"},
               args.dst)

    # timing: greedy decode of 32 tokens
    def time_decode(m):
        ids = torch.randint(0, cfg.vocab_size, (1, 64))
        t0 = time.time()
        m.generate(ids, max_new_tokens=32, temperature=0.0)
        return time.time() - t0

    t32, t8 = time_decode(model), time_decode(quantized)
    size_mb = Path(args.dst).stat().st_size / 1e6
    print(f"fp32 loss {loss_fp32.item():.4f}  int8 loss {loss_int8.item():.4f}  "
          f"(delta {loss_int8.item() - loss_fp32.item():+.4f})")
    print(f"decode 32 tokens: fp32 {t32:.2f}s  int8 {t8:.2f}s  speedup {t32/t8:.2f}x")
    print(f"wrote {args.dst} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
