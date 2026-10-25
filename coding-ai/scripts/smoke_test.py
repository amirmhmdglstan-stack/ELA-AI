#!/usr/bin/env python3
"""End-to-end smoke test: tokenizer -> data pack -> training -> checkpoint -> generation.

Uses a tiny synthetic corpus built from this repository's own source files, so it
runs in about a minute on any machine and needs no downloads:

    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model import GPT, ModelConfig  # noqa: E402
from tokenizer import train_bpe  # noqa: E402


def build_corpus() -> str:
    texts = []
    for p in sorted((ROOT / "src").glob("*.py")):
        texts.append(p.read_text(encoding="utf-8"))
    for p in sorted((ROOT / "tests").glob("*.py")):
        texts.append(p.read_text(encoding="utf-8"))
    return "\n".join(texts)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
    try:
        print("[1/6] building corpus")
        corpus = build_corpus()
        print(f"      {len(corpus)/1e6:.2f}M characters")

        print("[2/6] training tokenizer (vocab 1024)")
        tok = train_bpe([corpus], vocab_size=1024, min_freq=1, verbose=False)
        tok.save(tmp / "tokenizer.json")
        sample = "def add(a, b):\n    return a + b\n"
        assert tok.decode(tok.encode(sample)) == sample

        print("[3/6] packing tokens")
        ids = np.array(tok.encode(corpus, allow_special=False), dtype=np.uint16)
        # repeat so the toy dataset is large enough for a few batches
        ids = np.tile(ids, 4)
        split = int(len(ids) * 0.9)
        ids[:split].tofile(tmp / "train.bin")
        ids[split:].tofile(tmp / "val.bin")
        print(f"      {len(ids)} tokens")

        print("[4/6] training a tiny model for 30 steps")
        cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4,
                          n_kv_heads=4, d_ff=128, max_seq_len=64)
        torch.manual_seed(0)
        model = GPT(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        train = np.memmap(tmp / "train.bin", dtype=np.uint16, mode="r")
        losses = []
        for step in range(30):
            starts = np.random.randint(0, len(train) - 65, size=4)
            x = np.stack([np.asarray(train[s:s + 64]) for s in starts]).astype(np.int64)
            y = np.stack([np.asarray(train[s + 1:s + 65]) for s in starts]).astype(np.int64)
            _, loss, _ = model(torch.from_numpy(x), targets=torch.from_numpy(y))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        print(f"      loss {losses[0]:.3f} -> {losses[-1]:.3f}")
        assert losses[-1] < losses[0], "training did not reduce the loss"

        print("[5/6] saving and reloading checkpoint")
        torch.save({"model": model.state_dict(), "model_config": cfg.to_dict(), "step": 30,
                    "best_val": losses[-1], "tokens_seen": 30 * 4 * 64}, tmp / "tiny.pt")
        from generate import load_model
        model2, tok2, meta = load_model(tmp / "tiny.pt", "cpu", tmp / "tokenizer.json")
        assert meta["step"] == 30

        print("[6/6] generating")
        out = model2.generate(torch.tensor([tok2.encode("def ")],
                                           dtype=torch.long), max_new_tokens=24, temperature=0.0)
        text = tok2.decode(out[0].tolist())
        print(f"      generated {len(text)} chars: {text[:60]!r}")
        assert len(out[0]) > 1

        print("\nSMOKE TEST PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
