#!/usr/bin/env python3
"""Build a fill-in-the-middle (FIM) token stream for a short adaptation stage.

Documents are re-cut as   <|fim_prefix|>pre<|fim_suffix|>suf<|fim_middle|>mid<|endoftext|>
(PSM ordering, the format used by StarCoder/CodeLlama) with the split points drawn so
that the middle is typically a whole function or block.  ~30 % of documents are kept in
their natural order so the model does not forget ordinary left-to-right modelling.

    python scripts/build_fim_data.py --tokens 6000000
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tokenizer import BPETokenizer  # noqa: E402

DOCS = ROOT / "data/docs/train"
OUT = ROOT / "data/tokens/fim.bin"

# Languages that carry useful structure for FIM (prose is not split).
FIM_LANGS = {"python", "javascript", "typescript", "java", "c", "cpp", "csharp",
             "go", "rust", "sql", "shell", "lua", "ruby", "php", "kotlin", "swift"}


def split_points(text: str, rng: random.Random) -> tuple[str, str, str] | None:
    """Choose prefix/middle/suffix with the middle being 20-50 % of the document."""
    n = len(text)
    if n < 400:
        return None
    mid_len = int(n * rng.uniform(0.2, 0.5))
    start = rng.randint(int(0.1 * n), n - mid_len - 1)
    # snap to line boundaries so the pieces look like real code fragments
    nl_start = text.find("\n", start)
    if nl_start == -1:
        return None
    end = min(n, start + mid_len)
    nl_end = text.find("\n", end)
    end = n if nl_end == -1 else nl_end + 1
    pre, mid, suf = text[:nl_start + 1], text[nl_start + 1:end], text[end:]
    if len(mid) < 80:
        return None
    return pre, suf, mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=6_000_000)
    ap.add_argument("--fim-rate", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    tok = BPETokenizer.load(ROOT / "data/tokenizer.json")
    sp = tok.special_to_id
    eot = sp["<|endoftext|>"]
    rng = random.Random(args.seed)

    docs = []
    for shard in sorted(DOCS.glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["lang"] in FIM_LANGS:
                    docs.append((rec["text"], rec["lang"]))
    rng.shuffle(docs)
    print(f"{len(docs)} candidate documents")

    buf = []
    n_tokens = 0
    n_fim = n_plain = 0
    with open(OUT, "wb") as f:
        for text, lang in docs:
            if n_tokens >= args.tokens:
                break
            use_fim = rng.random() < args.fim_rate
            ids = None
            if use_fim:
                parts = split_points(text, rng)
                if parts is not None:
                    pre, suf, mid = parts
                    ids = ([sp["<|fim_prefix|>"]] + tok.encode(pre, allow_special=False)
                           + [sp["<|fim_suffix|>"]] + tok.encode(suf, allow_special=False)
                           + [sp["<|fim_middle|>"]] + tok.encode(mid, allow_special=False))
                    n_fim += 1
            if ids is None:
                ids = tok.encode(text, allow_special=False)
                n_plain += 1
            ids = ids[:2000] + [eot]
            buf.append(np.asarray(ids, dtype=np.uint16))
            n_tokens += len(ids)
        stream = np.concatenate(buf)
        f.write(stream.tobytes())

    print(f"wrote {OUT}: {len(stream)/1e6:.2f}M tokens "
          f"({n_fim} FIM documents, {n_plain} plain documents)")


if __name__ == "__main__":
    main()
