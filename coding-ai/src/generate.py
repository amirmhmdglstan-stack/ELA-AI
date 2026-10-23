#!/usr/bin/env python3
"""Load a trained checkpoint and generate code.

Examples
--------
    python src/generate.py --prompt "def factorial(n):" --max-new-tokens 120
    python src/generate.py --file snippet.py --temperature 0.2 --top-p 0.9
    python src/generate.py --fim --prefix "def add(a, b):" --suffix "return c" 

The model is loaded from a local checkpoint produced by src/train.py.  Nothing
here contacts a network or falls back to a different model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, ModelConfig  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "checkpoints" / "pretrain" / "best.pt"
DEFAULT_TOKENIZER = ROOT / "data" / "tokenizer.json"


class ModelLoadError(RuntimeError):
    pass


def load_model(ckpt_path: str | Path = DEFAULT_CKPT, device: str | None = None,
               tokenizer_path: str | Path = DEFAULT_TOKENIZER,
               dtype: str = "float32"):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise ModelLoadError(
            f"checkpoint not found: {ckpt_path}\n"
            f"Train one first (python src/train.py --config configs/pretrain.yaml) "
            f"or pass --checkpoint /path/to/model.pt"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_config" not in ckpt:
        raise ModelLoadError(f"{ckpt_path} is not a checkpoint produced by this project")
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU", file=sys.stderr)
        device = "cpu"
    model = model.to(device)
    if dtype in ("float16", "half") and device != "cpu":
        model = model.half()
    elif dtype == "bfloat16":
        model = model.to(torch.bfloat16)

    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        raise ModelLoadError(f"tokenizer not found: {tokenizer_path}")
    tok = BPETokenizer.load(tokenizer_path)
    if tok.vocab_size != cfg.vocab_size:
        raise ModelLoadError(
            f"tokenizer vocab ({tok.vocab_size}) does not match model vocab ({cfg.vocab_size})"
        )
    meta = {"step": ckpt.get("step"), "tokens_seen": ckpt.get("tokens_seen"),
            "best_val": ckpt.get("best_val"), "path": str(ckpt_path)}
    return model, tok, meta


def build_prompt(tokenizer: BPETokenizer, text: str) -> list[int]:
    """Plain completion prompt: the text as-is (prefixed with one space if
    the text starts mid-line, matching how files appear in training)."""
    return tokenizer.encode(text, allow_special=True)


@torch.no_grad()
def complete(model: GPT, tokenizer: BPETokenizer, prompt_ids: list[int],
             max_new_tokens: int = 128, temperature: float = 0.8, top_k: int = 40,
             top_p: float = 0.95, stop_strings: tuple[str, ...] = (),
             repetition_penalty: float = 1.0, seed: int | None = None,
             device: str | None = None) -> dict:
    device = device or next(model.parameters()).device
    max_ctx = model.cfg.max_seq_len
    prompt_ids = prompt_ids[-(max_ctx - 16):]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
    stop_ids = {tokenizer.special_to_id["<|endoftext|>"]}
    out = model.generate(
        idx, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, stop_token_ids=stop_ids, generator=gen,
        repetition_penalty=repetition_penalty,
    )
    new_ids = out[0, len(prompt_ids):].tolist()
    text = tokenizer.decode(new_ids, skip_special=True)
    for s in stop_strings:
        if s and s in text:
            text = text.split(s)[0]
    return {"prompt_tokens": len(prompt_ids), "new_tokens": len(new_ids), "text": text}


def fim_prompt(tokenizer: BPETokenizer, prefix: str, suffix: str) -> list[int]:
    sp = tokenizer.special_to_id
    return ([sp["<|fim_prefix|>"]] + tokenizer.encode(prefix, allow_special=False)
            + [sp["<|fim_suffix|>"]] + tokenizer.encode(suffix, allow_special=False)
            + [sp["<|fim_middle|>"]])


def main():
    ap = argparse.ArgumentParser(description="Generate code with the trained model")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--prompt", default="def add(a, b):")
    ap.add_argument("--file", default=None, help="use this file's contents as the prompt")
    ap.add_argument("--fim", action="store_true", help="fill-in-the-middle mode")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        model, tok, meta = load_model(args.checkpoint, args.device, args.tokenizer, args.dtype)
    except ModelLoadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.fim:
        prefix = args.prefix if args.prefix is not None else args.prompt
        ids = fim_prompt(tok, prefix, args.suffix)
    else:
        text = Path(args.file).read_text(encoding="utf-8") if args.file else args.prompt
        if args.file:
            # files are typically truncated from the end to fit the context
            text = text[-(model.cfg.max_seq_len * 3):]
        ids = build_prompt(tok, text)

    result = complete(
        model, tok, ids, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, seed=args.seed,
    )
    if args.json:
        print(json.dumps({**result, "meta": meta}, indent=1))
    else:
        prefix_text = (args.prefix if args.fim and args.prefix is not None else
                       (Path(args.file).read_text(encoding="utf-8") if args.file else args.prompt))
        print(prefix_text + result["text"])
        print(f"\n--- {result['new_tokens']} new tokens | checkpoint step {meta['step']} "
              f"| val {meta['best_val']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
