#!/usr/bin/env python3
"""Local command-line coding assistant backed by the trained checkpoint.

    python src/chat.py                          # interactive REPL
    python src/chat.py --one-shot "write a bubble sort in python"
    python src/chat.py --checkpoint checkpoints/sft/best.pt --temperature 0.3

Commands inside the REPL: /reset, /system <text>, /temp <x>, /tokens, /quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chatfmt  # noqa: E402
from generate import ModelLoadError, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def detect_checkpoint(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for candidate in ("checkpoints/sft/best.pt", "checkpoints/pretrain/best.pt",
                      "checkpoints/pretrain/latest.pt"):
        p = ROOT / candidate
        if p.exists():
            return p
    return ROOT / "checkpoints/pretrain/best.pt"


class Assistant:
    def __init__(self, model, tokenizer, system=chatfmt.DEFAULT_SYSTEM,
                 temperature=0.5, top_k=40, top_p=0.9, max_new_tokens=256, seed=None,
                 repetition_penalty=1.15):
        self.model, self.tok = model, tokenizer
        self.system = system
        self.temperature = temperature
        self.top_k, self.top_p = top_k, top_p
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.repetition_penalty = repetition_penalty
        self.history: list[dict] = []

    def reset(self):
        self.history = []

    @torch.no_grad()
    def reply(self, user_message: str, stream: bool = False) -> str:
        self.history.append({"role": "user", "content": user_message})
        prompt = chatfmt.render_prompt(self.history, system=self.system)
        ids = self.tok.encode(prompt, allow_special=True)
        max_ctx = self.model.cfg.max_seq_len
        # Keep the system block and the most recent turns within the context.
        if len(ids) > max_ctx - 32:
            ids = ids[-(max_ctx - 32):]
        idx = torch.tensor([ids], dtype=torch.long,
                           device=next(self.model.parameters()).device)
        stop_ids = {self.tok.special_to_id[chatfmt.EOT],
                    self.tok.special_to_id[chatfmt.USER]}
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
        out = self.model.generate(
            idx, max_new_tokens=min(self.max_new_tokens, max_ctx - len(ids)),
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
            stop_token_ids=stop_ids, generator=gen,
            repetition_penalty=self.repetition_penalty,
        )
        new_ids = out[0, len(ids):].tolist()
        text = self.tok.decode(new_ids, skip_special=True).strip()
        self.history.append({"role": "assistant", "content": text})
        return text


HELP = """commands:
  /reset            clear the conversation
  /system <text>    set the system prompt
  /temp <value>     sampling temperature (0 = greedy)
  /tokens           show context length and model info
  /quit             exit"""


def main():
    ap = argparse.ArgumentParser(description="Chat with the locally trained coding model")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/tokenizer.json"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--one-shot", default=None)
    args = ap.parse_args()

    ckpt = detect_checkpoint(args.checkpoint)
    try:
        model, tok, meta = load_model(ckpt, args.device, args.tokenizer, args.dtype)
    except ModelLoadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if meta["step"] is not None and meta["step"] < 200:
        print("[warn] this checkpoint was trained for very few steps; "
              "expect low-quality output", file=sys.stderr)

    bot = Assistant(model, tok, temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p, max_new_tokens=args.max_new_tokens, seed=args.seed)
    banner = (f"coding-ai | checkpoint={Path(ckpt).name} step={meta['step']} "
              f"params={model.num_params()/1e6:.2f}M device={next(model.parameters()).device}")
    print(banner)
    if not sys.stdin.isatty() and args.one_shot is None:
        print("[info] stdin is not a terminal; reading lines from stdin", file=sys.stderr)

    def answer(prompt: str):
        t0 = time.time()
        text = bot.reply(prompt)
        dt = time.time() - t0
        print(text)
        print(f"\n[{dt:.1f}s]", file=sys.stderr)

    if args.one_shot:
        answer(args.one_shot)
        return 0

    print("type /help for commands\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/help":
            print(HELP)
            continue
        if line == "/reset":
            bot.reset()
            print("[conversation cleared]")
            continue
        if line.startswith("/system"):
            bot.system = line[len("/system"):].strip()
            print(f"[system] {bot.system}")
            continue
        if line.startswith("/temp"):
            try:
                bot.temperature = float(line.split()[1])
                print(f"[temperature] {bot.temperature}")
            except (IndexError, ValueError):
                print("usage: /temp 0.7")
            continue
        if line == "/tokens":
            prompt = chatfmt.render_prompt(bot.history, system=bot.system)
            n = len(tok.encode(prompt))
            print(f"[context] {n}/{model.cfg.max_seq_len} tokens, "
                  f"{len(bot.history)} messages, {model.num_params()/1e6:.2f}M params")
            continue
        answer(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
