#!/usr/bin/env python3
"""Honest evaluation of the trained model.

Suites
------
ppl         held-out validation loss, perplexity and bits-per-byte
syntax      generate k completions for held-out prompts and check they parse
completion  prefix -> continuation exact-match / normalised edit distance
humaneval   pass@1 on OpenAI HumanEval (MIT licensed, downloaded separately)
speed       tokens/second for prefill and decoding, peak RSS
all         runs everything and writes eval/results.json

Generated code is untrusted and is executed only inside a subprocess with a
timeout and resource limits, in a scratch directory.

    python src/evaluate.py --suite all --checkpoint checkpoints/pretrain/best.pt
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT  # noqa: E402
from generate import ModelLoadError, complete, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval_data"
RESULTS = ROOT / "eval"


# --------------------------------------------------------------------------- #
# Perplexity
# --------------------------------------------------------------------------- #

@torch.no_grad()
def suite_perplexity(model, tok, cfg) -> dict:
    from train import TokenDataset
    val_files = sorted((ROOT / "data/tokens").glob("val*.bin"))
    if not val_files:
        return {"error": "no validation tokens found (run prepare_data.py tokenize)"}
    seq_len = min(512, cfg.max_seq_len)      # match the training window for comparability
    ds = TokenDataset([str(p) for p in val_files], seq_len, seed=1234)
    n_batches = min(50, max(4, ds.total // (seq_len * 4)))
    losses, n_tokens = [], 0
    for _ in range(n_batches):
        x, y = ds.get_batch(4)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
        n_tokens += y.numel()
    mean_loss = float(np.mean(losses))
    # bits per byte is tokenizer-independent: (loss / ln2) / bytes-per-token
    raw = np.concatenate([np.asarray(np.memmap(p, dtype=np.uint16, mode="r")[:100_000])
                          for p in val_files])
    sample_text = tok.decode([int(i) for i in raw[:5000]])
    n_bytes = max(1, len(sample_text.encode("utf-8")))
    bpb = mean_loss / math.log(2) * (len(raw[:5000]) / n_bytes)
    return {"val_loss": round(mean_loss, 4), "val_ppl": round(math.exp(min(20, mean_loss)), 3),
            "bits_per_byte": round(bpb, 4), "batches": n_batches, "tokens": n_tokens}


# --------------------------------------------------------------------------- #
# Held-out completion prompts
# --------------------------------------------------------------------------- #

def held_out_prompts(n: int = 40) -> list[dict]:
    """Function-definition prefixes taken from the validation split."""
    out = []
    for shard in sorted((ROOT / "data/docs/val").glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["lang"] != "python":
                    continue
                lines = rec["text"].split("\n")
                for i, ln in enumerate(lines):
                    if ln.startswith("def ") and i + 1 < len(lines) and lines[i + 1].strip():
                        prefix = "\n".join(lines[i:i + 2]) + "\n"
                        body = "\n".join(lines[i + 2:i + 14])
                        if len(body) > 60:
                            out.append({"prompt": prefix, "reference": body,
                                        "id": rec["id"] + f":{i}"})
                if len(out) >= n * 3:
                    break
    return out[:n]


def normalised_edit_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def python_syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def python_syntax_ok_repaired(code: str, max_drop_lines: int = 4) -> bool:
    """Syntax check tolerant of a generation that was cut off mid-statement.

    Generated samples are truncated at a token budget, which very often leaves an
    unterminated string/bracket/block.  Dropping up to four trailing lines
    separates "the model wrote broken syntax" from "we stopped it mid-token".
    """
    lines = code.split("\n")
    for drop in range(max_drop_lines + 1):
        candidate = "\n".join(lines[:len(lines) - drop]) + "\n"
        if python_syntax_ok(candidate):
            return True
    return False


def js_syntax_ok(code: str) -> bool | None:
    node = _which("node")
    if not node:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([node, "--check", path], capture_output=True, timeout=20)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def c_syntax_ok(code: str, cpp: bool = False) -> bool | None:
    cc = _which("g++" if cpp else "gcc")
    if not cc:
        return None
    suffix = ".cpp" if cpp else ".c"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([cc, "-fsyntax-only", path], capture_output=True, timeout=30)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def _which(cmd: str) -> str | None:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(p) / cmd
        if cand.exists():
            return str(cand)
    return None


def suite_syntax(model, tok, k: int = 3) -> dict:
    prompts = held_out_prompts(24)
    results = {"python_ok": 0, "python_ok_repaired": 0, "python_n": 0, "examples": []}
    for p in prompts:
        for _ in range(k):
            out = complete(model, tok, tok.encode(p["prompt"]), max_new_tokens=160,
                           temperature=0.6, top_k=40, top_p=0.95, seed=None)
            code = p["prompt"] + out["text"]
            results["python_n"] += 1
            ok = python_syntax_ok(code.split("\n\n\ndef ")[0].rstrip() + "\n")
            results["python_ok"] += int(ok)
            results["python_ok_repaired"] += int(python_syntax_ok_repaired(code))
            if len(results["examples"]) < 4:
                results["examples"].append({"prompt": p["prompt"], "completion": out["text"],
                                            "syntax_ok": bool(ok)})
    results["python_syntax_rate"] = round(results["python_ok"] / max(1, results["python_n"]), 3)
    results["python_syntax_rate_repaired"] = round(
        results["python_ok_repaired"] / max(1, results["python_n"]), 3)
    # Other languages: single sample per prompt for cheap syntax checks
    for lang, ext, checker in (("javascript", ".js", js_syntax_ok), ("c", ".c", c_syntax_ok)):
        samples = [r for r in _iter_val_records(lang, 6)]
        ok_n = n = 0
        for rec in samples:
            prefix = rec["text"][:300]
            out = complete(model, tok, tok.encode(prefix), max_new_tokens=64,
                           temperature=0.4, top_k=30, top_p=0.95)
            n += 1
            res = checker(prefix + out["text"])
            if res is not None:
                ok_n += int(res)
        if n:
            results[f"{lang}_syntax_rate"] = round(ok_n / n, 3)
            results[f"{lang}_n"] = n
    return results


def _iter_val_records(lang: str, n: int):
    count = 0
    for shard in sorted((ROOT / "data/docs/val").glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["lang"] == lang and len(rec["text"]) > 400:
                    yield rec
                    count += 1
                    if count >= n:
                        return


def suite_completion(model, tok, n: int = 30) -> dict:
    prompts = held_out_prompts(n)
    em = 0
    dists = []
    for p in prompts:
        out = complete(model, tok, tok.encode(p["prompt"]), max_new_tokens=64,
                       temperature=0.0, top_k=0, top_p=0.0)
        gen = out["text"].strip()
        ref = p["reference"].strip()
        dists.append(normalised_edit_distance(gen, ref))
        em += int(gen.split("\n")[0].strip() == ref.split("\n")[0].strip() and gen)
    return {"n": len(prompts), "first_line_exact_match": round(em / max(1, len(prompts)), 3),
            "mean_normalised_edit_distance": round(float(np.mean(dists)), 3) if dists else None}


# --------------------------------------------------------------------------- #
# HumanEval
# --------------------------------------------------------------------------- #

SANDBOX_PREAMBLE = """
import os, sys, signal, resource, random, math, re, collections, itertools, functools, string
try:
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
except Exception:
    pass
"""


def run_python_untrusted(code: str, timeout: float = 8.0) -> tuple[bool, str]:
    """Execute generated code in a separate process with limits (best effort)."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prog.py"
        path.write_text(SANDBOX_PREAMBLE + code, encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "HOME": d}
        try:
            r = subprocess.run([sys.executable, "-I", str(path)], cwd=d, env=env,
                               capture_output=True, text=True, timeout=timeout)
            return r.returncode == 0, (r.stderr or r.stdout)[-500:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


STOP_SEQUENCES = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@", "\nasync def "]


def truncate_at_stop(text: str, stops=STOP_SEQUENCES) -> str:
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def suite_humaneval(model, tok, max_problems: int | None = None, max_new_tokens: int = 200,
                    temperature: float = 0.0, n_samples: int = 1) -> dict:
    path = EVAL_DIR / "HumanEval.jsonl.gz"
    if not path.exists():
        return {"error": "eval_data/HumanEval.jsonl.gz missing (see docs/DATA_SOURCES.md)"}
    problems = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]
    if max_problems:
        problems = problems[:max_problems]
    passed = 0
    failures = []
    t0 = time.time()
    for i, p in enumerate(problems, 1):
        any_pass = False
        last_err = ""
        for s in range(n_samples):
            out = complete(model, tok, tok.encode(p["prompt"]),
                           max_new_tokens=max_new_tokens, temperature=temperature,
                           top_k=40 if temperature > 0 else 0,
                           top_p=0.95 if temperature > 0 else 0.0,
                           seed=(1000 + s) if temperature > 0 else None,
                           stop_strings=tuple(STOP_SEQUENCES))
            completion = truncate_at_stop(out["text"])
            program = (p["prompt"] + completion + "\n\n" + p["test"] +
                       f"\n\ncheck({p['entry_point']})\n")
            ok, err = run_python_untrusted(program)
            if ok:
                any_pass = True
                break
            last_err = err
        passed += int(any_pass)
        if not any_pass and len(failures) < 8:
            failures.append({"task_id": p["task_id"], "completion": completion[:400],
                             "error": last_err.strip()[:300]})
        if i % 10 == 0:
            print(f"  humaneval {i}/{len(problems)} pass={passed} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return {"pass@1": round(passed / max(1, len(problems)), 4), "passed": passed,
            "total": len(problems), "n_samples": n_samples,
            "temperature": temperature, "elapsed_s": round(time.time() - t0, 1),
            "failures": failures}


# --------------------------------------------------------------------------- #
# Speed / memory
# --------------------------------------------------------------------------- #

@torch.no_grad()
def suite_speed(model, tok) -> dict:
    device = str(next(model.parameters()).device)
    prompt = "def quicksort(arr):\n    " * 8
    ids = tok.encode(prompt)[:256]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    t0 = time.time()
    model.generate(idx, max_new_tokens=32, temperature=0.0)
    prefill_and_32 = time.time() - t0
    t0 = time.time()
    out = model.generate(idx, max_new_tokens=128, temperature=0.0)
    total = time.time() - t0
    new = out.shape[1] - idx.shape[1]
    decode_tps = new / total
    try:
        import resource
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        rss_gb = None
    vram = None
    if device.startswith("cuda"):
        vram = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return {"device": device, "prompt_tokens": len(ids), "new_tokens": new,
            "decode_tokens_per_s": round(decode_tps, 1),
            "total_seconds_128": round(total, 2),
            "prefill_plus_32_s": round(prefill_and_32, 2),
            "peak_rss_gb": round(rss_gb, 2) if rss_gb else None,
            "peak_vram_gb": vram,
            "params_millions": round(model.num_params() / 1e6, 2)}


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/pretrain/best.pt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data/tokenizer.json"))
    ap.add_argument("--suite", default="all",
                    choices=["all", "ppl", "syntax", "completion", "humaneval", "speed"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--humaneval-limit", type=int, default=None)
    ap.add_argument("--humaneval-temperature", type=float, default=0.0)
    ap.add_argument("--humaneval-samples", type=int, default=1)
    ap.add_argument("--out", default=str(RESULTS / "results.json"))
    args = ap.parse_args()

    try:
        model, tok, meta = load_model(args.checkpoint, args.device, args.tokenizer)
    except ModelLoadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"[eval] {args.checkpoint} step={meta['step']} "
          f"params={model.num_params()/1e6:.2f}M device={next(model.parameters()).device}")

    results = {"checkpoint": str(args.checkpoint), "model_meta": {k: str(v) for k, v in meta.items()},
               "params": model.num_params(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
    suites = ["ppl", "speed", "syntax", "completion", "humaneval"] if args.suite == "all" else [args.suite]
    for s in suites:
        print(f"[eval] running {s} ...", flush=True)
        t0 = time.time()
        if s == "ppl":
            results["perplexity"] = suite_perplexity(model, tok, model.cfg)
        elif s == "speed":
            results["speed"] = suite_speed(model, tok)
        elif s == "syntax":
            results["syntax"] = suite_syntax(model, tok)
        elif s == "completion":
            results["completion"] = suite_completion(model, tok)
        elif s == "humaneval":
            results["humaneval"] = suite_humaneval(
                model, tok, args.humaneval_limit,
                temperature=args.humaneval_temperature, n_samples=args.humaneval_samples)
        print(f"[eval] {s} done in {time.time()-t0:.0f}s: "
              f"{ {k: v for k, v in results[s if s != 'ppl' else 'perplexity'].items() if not isinstance(v, list)} }",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            previous = json.loads(out.read_text())
            previous.update(results)
            results = previous
        except json.JSONDecodeError:
            pass
    out.write_text(json.dumps(results, indent=1))
    print(f"[eval] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
