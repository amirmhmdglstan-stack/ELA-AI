# Training report — coding-ai 17M

Everything below was produced by the runs in this repository. Numbers are copied from
`logs/*.jsonl`, `eval/results_*.json` and `checkpoints/*/train_summary.json`; nothing is
estimated unless it says so.

**Summary in one line:** a 16.91 M-parameter decoder-only Transformer, trained from random
initialisation on 22.5 M tokens of permissively licensed code, learns Python/JS/C syntax,
indentation, idioms and short completions, but does **not** learn to solve programming
problems: HumanEval pass@1 is 0/164.

---

## 1. Hardware actually used

| | |
|---|---|
| Training | 2 vCPU (Xeon @2.6 GHz), 3.9 GB RAM, **no GPU**, PyTorch 2.5.1 CPU, fp32 |
| Training throughput | 1 400–1 600 tokens/s (forward+backward+AdamW, seq 512, batch 3 × 12 accumulation) |
| Inference measured here | **164–190 tokens/s** decode, 0.93 GB peak RSS (CPU, 2 threads) |
| Target | GTX 1060 6 GB (sm_61), i5-12400F, 16 GB — expected to be much faster; **not physically tested** (see §8) |

## 2. Model

| | |
|---|---|
| Architecture | decoder-only Transformer: RMSNorm (pre-norm), RoPE θ=10000, SwiGLU (d_ff 1024), causal attention, tied embeddings, no biases |
| Layers / d_model / heads | 6 / 384 / 6 (head dim 64, no GQA) |
| Vocabulary | 16 384 (byte-level BPE trained on our corpus + 8 special tokens) |
| Parameters | **16 913 280** total = 10 621 824 non-embedding + 6 291 456 tied embedding |
| Context | 1024 positions (RoPE), trained mostly at 512, FIM stage at 1024 |
| Weight file | 67.7 MB fp32 · 33.8 MB fp16 · 42.1 MB int8-dynamic |

## 3. Data

| | |
|---|---|
| Repositories | 162 proposed, **155 accepted** after reading their LICENSE files; 7 rejected (GPL/AGPL/LGPL/Hippocratic/custom) |
| Files | 326 049 scanned → **177 269 kept** (53 613 too short, 74 000 near-duplicates, 9 690 exact duplicates, 4 676 generated headers, 4 306 minified, 203 files with credentials, …) |
| Filtered text | **719.7 MB** across 23 languages |
| Training tokens | **41.46 M** (budgeted mixture: Python 11.0 M, JavaScript 6.5 M, TypeScript 4.5 M, Java 3.5 M, C++ 3.5 M, C 3.0 M, Rust/C#/Go 1.6 M each, markdown 1.4 M, …) |
| Validation / test | 15.57 M / 2.06 M tokens, **whole repositories held out** |
| Tokenizer efficiency | 3.25 chars/token on held-out code (8.06 M chars → 2.48 M tokens) |
| FIM corpus | 6.00 M tokens (4 762 PSM examples + 3 758 plain documents) |
| Instruction data | 2 171 train / 110 val examples, derived from docstrings/JSDoc in the corpus + 30 hand-written concept answers (no LLM teacher) |

## 4. Runs

| Stage | Checkpoint | Steps | Tokens | Wall clock | Final val |
|---|---|---|---|---|---|
| Pretraining | `checkpoints/pretrain/best.pt` | 1 100 | 20.28 M | 237 min (two sessions, resumed after a silent kill) | 3.1113 (fixed 120-window suite) |
| FIM adaptation | `checkpoints/fim/best.pt` | 120 | 2.21 M | 38 min | 3.0670 (same suite) |
| Instruction tuning | `checkpoints/sft/best.pt` | 300 | ~0.92 M (5 400 examples) | 21 min | 2.37 (SFT val split) |

Pretraining recipe: AdamW (β 0.9/0.95, wd 0.1), grad clip 1.0, lr 7e-4 → 8e-5 cosine after
100-step warmup, 18 432 tokens per optimizer step, fp32, validation every 200 steps,
checkpoint every 100 steps with optimizer state and RNG state (bit-comparable resume).

### Validation curve (pretraining)

| step | tokens (M) | train loss | val loss | val ppl |
|---|---|---|---|---|
| 1 | 0.02 | 9.73 | – | – |
| 50 | 0.92 | 6.32 | – | – |
| 100 | 1.84 | 5.13 | – | – |
| 200 | 3.69 | 3.96 | 4.75 | 115.7 |
| 400 | 7.37 | 3.30 | 4.24 | 69.5 |
| 600 | 11.06 | 2.90 | 3.77 | 43.4 |
| 800 | 14.75 | 2.60 | 3.57 | 35.5 |
| 1 000 | 18.43 | 2.45 | 2.92 | 18.5 |
| 1 100 | 20.28 | 2.15 | 3.12 | 22.6 |

The per-200-step numbers use *fresh* random validation windows, which is noisy at ±0.2 nats.
The fair comparison on 120 fixed windows is
`step 900: 3.2657 · step 1000: 3.1841 · step 1100: 3.1113` — i.e. the last checkpoint is the
best one, consistent with the cosine decay.

## 5. Measured quality

All numbers from `src/evaluate.py` (see `eval/results_base.json`, `eval/results_sft.json`).

| Metric | Base (pretrained) | Instruction-tuned |
|---|---|---|
| Held-out loss / perplexity (102 400 tokens) | **3.2288 / 25.25** | 3.4817 / 32.52 |
| Bits per byte (tokenizer-independent) | **1.409** | 1.519 |
| Python syntax validity (72 greedy-ish samples, temp 0.6) | **34.7 %** | 27.8 % |
| Python syntax validity, trailing truncation dropped (≤4 lines) | 70.8 %* | **70.8 %** |
| JavaScript syntax validity (`node --check`, n=6) | **83.3 %** | 66.7 % |
| C syntax validity (`gcc -fsyntax-only`, n=6) | 0 % | 0 % |
| Completion: first-line exact match (n=30) | 0.0 % | 0.0 % |
| Completion: normalised edit distance (n=30) | 0.793 | 0.815 |
| **HumanEval pass@1** (164 problems, greedy) | **0/164** | **0/164** |
| HumanEval pass@1, temperature 0.4 | 0/164 | – |
| Decode speed (CPU, 2 threads) | 164–190 tok/s | 189 tok/s |
| Peak RSS (inference) | 0.93–0.98 GB | 0.93 GB |

\* the base model's syntax suite did not yet use the truncation-tolerant metric; it is
included for completeness only.

**Honest reading of these numbers.** The model has learned the surface statistics of code:
indentation, brackets, decorators, imports, docstring style, common function shapes, and
which identifiers appear near which keywords. It can continue a function definition into
syntactically valid Python roughly a third of the time strictly, and ~70 % of the time when
the last few truncated lines are discounted. It cannot solve HumanEval problems, cannot
reason about an algorithm, and its completions are semantically wrong even when they parse.
That is the expected outcome for 17 M parameters trained on 22 M tokens; state-of-the-art
code models of this size class do not exist because this size class is not capable of it.

## 6. What it looks like in practice

Base model, temperature 0.5, top-p 0.9, repetition penalty 1.15 (verbatim from
`eval/samples.json`):

```python
def bubble_sort(arr):
    """Sort a list in place using bubble sort."""
    return [1, 2, 3, 5]


@pytest.mark.parametrize("func", "func", "str", "func")
def test_all_all_not_to_from_list():
    # Check that the last time is set with an array of input, it will be set
```

```javascript
const express = require('express');
const app = express();

app.get('/api/users', (req, res) => {
  return new Error(
    'data-x' +
      'data-y' +
```

Instruction-tuned model, asking for the sum of a list:

```text
def is_sist(self, dtype=None):
    """
    The sum of a list of numbers are valid.

    Args:
        self.sof_string_name(self.sof_string_name())
```

Prompt -> response *shapes* are right (it answers with a function and a docstring), the
content is not. Instruction tuning at this scale mostly teaches the chat format, not new
skills.

Typical failure modes (all reproducible):

1. **Semantic drift** — syntactically valid, semantically empty bodies (`return self.items`).
2. **Loop collapse** — at low temperature the model repeats a line (`text = "Hello World"` ×12);
   repetition penalty 1.15–1.2 is enabled by default because of this.
3. **Test-shaped output** — the corpus contains many test files, so completions often jump
   into `pytest` boilerplate.
4. **No long-range plan** — nothing that needs state across more than a few lines survives.

## 7. Reproducing this result

```bash
python scripts/prepare_data.py all        # ~55 min: download 1.5 GB, filter, tokenize, pack
python src/train.py --config configs/pretrain.yaml          # ~4 h on 2 CPU cores
python scripts/build_fim_data.py && \
  python src/train.py --config configs/pretrain_fim.yaml \
      --resume checkpoints/pretrain/best.pt --reset-step     # ~40 min
python scripts/build_sft_data.py && \
  python src/finetune.py --config configs/sft.yaml \
      --base checkpoints/fim/best.pt --out checkpoints/sft   # ~21 min
python src/evaluate.py --suite all --checkpoint checkpoints/sft/best.pt
```

To resume pretraining exactly where it stopped:

```bash
python src/train.py --config configs/pretrain.yaml --resume checkpoints/pretrain/latest.pt
```

To continue training for *more* steps than the original schedule (recommended if you have
more compute), raise `max_steps` in `configs/pretrain.yaml` and resume with
`--reset-step` off; the cosine schedule will stretch over the new length.

Estimated remaining work to reach a markedly better model on this hardware: at 1 500 tok/s,
100 M tokens ≈ 18 hours on this 2-core sandbox, or roughly 2–3 hours on the user's
i5-12400F (12 threads) and minutes on the GTX 1060. Scaling guidance and the exact command
are in `docs/TECHNICAL_PLAN.md` §4.

## 8. Hardware compatibility statement

* Training happened on **CPU only**. `torch.cuda.is_available()` was false and no
  `nvidia-smi` existed, so **no GPU code path was executed** in this environment.
* The CUDA path (`--device cuda`, optional `--dtype float16`) is implemented and guarded:
  it falls back to CPU with a warning if CUDA is unavailable, and raises a clear error for
  int8 checkpoints on GPU.
* The GTX 1060 needs Pascal (sm_61) kernels. PyTorch ≥ 2.8 with cu128 wheels **do not
  include them**. Install `torch==2.4.1` (or 2.5.1–2.7.x) from the cu121/cu124/cu126 index
  and run `scripts/check_environment.py`, which executes a real matmul on the GPU.
* Memory: the model needs ~70 MB of weights, ~19 MB of KV cache per 1024 tokens (fp32), and
  under 1 GB of RSS at inference. It fits the GTX 1060 with enormous headroom, and it runs
  on CPU at 160–190 tok/s on two cores (the i5-12400F has 12 threads, so expect roughly
  3–5× that).

## 9. Optimisations that were verified

| Variant | Size | Held-out loss | Decode speed |
|---|---|---|---|
| fp32 (default) | 67.7 MB | 4.1418 (256-token probe) | 1.00× |
| int8 dynamic (`scripts/quantize.py`) | 42.1 MB | 4.1568 (+0.015 nats) | **1.5×** |

`checkpoints/pretrain/best_int8.pt` is loadable by `src/generate.py` (CPU only).

## 10. Tests

`python -m pytest tests -q` → **40 passed** (model, tokenizer, generation, checkpoints,
data filters/dedup, SFT label alignment), plus `scripts/smoke_test.py` which trains a tiny
model end-to-end from a synthetic corpus in ~50 s.

## 11. Limitations and honest conclusion

* 20 M training tokens is 1/1000 of what a useful code model needs; the corpus (41 M tokens)
  is 2× larger than what was consumed, so resuming with more steps is genuinely useful.
* Instruction tuning has only ~2 200 examples; it teaches the format, not behaviour.
* The FIM interface works mechanically: the model fills a `<|fim_middle|>` slot with
  plausible code, but does not reliably emit the stop token afterwards (only 2.2 M tokens of
  FIM training). Treat FIM as experimental.
* HumanEval pass@1 = 0 is the single most important number in this report. The model should
  be described as a **syntax-aware code completion toy**, not as a coding assistant.
* If the goal is a useful assistant on the target machine, the next steps are: 10× more
  pretraining tokens on the same architecture, a 4–8× larger model (still trivial for a
  6 GB GPU), and a real instruction corpus — the code here supports all three.
