# coding-ai — a coding language model built from scratch, sized for a GTX 1060

A decoder-only Transformer coding model that is **implemented, tokenized, trained and
evaluated entirely inside this repository**. No pretrained weights, no pretrained
tokenizer, no hosted API: every parameter starts from random initialisation and is
updated by our own training loop on a curated, license-verified corpus.

> **Read `docs/TRAINING_REPORT.md` before forming expectations.** A model of this size
> (≈17 M parameters) trained on a CPU for a few hours learns the *shape* of code —
> indentation, common syntax, idioms, short functions, completions — but it is nowhere
> near a modern assistant. The report contains the measured numbers, good and bad.

---

## What is in the box

| Piece | File | Notes |
|---|---|---|
| Model architecture | `src/model.py` | RMSNorm, RoPE, SwiGLU, causal attention, GQA option, KV cache, sampling |
| Tokenizer | `src/tokenizer.py` | byte-level BPE trained on our corpus; code-aware pre-tokenizer; FIM + chat special tokens |
| Data pipeline | `src/data.py`, `scripts/prepare_data.py` | license verification, filtering, secret scanning, exact + near dedupe, repo-level splits |
| Pretraining | `src/train.py`, `configs/pretrain.yaml` | AdamW, warmup+cosine, grad accumulation/clipping, resumable checkpoints, val perplexity |
| Instruction tuning | `src/finetune.py`, `configs/sft.yaml` | chat template with prompt-masked loss |
| Inference | `src/generate.py`, `src/chat.py` | CLI completion, FIM completion, interactive chat, CPU/GPU |
| Evaluation | `src/evaluate.py` | perplexity/bits-per-byte, syntax validity, HumanEval pass@1, completion metrics, speed |
| Tests | `tests/` | 30 unit tests covering model, tokenizer, generation, checkpoints |

## Model card

| | |
|---|---|
| Architecture | decoder-only Transformer, pre-RMSNorm, RoPE, SwiGLU, tied embeddings |
| Layers / d_model / heads | 6 / 384 / 6 (head dim 64) |
| Feed-forward | SwiGLU, d_ff = 1024 |
| Vocabulary | 16 384 tokens (byte-level BPE + 8 special tokens) |
| Parameters | 16.93 M total · 11.93 M non-embedding |
| Context | 1024 tokens (trained at 512, extended afterwards) |
| Precision | fp32 (CPU verified) · fp16 optional on GPU |
| Checkpoint size | ≈ 68 MB fp32 (fits any GPU, loads in <1 s) |

Exact numbers, including per-layer parameter counts, are printed by
`python src/model.py --info`.

## Quick start on Windows 11 (GTX 1060, 16 GB RAM)

```powershell
git clone <this repo> && cd ELA-AI\coding-ai
py -3.11 -m venv .venv
.\.venv\Scripts\activate

# PyTorch for Pascal (sm_61). CUDA 12.8+ wheels DROPPED Pascal, so use cu121:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python scripts\check_environment.py     # verifies GPU kernels actually run

# ---- inference (needs the released checkpoint, see "Checkpoints") ----
python src\chat.py                      # interactive
python src\generate.py --prompt "def fib(n):" --max-new-tokens 120
python src\generate.py --fim --prefix "def add(a, b):" --suffix "return c"

# ---- from-scratch reproduction ----
python scripts\prepare_data.py all      # download ~1.5 GB of licensed repos, filter, tokenize
python src\train.py --config configs\pretrain.yaml
python src\finetune.py --config configs\sft.yaml
python src\evaluate.py --suite all
```

CPU-only works everywhere (that is how this checkpoint was trained):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Using the model

```bash
# completion
python src/generate.py --prompt "def parse_args(argv):" --temperature 0.4 --top-p 0.9
# fill-in-the-middle
python src/generate.py --fim --prefix "def area(r):" --suffix "    return area"
# chat (after instruction tuning)
python src/chat.py --one-shot "Write a Python function that reverses a list."
```

Sampling controls: `--temperature` (0 = greedy), `--top-k`, `--top-p`,
`--repetition-penalty`, `--max-new-tokens`, `--seed`, `--dtype float16` on GPU.

## Checkpoints

`checkpoints/` is gitignored (weights are large binaries). The released artifacts are
described in `docs/TRAINING_REPORT.md`:

```
checkpoints/pretrain/best.pt     final base model (used by default)
checkpoints/pretrain/latest.pt   last checkpoint (resume point)
checkpoints/sft/best.pt          instruction-tuned variant (if stage 2 quality held up)
```

Resume training exactly where it stopped:

```bash
python src/train.py --config configs/pretrain.yaml --resume checkpoints/pretrain/latest.pt
```

## Repository layout

```text
coding-ai/
  configs/            pretrain.yaml, sft.yaml
  src/                model, tokenizer, data, train, finetune, generate, chat, evaluate
  scripts/            prepare_data.py, check_environment.py, build_sft_data.py, smoke_test.py
  tests/              pytest suite
  docs/               TECHNICAL_PLAN.md, DATA_SOURCES.md, TRAINING_REPORT.md
  data/               (gitignored) downloaded corpora, tokenizer, packed tokens
  checkpoints/        (gitignored) trained weights
```

## Licence

Code: MIT. Training data: permissively licensed repositories, listed with their
licences in `docs/DATA_SOURCES.md`. Model weights: checkpoints trained on that data are
released for research and personal use; the underlying source files remain under their
original licences.
