# Technical plan and engineering decisions

This document records what was measured before the design was fixed, the decisions that
followed, and the constraints that shaped them. Numbers come from real runs on the
machine described in §1; nothing here is copied from a paper.

---

## 1. Environment as actually found

The training sandbox is **not** the target computer. Inspected with
`nvidia-smi`, `lscpu`, `free`, `torch`:

| Resource | Sandbox (where training happened) | Target (the user's PC) |
|---|---|---|
| GPU | **none** (`nvidia-smi` absent, `torch.cuda.is_available() == False`) | GTX 1060 6 GB, Pascal **sm_61** |
| CPU | 2 vCPU (Xeon @2.6 GHz, AVX-512), 2 threads | i5-12400F, 6C/12T |
| RAM | 3.9 GB | 16 GB DDR4 |
| Disk | 21 GB (20 GB free at start) | ~500 GB SSD |
| OS | Linux (Debian, Python 3.11.2) | Windows 11 |
| PyTorch | 2.5.1+cu124 wheel, CPU execution | see §2 |

Measured CPU throughput (PyTorch 2.5.1, `torch.set_num_threads(2)`, fp32):

* 1024³ matmul: **282 GFLOPS**
* Transformer forward+backward at d_model 384, 6 layers, seq 512, batch 4:
  **~1350 tokens/s**
* bf16 autocast on CPU: **3× slower** (465 tok/s) → not used
* `F.scaled_dot_product_attention` on CPU: within noise of the explicit
  implementation (~5 % faster in one run, slower in another) → explicit attention is
  the default, SDPA is an opt-in flag

**Consequence:** this is a CPU-only training environment with about 1/30 of the compute
of a single GTX 1060. Everything below is sized for that reality, and the resulting
checkpoint is tiny enough to be *comfortable* on the target GPU.

## 2. GPU compatibility research (GTX 1060, Pascal sm_61)

The GTX 1060 is compute capability 6.1. NVIDIA dropped Pascal from CUDA 13, and PyTorch
dropped Pascal from its CUDA 12.8+ wheels:

* PyTorch release notes / dev-discuss: *"Starting with PyTorch Release 2.8 … we are
  removing Maxwell and Pascal GPU support from CUDA-12.8 binaries"*; cu126 wheels retain
  `sm_50;sm_60;sm_70;sm_75;sm_80;sm_86;sm_90`.
* The final PyTorch release shipping any CUDA 12.x wheel (and therefore any Pascal
  kernels) is **2.14**; from 2.15 the CUDA 12.6 build is source-only.

**Decision:** the project targets `torch>=2.0,<2.9` and documents, for Windows:

```powershell
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

(2.4.1+cu121 is the combination most widely reported to work on Pascal; 2.5.1–2.7.x with
`cu121`/`cu124`/`cu126` also contain sm_61 kernels. **Do not** use 2.8.0+cu128: it
imports fine, reports CUDA as available, then fails at the first kernel with
*"no kernel image is available for execution on the device"*.)

The code deliberately avoids anything Pascal cannot do:

* no FlashAttention, no `torch.compile` requirement, no `scaled_dot_product_attention`
  dependency (opt-in only), no bf16 requirement (fp16 is optional and validated by
  `scripts/check_environment.py`; fp32 is the fallback), no CUDA extensions.
* `torch.float16` on GPU halves weight memory (34 MB) and KV cache; **fp32 is the
  default** because the model is small enough that memory is not the bottleneck and fp16
  on Pascal has no Tensor Cores to exploit.

`scripts/check_environment.py` performs a real GPU matmul (not just a device listing) so
that a broken Pascal/wheel combination is detected immediately instead of silently
falling back to CPU.

## 3. Architecture

Decoder-only Transformer, written directly in PyTorch in `src/model.py`:

| Component | Choice | Why |
|---|---|---|
| Normalisation | RMSNorm, pre-norm | cheap, stable, no mean subtraction |
| Positions | RoPE (θ=10000) | relative, extends the context without learned position embeddings; zero parameters |
| Attention | causal, multi-head, optional GQA | GQA (`n_kv_heads`) is implemented and unit-tested but not used: with 6 heads and a 1 KB context the KV cache is < 200 KB, so the complexity is not justified |
| FFN | SwiGLU, d_ff = 2.67·d_model | better quality per parameter than GELU MLP |
| Embeddings | tied input/output | saves 6.3 M parameters (37 % of the model) |
| Biases | none | standard, saves memory |
| Init | N(0, 0.02) + residual projections scaled by 1/√(2L) | avoids early loss spikes |
| Attention impl. | explicit matmul + `masked_fill` + softmax | works everywhere; SDPA opt-in |
| dtype | fp32 training; fp16 optional at inference | no Tensor Cores on Pascal |

Loss: cross-entropy with `ignore_index=-100` (needed for masked instruction tuning).
Inference: KV-cached autoregressive sampling with temperature / top-k / top-p /
repetition penalty / stop tokens / context-length guard.

Alternatives considered and rejected:

* **Mamba/SSM or RWKV** — plausible efficiency win at long context, but the ecosystem
  evidence at 10–20 M parameters on code is thin, and a hand-rolled implementation
  would consume the entire compute budget in debugging rather than training.
* **Mixture-of-experts** — at 17 M total parameters, routing overhead and data hunger
  dominate any capacity benefit.
* **Encoder-decoder / prefix-LM** — no advantage for next-token code completion, and it
  complicates FIM.

## 4. Model-size decision (measured, not guessed)

Candidates benchmarked on the real machine (forward + backward + AdamW step,
seq 512):

| d_model | layers | d_ff | params (total / non-emb) | tokens/s | tokens in 4 h |
|---|---|---|---|---|---|
| 256 | 6 | 768 | 9.3 M / 5.1 M | 2193 | 31.6 M |
| **384** | **6** | **1024** | **16.9 M / 10.6 M** | **1350–1560** | **~20 M** |
| 512 | 8 | 1536 | 35.7 M / 27.3 M | 802 | 11.5 M |

Scaling-law arithmetic (L = E + A·N^-0.34 + B·D^-0.28) with the measured
throughput curve puts all three within ~0.2 nats of each other, i.e. the *data* term
dominates at this budget. The 17 M-parameter point was chosen because it (a) fits the
throughput budget with a real amount of data, (b) keeps a 16 K vocabulary embedding
that costs only 6.3 M parameters thanks to weight tying, and (c) runs in ~70 MB of
memory, leaving huge headroom on a 6 GB GPU.

Training length was then set from measured throughput rather than a token/parameter
target: **~20 M tokens ≈ 0.5 epoch of the 41.5 M-token corpus** (different random
windows each step, so almost no repeated data). More tokens would be better; the
available CPU time caps it. The corpus is 2.4× larger than one pass so the run can be
resumed and extended without repeating data.

## 5. Tokenizer

* Byte-level BPE, 16 384 tokens = 256 byte tokens + 16 120 learned merges + 8 specials.
* Code-aware pre-tokeniser (newline / space-run / optional-leading-space identifiers /
  single digits / punctuation runs) rather than GPT-2's English-oriented regex.
* Trained on a 40 M-character sample of our own filtered corpus (never on a pretrained
  tokenizer).
* Specials: `<|endoftext|>`, `<|pad|>`, `<|fim_prefix|>`, `<|fim_suffix|>`,
  `<|fim_middle|>`, `<|user|>`, `<|assistant|>`, `<|system|>`.
* Measured on held-out code: **3.25 chars/token** (2.48 M tokens for 8.06 M chars),
  versus ~3.0 for a GPT-2-style tokenizer on code and 1.0 for raw bytes.
* Lossless round-trip is unit-tested for ASCII, unicode and messy/control characters.

FIM tokens are in the vocabulary and the FIM interface is implemented, but FIM training
requires a suffix-conditioned objective that the compute budget did not allow us to
train to a useful level; see the training report for the honest status.

## 6. Data pipeline

* Explicit list of 162 GitHub repositories (`scripts/repos.py`), 155 accepted after
  reading their `LICENSE`/`COPYING` files from the archive (7 rejected: GPL/AGPL/
  Hippocratic/custom).
* Repository-level train/val/test split (no file from a validation repository is ever
  trained on).
* Filters: vendored/build directories, minified/bundled files, lockfiles, generated
  headers, binaries, too-short/too-long files, low entropy, embedded base64 blobs,
  credential regexes.
* Deduplication: exact SHA-1 plus 64-bit SimHash near-duplicate detection
  (5-gram shingles, 8 LSH bands, Hamming ≤ 3).
* Result: 326 049 files scanned → **177 269 kept, 719.7 MB** of source text
  (74 000 near-duplicates and 9 690 exact duplicates removed).
* Packing: documents concatenated with `<|endoftext|>`, stored as `uint16` memmaps
  (`data/tokens/*.bin`) so training uses constant memory.
* Language mixture is budgeted explicitly (`LANG_BUDGET`) instead of taking whatever the
  repositories happen to contain: Python 11.0 M tokens, JavaScript 6.5 M, TypeScript
  4.5 M, Java/C++ 3.5 M each, C 3.0 M, Rust/C#/Go 1.6 M each, plus documentation
  (1.4 M) and smaller allocations for markup, SQL, shell, Lua.

Licensing, provenance and per-repository statistics: `docs/DATA_SOURCES.md`
(auto-generated from the run metadata).

## 7. Training recipe

* AdamW (β₁ 0.9, β₂ 0.95, wd 0.1), gradient clipping 1.0.
* LR: linear warmup 100 steps to 7e-4, cosine decay to 8e-5.
* 18 432 tokens per optimizer step (3 × 512 × 12 accumulation), fp32.
* Validation loss + perplexity every 200 steps on held-out repositories.
* Checkpoints every 100 steps, `latest.pt` + rolling `step_*.pt` + `best.pt`;
  RNG states are stored so a resumed run is bit-comparable.
* NaN/inf detection on loss and gradient norm with automatic step skipping.
* Wall-clock budget flag (`--max-minutes`) for unattended runs.

## 8. Evaluation plan

| Suite | What it measures |
|---|---|
| perplexity / bits-per-byte | language-modelling quality on held-out repositories |
| syntax validity | `ast.parse` for Python, `node --check` for JS, `gcc -fsyntax-only` for C |
| completion | first-line exact match and normalised edit distance on held-out function prefixes |
| HumanEval pass@1 | functional correctness (executed, 164 MIT-licensed problems, never trained on) |
| speed/memory | prefill and decode tokens/s, peak RSS, peak VRAM |

Generated code is executed only in a subprocess with a wall-clock timeout, CPU/memory
rlimits and a scratch directory.

## 9. Known limitations (decided up front)

1. 17 M parameters cannot match a modern assistant. Expected outcome: syntactically
   valid Python and idiomatic local completions, weak multi-step reasoning, near-zero
   HumanEval pass@1.
2. CPU training means ~20 M tokens, far below what even the "small model, many tokens"
   literature recommends. Checkpoints are therefore saved and resumable, and the exact
   continuation command is documented.
3. Instruction tuning is limited by the base model and by data that can be derived from
   permissively licensed sources without an LLM teacher.
4. No GPU was available in the training sandbox, so GPU execution is **documented and
   code-pathed but not verified on real Pascal hardware**; CPU execution is verified
   end-to-end.
