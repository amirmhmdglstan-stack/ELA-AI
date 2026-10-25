"""A small decoder-only Transformer language model, implemented from scratch.

Everything here is written directly against PyTorch primitives (nn.Linear,
matmul, softmax, layernorm-free RMSNorm, ...).  There is no pretrained
checkpoint, no downloaded architecture code and no fused CUDA kernels, so the
model runs identically on CPU, on an old Pascal GPU (GTX 1060, sm_61) or on
modern accelerators.

Design choices
--------------
* Pre-normalisation with RMSNorm (Zhang & Sennrich, 2019).
* Rotary positional embeddings (RoPE, Su et al., 2021) applied to Q and K.
* Causal self-attention, optional grouped-query attention (GQA).
* SwiGLU feed-forward blocks (Shazeer, 2020).
* Tied input/output embeddings (saves ~vocab*d_model parameters).
* No biases anywhere (standard for modern LMs, saves memory).
* Manual (explicit) attention by default; SDPA is available as an opt-in but
  is not required.  Nothing depends on FlashAttention or Tensor Cores.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6          # set < n_heads for grouped-query attention
    d_ff: int = 1024
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    use_sdpa: bool = False       # explicit attention is the tested default

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head dimension must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation without mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float,
                     device=None, dtype=torch.float32):
    """Precompute cos/sin tables for RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                     # (T, head_dim/2)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    """Rotate (B, H, T, D) tensors by the RoPE angles.

    x is split into even/odd feature pairs, which is equivalent to the
    interleaved formulation used in the original paper.
    """
    T = x.shape[-2]
    cos = cos[offset:offset + T].unsqueeze(0).unsqueeze(0)   # (1,1,T,D/2)
    sin = sin[offset:offset + T].unsqueeze(0).unsqueeze(0)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    return out


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, past_kv=None, use_cache: bool = False, offset: int = 0):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        new_cache = (k, v) if use_cache else None
        Tk = k.shape[2]

        # Grouped-query attention: repeat KV heads to match Q heads.
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if self.cfg.use_sdpa and T > 1:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(past_kv is None),
            )
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale   # (B,H,T,Tk)
            if T > 1:
                if past_kv is None:
                    causal = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril()
                else:
                    # single new position (or a chunk) attending to everything
                    causal = torch.ones(T, Tk, dtype=torch.bool, device=x.device)
                    causal = causal.tril(diagonal=Tk - T)
                scores = scores.masked_fill(~causal, float("-inf"))
            attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            if self.dropout and self.training:
                attn = F.dropout(attn, p=self.dropout)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_cache


class SwiGLU(nn.Module):
    """Feed-forward network: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.d_ff
        self.gate_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x):
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.dropout and self.training:
            h = F.dropout(h, p=self.dropout)
        return self.down_proj(h)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, offset=0):
        h, new_cache = self.attn(self.norm1(x), cos, sin, past_kv, use_cache, offset)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class GPT(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.register_buffer(
            "rope_cos", build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)[0],
            persistent=False,
        )
        self.register_buffer(
            "rope_sin", build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)[1],
            persistent=False,
        )
        self.apply(self._init_weights)
        # Scale residual projections down by 1/sqrt(2*n_layers) for stability.
        std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ #

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # tied weights are only counted once by .parameters()
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets=None, past_kv=None, use_cache=False, offset=0):
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        offset = offset if past_kv is not None else 0

        x = self.tok_emb(idx)
        cos, sin = self.rope_cos, self.rope_sin
        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            pkv = past_kv[i] if past_kv is not None else None
            x, cache = block(x, cos, sin, pkv, use_cache, offset)
            if use_cache:
                new_caches.append(cache)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss, new_caches

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        stop_token_ids=None,
        generator: torch.Generator | None = None,
        repetition_penalty: float = 1.0,
    ):
        """Autoregressive sampling with a KV cache.

        temperature == 0 selects greedy decoding.  Returns the full sequence
        (prompt + continuation).
        """
        self.eval()
        device = idx.device
        stop = set(stop_token_ids or [])
        B = idx.shape[0]
        assert B == 1, "generate currently supports batch size 1"

        prompt_len = idx.shape[1]
        pos = prompt_len          # next sequence position to fill
        offset = 0                # RoPE offset of the tokens passed to the model
        out_tokens: list[int] = []
        caches = None
        cur = idx
        window = idx[0].tolist()

        for step in range(max_new_tokens):
            if pos >= self.cfg.max_seq_len:
                break
            logits, _, caches = self.forward(
                cur, past_kv=caches, use_cache=True, offset=offset
            )
            next_logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0 and window:
                recent = torch.tensor(sorted(set(window[-256:])), device=device)
                sel = next_logits[0, recent]
                next_logits[0, recent] = torch.where(
                    sel < 0, sel * repetition_penalty, sel / repetition_penalty
                )

            if temperature <= 1e-5:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                logits_s = next_logits / temperature
                if top_k and top_k > 0:
                    k = min(top_k, logits_s.size(-1))
                    kth = torch.topk(logits_s, k, dim=-1).values[:, -1:]
                    logits_s = logits_s.masked_fill(logits_s < kth, float("-inf"))
                if top_p and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits_s, descending=True, dim=-1)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative = torch.cumsum(probs, dim=-1)
                    remove = cumulative - probs > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits_s = torch.full_like(logits_s, float("-inf")).scatter(
                        1, sorted_idx, sorted_logits
                    )
                probs = torch.softmax(logits_s, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)

            token = int(next_id.item())
            out_tokens.append(token)
            window.append(token)
            pos += 1
            offset = pos - 1
            cur = next_id.to(device)

            if token in stop:
                break
        gen = torch.tensor([out_tokens], dtype=idx.dtype, device=device)
        return torch.cat([idx, gen], dim=1) if out_tokens else idx


def count_parameters(cfg: ModelConfig) -> dict:
    m = GPT(cfg)
    total = m.num_params()
    non_emb = m.num_params(non_embedding=True)
    del m
    return {"total": total, "non_embedding": non_emb}


if __name__ == "__main__":  # `python src/model.py` prints the model card
    import json
    import sys

    cfg = ModelConfig()
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
        import yaml
        raw = yaml.safe_load(open(sys.argv[1]))
        cfg = ModelConfig(**raw.get("model", {}))
    m = GPT(cfg)
    per_layer = cfg.n_layers and (m.blocks[0].attn.q_proj.weight.numel() * 3
                                  + m.blocks[0].attn.o_proj.weight.numel()
                                  + sum(p.numel() for p in m.blocks[0].mlp.parameters())
                                  + sum(p.numel() for p in m.blocks[0].norm1.parameters())
                                  + sum(p.numel() for p in m.blocks[0].norm2.parameters()))
    print(json.dumps({
        "config": cfg.to_dict(),
        "parameters_total": m.num_params(),
        "parameters_non_embedding": m.num_params(non_embedding=True),
        "parameters_per_layer": per_layer,
        "embedding_parameters": cfg.vocab_size * cfg.d_model,
        "kv_cache_bytes_per_1024_tokens": 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * 1024 * 4,
        "fp32_checkpoint_mb": round(m.num_params() * 4 / 1e6, 1),
        "fp16_checkpoint_mb": round(m.num_params() * 2 / 1e6, 1),
    }, indent=1))
