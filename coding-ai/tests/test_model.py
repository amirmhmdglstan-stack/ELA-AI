import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import GPT, ModelConfig, apply_rope, build_rope_cache  # noqa: E402

SMALL = dict(vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4,
             d_ff=128, max_seq_len=64)


def small_model(**over):
    cfg = ModelConfig(**{**SMALL, **over})
    torch.manual_seed(0)
    return GPT(cfg)


def test_forward_shapes_and_loss():
    m = small_model()
    x = torch.randint(0, 512, (2, 16))
    y = torch.randint(0, 512, (2, 16))
    logits, loss, caches = m(x, targets=y)
    assert logits.shape == (2, 16, 512)
    assert caches is None
    assert loss.item() > 0 and torch.isfinite(loss)
    # untrained uniform-ish model loss should be near ln(vocab)
    assert abs(loss.item() - 6.24) < 0.5


def test_causality():
    """Changing a future token must not change earlier logits."""
    m = small_model().eval()
    x = torch.randint(0, 512, (1, 12))
    with torch.no_grad():
        a, _, _ = m(x)
        x2 = x.clone()
        x2[0, 7] = (x2[0, 7] + 1) % 512
        b, _, _ = m(x2)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-6)
    assert not torch.allclose(a[:, 7], b[:, 7])


def test_kv_cache_matches_full_forward():
    m = small_model().eval()
    x = torch.randint(0, 512, (1, 10))
    with torch.no_grad():
        full, _, _ = m(x)
        caches = None
        outs = []
        for t in range(x.shape[1]):
            logits, _, caches = m(x[:, t:t + 1], past_kv=caches, use_cache=True, offset=t)
            outs.append(logits[:, -1])
        stepwise = torch.stack(outs, dim=1)
    assert torch.allclose(full, stepwise, atol=1e-5)


def test_parameter_count_and_tied_weights():
    m = small_model(tie_embeddings=True)
    assert m.lm_head.weight is m.tok_emb.weight
    d, ff, L = SMALL["d_model"], SMALL["d_ff"], SMALL["n_layers"]
    expected_non_emb = L * (4 * d * d + 3 * d * ff) + (2 * L + 1) * d
    assert abs(m.num_params(non_embedding=True) - expected_non_emb) <= 8


def test_gqa_shapes():
    m = small_model(n_kv_heads=2)  # 4 query heads share 2 kv heads
    x = torch.randint(0, 512, (2, 8))
    logits, loss, _ = m(x, targets=x)
    assert logits.shape == (2, 8, 512)


def test_rope_preserves_norm():
    cos, sin = build_rope_cache(32, 8, 10000.0)
    x = torch.randn(1, 2, 32, 8)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_length_guard():
    m = small_model()
    with pytest.raises(ValueError):
        m(torch.zeros((1, 65), dtype=torch.long))


def test_gradients_flow():
    m = small_model()
    x = torch.randint(0, 512, (2, 8))
    _, loss, _ = m(x, targets=x)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
