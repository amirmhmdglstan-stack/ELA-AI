import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import GPT, ModelConfig  # noqa: E402


def build():
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4,
                      d_ff=128, max_seq_len=32)
    torch.manual_seed(1)
    return GPT(cfg).eval()


def test_greedy_is_deterministic():
    m = build()
    prompt = torch.randint(0, 256, (1, 5))
    a = m.generate(prompt, max_new_tokens=10, temperature=0.0)
    b = m.generate(prompt, max_new_tokens=10, temperature=0.0)
    assert torch.equal(a, b)
    assert a.shape == (1, 15)
    assert torch.equal(a[:, :5], prompt)


def test_sampling_reproducible_with_seed():
    m = build()
    prompt = torch.randint(0, 256, (1, 4))
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = m.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=20, generator=g1)
    b = m.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=20, generator=g2)
    assert torch.equal(a, b)


def test_top_k_limits_vocabulary():
    m = build()
    prompt = torch.randint(0, 256, (1, 3))
    torch.manual_seed(0)
    out = m.generate(prompt, max_new_tokens=32, temperature=1.5, top_k=1)
    # with top_k=1 sampling degenerates to greedy
    greedy = m.generate(prompt, max_new_tokens=32, temperature=0.0)
    assert torch.equal(out, greedy)


def test_context_length_is_enforced():
    m = build()
    prompt = torch.randint(0, 256, (1, 28))
    out = m.generate(prompt, max_new_tokens=100, temperature=0.0)
    assert out.shape[1] <= 32


def test_stop_token():
    m = build()
    prompt = torch.randint(0, 256, (1, 3))
    greedy = m.generate(prompt, max_new_tokens=10, temperature=0.0)
    stop = int(greedy[0, 5])
    out = m.generate(prompt, max_new_tokens=10, temperature=0.0, stop_token_ids=[stop])
    assert int(out[0, -1]) == stop


def test_repetition_penalty_runs():
    m = build()
    prompt = torch.randint(0, 256, (1, 4))
    out = m.generate(prompt, max_new_tokens=8, temperature=0.0, repetition_penalty=1.2)
    assert out.shape[1] == 12
