import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import GPT, ModelConfig  # noqa: E402
from train import TokenDataset, save_checkpoint, load_checkpoint, TrainConfig, lr_at  # noqa: E402
from generate import ModelLoadError  # noqa: E402

CFG = dict(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4,
           d_ff=128, max_seq_len=32)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    cfg = ModelConfig(**CFG)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tcfg = TrainConfig(run_name="unit", model=cfg, lr=1e-3, max_steps=10)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, tcfg, step=5, best_val=1.23, tokens_seen=4096,
                    tokens_history=[1024, 1024])
    assert path.exists()

    model2 = GPT(cfg)
    ckpt = load_checkpoint(path, model2)
    assert ckpt["step"] == 5
    assert ckpt["best_val"] == 1.23
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(p1, p2), n1
    assert ckpt["model_config"]["d_model"] == 64


def test_corrupt_checkpoint_detected(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a torch file")
    with pytest.raises(Exception):
        torch.load(bad, weights_only=False)


def test_dataset_batches(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=5000, dtype=np.uint16)
    p = tmp_path / "a.bin"
    arr.tofile(p)
    ds = TokenDataset([str(p)], seq_len=32)
    x, y = ds.get_batch(2)
    assert x.shape == (2, 32) and y.shape == (2, 32)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_lr_schedule():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=110)
    assert lr_at(0, cfg) < lr_at(9, cfg)
    assert lr_at(9, cfg) == pytest.approx(1e-3, rel=0.1)
    assert lr_at(110, cfg) == pytest.approx(1e-4, rel=1e-3)
    assert lr_at(60, cfg) < lr_at(10, cfg)


def test_config_yaml_loads():
    from train import load_config
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(str(root / "configs/pretrain.yaml"))
    assert cfg.model.vocab_size > 1000
    assert cfg.max_steps > 0
    assert cfg.model.d_model % cfg.model.n_heads == 0


def test_missing_checkpoint_raises():
    from generate import load_model
    with pytest.raises(ModelLoadError):
        load_model("/nonexistent/model.pt")
