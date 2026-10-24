"""Regression tests for the instruction-tuning data path.

The most damaging bug in a small SFT pipeline is a misaligned label shift: the
model then learns to copy the current token instead of predicting the next one,
training loss collapses to ~0 and generation emits the stop token immediately.
These tests pin the alignment down.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import chatfmt  # noqa: E402
from finetune import SFTDataset  # noqa: E402


class CharTokenizer:
    """Deterministic 1-token-per-character stand-in for the BPE tokenizer."""

    vocab_size = 300

    def encode(self, text, allow_special=True):
        return [ord(c) % 300 for c in text]


def test_labels_predict_the_next_token(tmp_path):
    msgs = [{"role": "user", "content": "write add"},
            {"role": "assistant", "content": "def add(a, b):\n    return a + b"}]
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps({"messages": msgs}) + "\n")
    ds = SFTDataset(str(path), CharTokenizer(), max_seq_len=256)
    assert len(ds) == 1
    x, y = ds.batch(1)
    prompt, response = chatfmt.render_example(msgs)
    prompt_ids = CharTokenizer().encode(prompt)
    seq = prompt_ids + CharTokenizer().encode(response)

    # every supervised label must equal the following input token
    sup = (y[0] != -100).nonzero(as_tuple=True)[0]
    assert len(sup) == len(response)      # exactly the response tokens are supervised
    first = int(sup[0])
    assert first == len(prompt_ids) - 1
    for t in sup:
        assert int(y[0, t]) == seq[int(t) + 1]
    # the first supervised token is the first response token
    assert int(y[0, first]) == seq[len(prompt_ids)]


def test_prompt_tokens_are_not_supervised(tmp_path):
    msgs = [{"role": "user", "content": "explain loops"},
            {"role": "assistant", "content": "A loop repeats code."}]
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps({"messages": msgs}) + "\n")
    ds = SFTDataset(str(path), CharTokenizer(), max_seq_len=256)
    x, y = ds.batch(1)
    prompt, _ = chatfmt.render_example(msgs)
    n_prompt = len(CharTokenizer().encode(prompt))
    assert torch.all(y[0, :n_prompt - 1] == -100)
    assert torch.all(y[0, n_prompt - 1:-1] != -100)
    assert int(y[0, -1]) == -100      # nothing to predict after the last token


def test_long_examples_are_truncated(tmp_path):
    long_code = "x = 1\n" * 400
    msgs = [{"role": "user", "content": "write something"},
            {"role": "assistant", "content": long_code}]
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps({"messages": msgs}) + "\n")
    ds = SFTDataset(str(path), CharTokenizer(), max_seq_len=128)
    x, y = ds.batch(1)
    assert x.shape[1] <= 128


def test_chat_template_roundtrip():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    prompt, response = chatfmt.render_example(msgs)
    assert prompt.endswith(chatfmt.ASSISTANT)
    assert chatfmt.USER in prompt and chatfmt.SYSTEM in prompt
    assert response.endswith(chatfmt.EOT)
    # inference-time prompt for the same history must match the training prompt
    assert chatfmt.render_prompt(msgs[:-1]) == prompt
