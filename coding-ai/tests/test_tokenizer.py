import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tokenizer import BPETokenizer, DEFAULT_SPECIALS, train_bpe  # noqa: E402

CORPUS = ["""
def fibonacci(n: int) -> int:
    \"\"\"Return the n-th Fibonacci number.\"\"\"
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)
""" * 40]


@pytest.fixture(scope="module")
def tok():
    return train_bpe(CORPUS, vocab_size=400, min_freq=1, verbose=False)


def test_roundtrip_ascii(tok):
    s = "def foo(x):\n    return x * 2  # doubles\n"
    assert tok.decode(tok.encode(s)) == s


def test_roundtrip_unicode(tok):
    s = "x = 'héllo wörld ünïcode 日本語 🎉'\n"
    assert tok.decode(tok.encode(s)) == s


def test_roundtrip_messy_bytes(tok):
    s = "\t\tindented\r\nmixed\t\ttabs   and   spaces\n\x7f\x01 weird"
    ids = tok.encode(s)
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == s


def test_special_tokens(tok):
    s = "a<|endoftext|>b<|fim_prefix|>c"
    ids = tok.encode(s)
    assert tok.special_to_id["<|endoftext|>"] in ids
    assert tok.special_to_id["<|fim_prefix|>"] in ids
    assert tok.decode(ids) == s
    assert "<|" not in tok.decode(ids, skip_special=True)


def test_deterministic(tok):
    s = "for i in range(10): print(i)\n"
    assert tok.encode(s) == tok.encode(s)


def test_save_load(tmp_path, tok):
    p = tmp_path / "tok.json"
    tok.save(p)
    t2 = BPETokenizer.load(p)
    assert t2.vocab_size == tok.vocab_size
    assert t2.merges == tok.merges
    s = "import os\nos.path.join('a', 'b')\n"
    assert t2.encode(s) == tok.encode(s)


def test_indentation_preserved(tok):
    src = "if x:\n    y = 1\n    z = 2\n"
    assert tok.decode(tok.encode(src)) == src


def test_compression_beats_characters(tok):
    s = "def fibonacci(n: int) -> int:\n    return n\n" * 20
    ids = tok.encode(s)
    assert len(ids) < len(s) / 1.5
