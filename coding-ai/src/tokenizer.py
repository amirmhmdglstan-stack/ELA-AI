"""Byte-level BPE tokenizer, trained from scratch on the project's own corpus.

No pretrained tokenizer is used.  The implementation follows the same family of
algorithms as GPT-2 / StarCoder tokenizers (byte-level alphabet + learned BPE
merges), but the code here is self-contained and the merges are learned on our
own code corpus by ``scripts/prepare_data.py``.

Why byte-level?
---------------
* Arbitrary source text (including unicode identifiers, emoji in comments and
  invalid fragments) round-trips losslessly.
* Whitespace and indentation can be represented exactly, which matters for
  Python and for any indentation-sensitive code.

Why a custom pre-tokenizer?
---------------------------
The regex below is tuned for source code:
  * newlines are their own chunk, so line structure is learnable;
  * runs of spaces (indentation) are a separate chunk;
  * identifiers keep one optional leading space (" foo"), which captures the
    extremely common ``word + space`` pattern without gluing indentation to it;
  * digits are split out individually, which makes numbers composable;
  * runs of punctuation are grouped so operators such as ``==``, ``->``,
    ``::`` and ``</`` can become single tokens through merges.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

# --------------------------------------------------------------------------- #
# byte <-> unicode translation table (GPT-2 scheme, re-derived here)
# --------------------------------------------------------------------------- #


def _build_byte_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_UNICODE = _build_byte_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

DEFAULT_SPECIALS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
]

# Pre-tokenisation pattern (see module docstring for rationale).
CODE_PAT = re.compile(
    r"\r\n|\n"
    r"| ?[A-Za-z]+"
    r"| ?[0-9]"
    r"| ?[^\sA-Za-z0-9]+"
    r"| +"
    r"|\t+"
    r"|\s+"
)


class BPETokenizer:
    def __init__(self, merges: Sequence[Sequence[int]], specials: Sequence[str],
                 pattern: str = CODE_PAT.pattern):
        self.merges: List[tuple] = [tuple(m) for m in merges]
        self.specials: List[str] = list(specials)
        self.pattern = re.compile(pattern)
        self._rebuild()

    # -- construction ------------------------------------------------------ #

    def _rebuild(self):
        self.special_to_id = {s: 256 + len(self.merges) + i for i, s in enumerate(self.specials)}
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self.vocab_size = 256 + len(self.merges) + len(self.specials)
        self.byte_to_id = {b: b for b in range(256)}
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.rank = {}
        for i, (a, b) in enumerate(self.merges):
            new_id = 256 + i
            self.rank[(a, b)] = i
            self.id_to_bytes[new_id] = self.id_to_bytes[a] + self.id_to_bytes[b]
        for s, i in self.special_to_id.items():
            self.id_to_bytes[i] = s.encode("utf-8")
        self._cache = {}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(s) for s in self.specials) + ")")
            if self.specials else None
        )

    # -- encoding ---------------------------------------------------------- #

    def _encode_word(self, word: str) -> List[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        # Base ids are simply the raw bytes of the UTF-8 encoding.
        ids = list(word.encode("utf-8"))

        while len(ids) > 1:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            new_id = 256 + best_rank
            ids = ids[:best_i] + [new_id] + ids[best_i + 2:]
        out = tuple(ids)
        if len(self._cache) < 500_000:
            self._cache[word] = out
        return out

    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        ids: List[int] = []
        if allow_special and self._special_re is not None:
            parts = self._special_re.split(text)
            for part in parts:
                if not part:
                    continue
                if part in self.special_to_id:
                    ids.append(self.special_to_id[part])
                else:
                    ids.extend(self._encode_plain(part))
        else:
            ids.extend(self._encode_plain(text))
        return ids

    def _encode_plain(self, text: str) -> List[int]:
        out: List[int] = []
        for m in self.pattern.finditer(text):
            out.extend(self._encode_word(m.group(0)))
        return out

    # -- decoding ---------------------------------------------------------- #

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        chunks = []
        for i in ids:
            if skip_special and i in self.id_to_special:
                continue
            chunks.append(self.id_to_bytes.get(int(i), b""))
        return b"".join(chunks).decode("utf-8", errors="replace")

    # -- persistence ------------------------------------------------------- #

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "pattern": self.pattern.pattern,
            "specials": self.specials,
            "merges": self.merges,
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["merges"], data["specials"], data.get("pattern", CODE_PAT.pattern))


# --------------------------------------------------------------------------- #
# BPE training
# --------------------------------------------------------------------------- #


def _pretokenize_counts(texts: Iterable[str], max_chars: int | None = None
                        ) -> tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    buf: List[str] = []
    for text in texts:
        for m in CODE_PAT.finditer(text):
            buf.append(m.group(0))
            total += len(m.group(0))
        if len(buf) > 200_000:
            counts.update(buf)
            buf = []
        if max_chars is not None and total >= max_chars:
            break
    if buf:
        counts.update(buf)
    return counts, total


def train_bpe(texts: Iterable[str], vocab_size: int,
              specials: Sequence[str] = DEFAULT_SPECIALS,
              min_freq: int = 2,
              max_chars: int | None = None,
              verbose: bool = True) -> BPETokenizer:
    """Learn byte-level BPE merges from an iterable of text documents.

    Uses incremental pair counting: after a merge only the words that contained
    that pair are re-scored, which keeps the Python implementation fast enough
    to train a 16k vocabulary on tens of megabytes of code in a few minutes.
    """
    counts, total_chars = _pretokenize_counts(texts, max_chars=max_chars)
    if verbose:
        print(f"[tokenizer] pre-tokenized {total_chars/1e6:.1f}M chars, "
              f"{len(counts)} unique pre-tokens")
    # Prune rare words; they mostly add noise and cost quadratic time.
    words = [w for w, c in counts.items() if c >= min_freq]
    freqs = [counts[w] for w in words]
    # Word -> tuple of base ids
    word_ids = [tuple(BYTE_TO_UNICODE[b] for b in w.encode("utf-8")) for w in words]
    del counts
    char_to_id = {BYTE_TO_UNICODE[b]: b for b in range(256)}
    word_ids = [[char_to_id[c] for c in ids] for ids in word_ids]

    n_merges_target = vocab_size - 256 - len(specials)
    pair_counts: Counter = Counter()
    pair_words: dict = defaultdict(set)
    for wi, ids in enumerate(word_ids):
        f = freqs[wi]
        for pair in zip(ids, ids[1:]):
            pair_counts[pair] += f
            pair_words[pair].add(wi)

    merges: List[tuple] = []
    progress_every = max(1, n_merges_target // 20)
    for step in range(n_merges_target):
        if not pair_counts:
            break
        # Pick the most frequent pair; ties broken deterministically.
        pair, freq = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        if freq < min_freq:
            break
        new_id = 256 + len(merges)
        merges.append(pair)

        affected = list(pair_words.pop(pair, ()))
        for wi in affected:
            ids = word_ids[wi]
            f = freqs[wi]
            # remove old pairs
            for p in zip(ids, ids[1:]):
                pair_counts[p] -= f
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                    pw = pair_words.get(p)
                    if pw is not None:
                        pw.discard(wi)
            # merge
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            word_ids[wi] = out
            # add new pairs
            for p in zip(out, out[1:]):
                pair_counts[p] += f
                pair_words[p].add(wi)
        pair_counts.pop(pair, None)

        if verbose and (step + 1) % progress_every == 0:
            print(f"[tokenizer] merge {step+1}/{n_merges_target} "
                  f"pair={pair} freq={freq}")

    return BPETokenizer(merges, specials)
