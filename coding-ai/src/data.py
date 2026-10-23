"""Dataset pipeline: filtering, cleaning, deduplication, splitting, packing.

The pipeline is deliberately rule-based and auditable: every document that
reaches the tokenizer has passed a named filter, and the filters (and how many
documents they removed) are recorded in the run manifest.

Stages implemented here
-----------------------
1. ``detect_license``      - classify the LICENSE/COPYING text found in a
                             repository archive against a permissive allow-list.
2. ``iter_repo_files``     - walk a tarball, select source files by extension,
                             skipping vendored/generated/binary content.
3. ``clean_document``      - normalise newlines, reject non-UTF-8/binary blobs.
4. ``passes_quality``      - length / entropy / line-length / boilerplate checks.
5. ``contains_secret``     - reject files that appear to contain credentials.
6. ``dedupe``              - exact hashing + SimHash near-duplicate detection.
7. ``pack_tokens``         - concatenate token ids into uint16 shards.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

# --------------------------------------------------------------------------- #
# Languages
# --------------------------------------------------------------------------- #

LANG_BY_EXT = {
    ".py": "python", ".pyi": "python", ".pyx": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript",
    ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby", ".php": "php", ".pl": "perl", ".r": "r",
    ".lua": "lua",
    ".sql": "sql",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell", ".ps1": "powershell",
    ".html": "html", ".htm": "html", ".vue": "vue", ".svelte": "svelte",
    ".css": "css", ".scss": "css", ".sass": "css", ".less": "css",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".md": "markdown", ".rst": "markdown", ".txt": "markdown", ".adoc": "markdown",
    ".xml": "xml", ".ini": "ini", ".cfg": "ini", ".env.example": "ini",
    ".ex": "elixir", ".exs": "elixir", ".erl": "erlang", ".hs": "haskell",
    ".swift": "swift", ".m": "objective-c", ".dart": "dart", ".jl": "julia",
    ".zig": "zig", ".nim": "nim", ".clj": "clojure", ".lisp": "lisp", ".el": "lisp",
    ".tex": "tex", ".proto": "protobuf", ".graphql": "graphql",
}

# Languages we explicitly keep.  Anything else is dropped: with a small model
# every extra language dilutes limited capacity (documented in DATA_SOURCES.md).
KEEP_LANGS = {
    "python", "javascript", "typescript", "java", "c", "cpp", "csharp", "go",
    "rust", "sql", "shell", "html", "css", "lua", "json", "yaml", "toml",
    "markdown", "ruby", "php", "kotlin", "swift", "xml",
}

SKIP_DIR_PARTS = {
    "node_modules", "bower_components", "vendor", "vendors", "third_party",
    "third-party", "3rdparty", "external", "extern", ".git", ".hg", ".svn",
    "dist", "build", "out", "target", "obj", "__pycache__", ".mypy_cache",
    ".pytest_cache", "coverage", "htmlcov", "site-packages", "venv", ".venv",
    "env", "virtualenv", "migrations", "generated", "gen", "autogen",
    ".idea", ".vscode", "Pods", "jspm_packages", ".next", ".nuxt",
    "site", "_build", "docs/_build", "bazel-out", "cmake-build-debug",
}

SKIP_FILE_RE = re.compile(
    r"("
    r"\.min\.(js|css)$|\.map$|\.bundle\.js$|\.chunk\.js$"
    r"|package-lock\.json$|yarn\.lock$|pnpm-lock\.yaml$|poetry\.lock$|Pipfile\.lock$"
    r"|go\.sum$|Cargo\.lock$|composer\.lock$|Gemfile\.lock$"
    r"|_pb2\.py$|_pb2_grpc\.py$|\.pb\.go$|\.pb\.cc$|\.pb\.h$|\.g\.dart$|\.generated\..*"
    r"|\.designer\.cs$|\.g\.i\.cs$|AssemblyInfo\.cs$"
    r"|\.snap$|\.svg$|\.png$|\.jpg$|\.jpeg$|\.gif$|\.ico$|\.woff2?$|\.ttf$|\.eot$"
    r"|\.pdf$|\.zip$|\.gz$|\.tar$|\.jar$|\.exe$|\.dll$|\.so$|\.dylib$|\.o$|\.a$"
    r"|\.pyc$|\.class$|\.ipynb$|\.csv$|\.tsv$|\.parquet$|\.bin$|\.dat$"
    r")",
    re.IGNORECASE,
)

# Secret / credential patterns.  Presence of any of these drops the whole file:
# a small amount of lost training signal is a good trade for never learning to
# emit live credentials.
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
    re.compile(r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*"
               r"[\"'][^\"'\s]{12,}[\"']"),
]

GENERATED_MARKERS = (
    "autogenerated", "auto-generated", "do not edit", "generated by the protocol",
    "this file was generated", "code generated by", "@generated",
)

# --------------------------------------------------------------------------- #
# License detection
# --------------------------------------------------------------------------- #

PERMISSIVE_LICENSES = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib",
    "Unlicense", "CC0-1.0", "PostgreSQL", "PSF-2.0", "BSL-1.0", "X11", "curl",
}
COPYLEFT_LICENSES = {
    "GPL-2.0", "GPL-3.0", "LGPL-2.1", "LGPL-3.0", "AGPL-3.0", "MPL-2.0",
    "EPL-1.0", "EPL-2.0", "SSPL-1.0", "CC-BY-NC-4.0", "CDDL-1.0",
}

def detect_license(text: str) -> str:
    """Classify a license document.  Returns an SPDX-ish identifier or UNKNOWN.

    Order matters: distinctive permissive texts are tested before the generic
    copyleft markers, because licence bundles (e.g. Django's LICENSE.python)
    can mention the GPL while the actual code licence is BSD.
    """
    t = " ".join(text.lower().split())
    if not t:
        return "UNKNOWN"
    if "gnu affero general public license" in t:
        return "AGPL-3.0"
    if "server side public license" in t:
        return "SSPL-1.0"
    if "hippocratic license" in t:
        return "Hippocratic-2.1"
    if "business source license" in t:
        return "BUSL-1.1"
    if "boost software license" in t:
        return "BSL-1.0"
    if "mozilla public license" in t:
        return "MPL-2.0"
    if "eclipse public license" in t:
        return "EPL-2.0"
    if "creative commons attribution-noncommercial" in t:
        return "CC-BY-NC-4.0"
    if "cc0 1.0 universal" in t:
        return "CC0-1.0"
    if "this is free and unencumbered software released into the public domain" in t:
        return "Unlicense"
    if "permission is hereby granted, free of charge, to any person obtaining a copy" in t:
        if "the mit license" in t and "gnu general public license" not in t:
            return "MIT"
        if "redistributions of source code must retain" in t:
            return "BSD-3-Clause" if "neither the name" in t else "BSD-2-Clause"
        return "MIT"
    if "apache license" in t and "version 2.0" in t:
        return "Apache-2.0"
    if "permission to use, copy, modify, and/or distribute this software for any" in t:
        return "ISC"
    if "this software is provided 'as-is'" in t or 'this software is provided "as-is"' in t:
        return "Zlib"
    if "permission to use, copy, modify, and distribute this software" in t:
        return "PostgreSQL" if "postgresql" in t else "curl"
    if "redistribution and use in source and binary forms" in t:
        return "BSD-3-Clause" if "neither the name" in t else "BSD-2-Clause"
    if "python software foundation license" in t or "psf license agreement" in t:
        return "PSF-2.0"
    if "gnu lesser general public license" in t:
        return "LGPL-3.0"
    if "gnu general public license" in t:
        return "GPL-3.0" if "version 3" in t else "GPL-2.0"
    if "public domain" in t and len(t) < 2000:
        return "Unlicense"
    return "UNKNOWN"


# --------------------------------------------------------------------------- #
# Cleaning and quality filters
# --------------------------------------------------------------------------- #

_NONPRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_document(text: str) -> str | None:
    """Normalise a source document; return None if it is unusable."""
    if "\x00" in text[:4096]:
        return None
    try:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    except Exception:
        return None
    if _NONPRINTABLE.search(text[:20000]):
        return None
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip("\n")
    return text


def entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def passes_quality(text: str, lang: str) -> tuple[bool, str]:
    """Return (ok, reason).  Reason is recorded for the filter statistics."""
    n = len(text)
    min_len = 200 if lang in {"json", "yaml", "html", "css", "xml"} else 120
    if n < min_len:
        return False, "too_short"
    if n > 512_000:
        return False, "too_long"
    lines = text.split("\n")
    if len(lines) < 3:
        return False, "too_few_lines"
    if max(len(l) for l in lines) > 400:
        return False, "line_too_long"          # minified / bundled
    avg_line = n / max(1, len(lines))
    if avg_line < 8:
        return False, "lines_too_short"
    if entropy(text[:4000]) < 2.5:
        return False, "low_entropy"            # repeated filler / data blobs
    alnum = sum(c.isalnum() for c in text)
    if alnum / n < 0.25:
        return False, "low_alnum_ratio"
    # Lengthy base64 / hex blobs are usually embedded binaries.
    blob = re.search(r"[A-Za-z0-9+/=]{400,}", text)
    if blob and lang != "json":
        return False, "embedded_blob"
    head = text[:600].lower()
    if any(m in head for m in GENERATED_MARKERS):
        return False, "generated_header"
    if lang in {"json", "yaml", "toml"} and n > 100_000:
        return False, "huge_config"
    return True, "ok"


def contains_secret(text: str) -> bool:
    head = text[:200_000]
    return any(p.search(head) for p in SECRET_PATTERNS)


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #

_SIMHASH_SEED = np.uint64(0x9E3779B97F4A7C15)


def _shingle_hashes(text: str, k: int = 5, max_shingles: int = 512) -> np.ndarray:
    data = text.encode("utf-8", errors="ignore")
    if len(data) < k:
        return np.zeros(0, dtype=np.uint64)
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    windows = np.lib.stride_tricks.sliding_window_view(arr, k)
    if len(windows) > max_shingles:
        idx = np.linspace(0, len(windows) - 1, max_shingles).astype(np.int64)
        windows = windows[idx]
    h = np.zeros(len(windows), dtype=np.uint64)
    for i in range(k):
        h = h * np.uint64(1099511628211) + windows[:, i]
    h ^= _SIMHASH_SEED
    return h


def simhash64(text: str) -> int:
    """64-bit SimHash over 5-gram shingles (vectorised with numpy)."""
    h = _shingle_hashes(text)
    if len(h) == 0:
        return 0
    bits = np.unpackbits(h.view(np.uint8).reshape(-1, 8), axis=1)  # (N, 64)
    weights = bits.astype(np.int32).sum(axis=0) * 2 - len(h)       # +1/-1 votes
    signature = np.packbits((weights > 0).astype(np.uint8))
    return int.from_bytes(signature.tobytes(), "big")


class Deduper:
    """Exact-hash dedupe plus LSH-bucketed SimHash near-duplicate detection."""

    def __init__(self, near_dup_threshold: int = 3):
        self.seen_hashes: set[str] = set()
        self.buckets: dict[int, list[tuple[int, str]]] = {}
        self.threshold = near_dup_threshold
        self.n_exact = 0
        self.n_near = 0

    def is_duplicate(self, text: str, doc_id: str) -> str | None:
        digest = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
        if digest in self.seen_hashes:
            self.n_exact += 1
            return "exact"
        sig = simhash64(text)
        # 8 bands of 8 bits: candidates share at least one band.
        bands = [(sig >> (8 * b)) & 0xFF for b in range(8)]
        for b, band in enumerate(bands):
            for other_sig, other_id in self.buckets.get((b, band), ()):  # type: ignore[misc]
                if bin(sig ^ other_sig).count("1") <= self.threshold:
                    self.n_near += 1
                    return "near"
        self.seen_hashes.add(digest)
        for b, band in enumerate(bands):
            self.buckets.setdefault((b, band), []).append((sig, doc_id))  # type: ignore[arg-type]
        return None


# --------------------------------------------------------------------------- #
# Repository archives
# --------------------------------------------------------------------------- #

@dataclass
class RepoMeta:
    repo: str
    url: str
    license: str
    license_file: str = ""
    commit: str = ""
    n_files: int = 0
    n_bytes: int = 0


_LICENSE_NAMES = ("license", "licence", "copying", "copyright", "unlicense",
                  "license.txt", "license.md", "copying.txt", "mit-license")


def read_license_from_tar(tf: tarfile.TarFile) -> tuple[str, str]:
    """Classify the repository-level licence from files inside the archive.

    All top-level LICENSE/COPYING/COPYRIGHT files are inspected.  A permissive
    licence wins (repositories are often dual-licensed or ship third-party
    licence texts alongside); otherwise the first recognised licence is
    reported so the caller can reject it with a reason.
    """
    candidates = []
    for m in tf.getmembers():
        if not m.isfile() or m.size > 200_000:
            continue
        base = m.name.rsplit("/", 1)[-1].lower()
        if base.startswith(_LICENSE_NAMES) or base.split(".")[0] in _LICENSE_NAMES:
            candidates.append(m)
    candidates.sort(key=lambda m: (m.name.count("/"), 0 if m.name.rsplit("/", 1)[-1].lower()
                                   in ("license", "copying", "licence") else 1))
    found: list[tuple[str, str]] = []
    for m in candidates[:8]:
        f = tf.extractfile(m)
        if f is None:
            continue
        try:
            text = f.read().decode("utf-8", errors="ignore")
        except Exception:
            continue
        lic = detect_license(text)
        if lic != "UNKNOWN":
            found.append((lic, m.name))
    for lic, name in found:
        if lic in PERMISSIVE_LICENSES:
            return lic, name
    if found:
        return found[0]
    return "UNKNOWN", ""


def iter_repo_files(tf: tarfile.TarFile, repo: str) -> Iterator[tuple[str, str, str]]:
    """Yield (path, language, text) for every usable source file in an archive."""
    for m in tf:
        if not m.isfile() or m.size == 0 or m.size > 512_000:
            continue
        path = m.name
        parts = path.split("/")[1:]           # drop top-level "<repo>-<ref>/"
        if not parts:
            continue
        low_parts = [p.lower() for p in parts[:-1]]
        if any(p in SKIP_DIR_PARTS for p in low_parts):
            continue
        if any(p.startswith(".") and p not in {".github"} for p in low_parts[-1:]):
            continue
        base = parts[-1]
        if SKIP_FILE_RE.search(base):
            continue
        ext = "." + base.rsplit(".", 1)[-1].lower() if "." in base else ""
        lang = LANG_BY_EXT.get(ext)
        if lang is None:
            if base.lower() in {"dockerfile", "makefile", "justfile"}:
                lang = "shell"
            else:
                continue
        if lang not in KEEP_LANGS:
            continue
        f = tf.extractfile(m)
        if f is None:
            continue
        raw = f.read()
        if b"\x00" in raw[:4096]:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        yield "/".join(parts), lang, text
