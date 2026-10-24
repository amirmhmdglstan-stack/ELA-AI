#!/usr/bin/env python3
"""Build the training corpus: download, filter, deduplicate, train tokenizer, pack.

Usage
-----
    python scripts/prepare_data.py download            # fetch repo tarballs
    python scripts/prepare_data.py process             # filter -> jsonl.gz shards
    python scripts/prepare_data.py tokenizer           # train the BPE tokenizer
    python scripts/prepare_data.py tokenize            # text -> uint16 token shards
    python scripts/prepare_data.py all                 # everything in order

Artifacts (all gitignored, they are reproducible):
    data/raw/<owner>__<repo>.tar.gz   downloaded archives
    data/raw/repos.json               provenance + license evidence
    data/docs/<split>/<repo>.jsonl.gz filtered documents with metadata
    data/tokenizer.json               trained tokenizer
    data/tokens/<split>.bin           packed uint16 token stream
    data/tokens/manifest.json         counts, token totals, tokenizer hash
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
from collections import defaultdict
import json
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data import (  # noqa: E402
    PERMISSIVE_LICENSES, Deduper, clean_document, contains_secret, detect_license,
    iter_repo_files, passes_quality, read_license_from_tar,
)
from repos import REPOS  # noqa: E402

DATA = ROOT / "data"
RAW = DATA / "raw"
DOCS = DATA / "docs"
TOKENS = DATA / "tokens"

MAX_TARBALL_MB = 160       # skip archives larger than this (checked via HTTP)
DOWNLOAD_THREADS = 6


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Stage 1: download
# --------------------------------------------------------------------------- #

def _gh_api(path: str) -> dict | None:
    """Fetch repository metadata.

    Uses the ``gh`` CLI when it is installed (higher rate limit), otherwise the
    public REST API with ``urllib``.  Returns None if neither works, in which
    case the caller falls back to probing the default branch directly.
    """
    import subprocess
    try:
        out = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            return json.loads(out.stdout)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    except Exception:
        pass
    try:
        req = urllib.request.Request(
            f"https://api.github.com/{path}",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "coding-ai-dataprep"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _probe_tarball_url(repo: str) -> tuple[str, str] | None:
    """Find a downloadable archive without the API (tries main, then master, then HEAD)."""
    for branch in ("main", "master"):
        url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}"
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 200:
                    return url, branch
        except Exception:
            continue
    return None


def fetch_repo(entry: tuple[str, str, str]) -> dict:
    repo, lang, split = entry
    out_path = RAW / (repo.replace("/", "__") + ".tar.gz")
    meta = {"repo": repo, "lang": lang, "split": split, "url": f"https://github.com/{repo}",
            "status": "pending", "license": "UNKNOWN", "license_file": "",
            "default_branch": "", "size_kb": 0, "path": str(out_path.name)}

    info = _gh_api(f"repos/{repo}")
    if info is None:
        probed = _probe_tarball_url(repo)
        if probed is None:
            meta["status"] = "api_error"
            return meta
        meta["default_branch"] = probed[1]
        meta["archive_url"] = probed[0]
    else:
        meta["default_branch"] = info.get("default_branch", "main")
        meta["size_kb"] = info.get("size", 0)
        meta["stars"] = info.get("stargazers_count", 0)
        meta["archived"] = info.get("archived", False)
    if not out_path.exists():
        url = meta.get("archive_url") or (
            f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{meta['default_branch']}")
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                length = int(r.headers.get("Content-Length") or 0)
                if length > MAX_TARBALL_MB << 20:
                    meta["status"] = f"too_large ({length >> 20} MB)"
                    return meta
                with open(out_path, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            meta["status"] = f"download_error: {e}"
            if out_path.exists():
                out_path.unlink()
            return meta

    # Verify the license from the actual archive contents.
    try:
        with tarfile.open(out_path, "r:gz") as tf:
            lic, lic_file = read_license_from_tar(tf)
            members = [m for m in tf.getmembers() if m.isfile()][:1]
            meta["root"] = members[0].name.split("/")[0] if members else ""
    except (tarfile.TarError, OSError) as e:
        meta["status"] = f"tar_error: {e}"
        return meta

    meta["license"], meta["license_file"] = lic, lic_file
    if lic not in PERMISSIVE_LICENSES:
        meta["status"] = "license_rejected"
        out_path.unlink(missing_ok=True)
    else:
        meta["status"] = "ok"
    return meta


def stage_download(only_missing: bool = True):
    RAW.mkdir(parents=True, exist_ok=True)
    index_path = RAW / "repos.json"
    previous = {}
    if index_path.exists():
        previous = {m["repo"]: m for m in json.loads(index_path.read_text())}
    todo = [e for e in REPOS
            if not (only_missing and previous.get(e[0], {}).get("status") == "ok"
                    and (RAW / (e[0].replace('/', '__') + '.tar.gz')).exists())]
    log(f"downloading {len(todo)} repositories ({len(REPOS) - len(todo)} cached)")
    metas = dict(previous)
    with ThreadPoolExecutor(DOWNLOAD_THREADS) as ex:
        futures = {ex.submit(fetch_repo, e): e for e in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            m = fut.result()
            metas[m["repo"]] = m
            log(f"  [{i}/{len(todo)}] {m['repo']:45s} {m['status']:18s} {m['license']}")
    ordered = [metas[e[0]] for e in REPOS if e[0] in metas]
    index_path.write_text(json.dumps(ordered, indent=1))
    ok = [m for m in ordered if m["status"] == "ok"]
    log(f"usable repositories: {len(ok)}  "
        f"rejected: {sum(1 for m in ordered if m['status'] == 'license_rejected')} "
        f"(license)  too large: {sum(1 for m in ordered if m['status'] == 'too_large')}")


# --------------------------------------------------------------------------- #
# Stage 2: filter -> jsonl shards
# --------------------------------------------------------------------------- #

def stage_process(min_repo_bytes: int = 0):
    index_path = RAW / "repos.json"
    metas = json.loads(index_path.read_text())
    dedupe = Deduper()
    DOCS.mkdir(parents=True, exist_ok=True)
    stats = {"files_seen": 0, "kept": 0, "bytes_kept": 0,
             "reasons": {}, "by_lang": {}, "by_split": {}, "duplicates": {}}
    t0 = time.time()
    for m in metas:
        if m["status"] != "ok":
            continue
        tar_path = RAW / m["path"]
        split = m["split"]
        out_dir = DOCS / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (m["repo"].replace("/", "__") + ".jsonl.gz")
        n_kept = n_bytes = 0
        with tarfile.open(tar_path, "r:gz") as tf, gzip.open(out_path, "wt", encoding="utf-8") as out:
            for path, lang, text in iter_repo_files(tf, m["repo"]):
                stats["files_seen"] += 1
                doc = clean_document(text)
                if doc is None:
                    stats["reasons"]["binary_or_control"] = stats["reasons"].get("binary_or_control", 0) + 1
                    continue
                ok, reason = passes_quality(doc, lang)
                if not ok:
                    stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                    continue
                if contains_secret(doc):
                    stats["reasons"]["secret"] = stats["reasons"].get("secret", 0) + 1
                    continue
                doc_id = f"{m['repo']}:{path}"
                dup = dedupe.is_duplicate(doc, doc_id)
                if dup:
                    stats["duplicates"][dup] = stats["duplicates"].get(dup, 0) + 1
                    continue
                rec = {"id": doc_id, "repo": m["repo"], "path": path, "lang": lang,
                       "license": m["license"], "split": split, "text": doc}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_kept += 1
                n_bytes += len(doc)
                stats["by_lang"][lang] = stats["by_lang"].get(lang, 0) + len(doc)
        m["kept_files"] = n_kept
        m["kept_bytes"] = n_bytes
        stats["kept"] += n_kept
        stats["bytes_kept"] += n_bytes
        stats["by_split"][split] = stats["by_split"].get(split, 0) + n_bytes
        if n_kept == 0:
            out_path.unlink(missing_ok=True)
        log(f"  {m['repo']:45s} {split:5s} {n_kept:6d} files {n_bytes/1e6:7.2f} MB")

    index_path.write_text(json.dumps(metas, indent=1))
    (DATA / "process_stats.json").write_text(json.dumps(stats, indent=1))
    log(f"done in {time.time()-t0:.0f}s: kept {stats['kept']} files "
        f"({stats['bytes_kept']/1e6:.1f} MB); top drop reasons: "
        f"{sorted(stats['reasons'].items(), key=lambda kv: -kv[1])[:6]}")
    log(f"languages (MB): { {k: round(v/1e6, 1) for k, v in sorted(stats['by_lang'].items(), key=lambda kv: -kv[1])} }")


# --------------------------------------------------------------------------- #
# Stage 3: tokenizer
# --------------------------------------------------------------------------- #

def iter_docs(splits=("train", "val")):
    for split in splits:
        for shard in sorted((DOCS / split).glob("*.jsonl.gz")):
            with gzip.open(shard, "rt", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)


def stage_tokenizer(vocab_size: int = 16384, max_chars: int = 40_000_000):
    from tokenizer import DEFAULT_SPECIALS, train_bpe
    t0 = time.time()
    texts = (d["text"] for d in iter_docs(("train", "val")))
    tok = train_bpe(texts, vocab_size=vocab_size, specials=DEFAULT_SPECIALS,
                    min_freq=2, max_chars=max_chars)
    path = DATA / "tokenizer.json"
    tok.save(path)
    log(f"tokenizer trained: vocab={tok.vocab_size} merges={len(tok.merges)} "
        f"in {time.time()-t0:.0f}s -> {path}")
    # Report compression on held-out samples.
    sample = [d["text"] for _, d in zip(range(3000), iter_docs(("val",)))]
    chars = sum(len(s) for s in sample)
    toks = sum(len(tok.encode(s)) for s in sample)
    log(f"validation sample: {chars/1e6:.2f}M chars -> {toks/1e6:.2f}M tokens "
        f"({chars/max(1,toks):.2f} chars/token)")


# --------------------------------------------------------------------------- #
# Stage 4: pack tokens
# --------------------------------------------------------------------------- #

# Target token budget per language for the *training* split.  These numbers
# encode the editorial decision documented in DATA_SOURCES.md: general-purpose
# languages first, glue/markup languages capped so they cannot dominate a
# 17M-parameter model's capacity budget.
LANG_BUDGET = {
    "python": 11_000_000,
    "javascript": 6_500_000,
    "typescript": 4_500_000,
    "java": 3_500_000,
    "c": 3_000_000,
    "cpp": 3_500_000,
    "csharp": 1_600_000,
    "go": 1_600_000,
    "rust": 1_600_000,
    "html": 700_000,
    "css": 500_000,
    "markdown": 1_400_000,
    "shell": 400_000,
    "sql": 450_000,
    "lua": 200_000,
    "json": 250_000,
    "yaml": 200_000,
    "toml": 60_000,
    "ruby": 300_000,
    "php": 300_000,
    "kotlin": 200_000,
    "swift": 150_000,
    "xml": 100_000,
}


def _index_split(split: str) -> tuple[dict, dict]:
    """Map the split's documents by language without keeping their text."""
    by_lang: dict[str, list[tuple[str, int, int]]] = {}
    stats: dict[str, int] = {}
    for shard in sorted((DOCS / split).glob("*.jsonl.gz")):
        for lineno, line in enumerate(gzip.open(shard, "rt", encoding="utf-8")):
            rec = json.loads(line)
            lang = rec["lang"]
            n = len(rec["text"])
            by_lang.setdefault(lang, []).append((str(shard), lineno, n))
            stats[lang] = stats.get(lang, 0) + n
    return by_lang, stats


def _read_doc(shard: str, lineno: int) -> dict:
    with gzip.open(shard, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == lineno:
                return json.loads(line)
    raise IndexError(f"{shard}:{lineno} not found")


def stage_tokenize(budget_scale: float = 1.0):
    import numpy as np
    from tokenizer import BPETokenizer
    tok = BPETokenizer.load(DATA / "tokenizer.json")
    eot = tok.special_to_id["<|endoftext|>"]
    TOKENS.mkdir(parents=True, exist_ok=True)
    manifest = {"tokenizer_vocab": tok.vocab_size,
                "tokenizer_sha256": hashlib.sha256((DATA / "tokenizer.json").read_bytes()).hexdigest(),
                "lang_budget": {k: int(v * budget_scale) for k, v in LANG_BUDGET.items()},
                "splits": {}}

    for split in ("train", "val", "test"):
        if not (DOCS / split).exists():
            continue
        by_lang, lang_bytes = _index_split(split)
        if not by_lang:
            continue
        # Deterministic pseudo-random order inside each language so that the
        # budget samples many repositories instead of only the first ones.
        for lang, docs in by_lang.items():
            docs.sort(key=lambda d: hashlib.sha1(f"{d[0]}:{d[1]}".encode()).hexdigest())
        # Decide which documents to keep (per-language token budget), then read
        # each shard exactly once.  Documents inside a language are ordered by a
        # hash so the budget samples many repositories rather than a few.
        selected: dict[str, set[tuple[str, int]]] = defaultdict(set)
        if split == "train":
            for lang, docs in by_lang.items():
                quota = int(LANG_BUDGET.get(lang, 100_000) * budget_scale)
                used = 0
                for shard, lineno, nbytes in docs:
                    selected[shard].add(lineno)
                    used += nbytes / 3.25      # exact token cap applied while writing
                    if used >= quota * 1.2:
                        break
        else:
            for lang, docs in by_lang.items():
                for shard, lineno, _ in docs:
                    selected[shard].add(lineno)

        shards = sorted({sh for sh, _, _ in [d for ds in by_lang.values() for d in ds]})
        out_path = TOKENS / f"{split}.bin"
        n_written = 0
        lang_tokens: dict[str, int] = {}
        done_langs: dict[str, int] = {}
        buf = np.empty(1 << 22, dtype=np.uint16)
        pos = 0
        t0 = time.time()
        with open(out_path, "wb") as out_file:
            for shard in shards:
                keep = selected.get(shard)
                if not keep:
                    continue
                with gzip.open(shard, "rt", encoding="utf-8") as fh:
                    for lineno, line in enumerate(fh):
                        if lineno not in keep:
                            continue
                        rec = json.loads(line)
                        lang = rec["lang"]
                        if split == "train":
                            quota = int(LANG_BUDGET.get(lang, 100_000) * budget_scale)
                            if done_langs.get(lang, 0) >= quota:
                                continue
                        ids = tok.encode(rec["text"], allow_special=False)
                        ids.append(eot)
                        if split == "train":
                            ids = ids[: quota - done_langs.get(lang, 0)]
                        n = len(ids)
                        if n <= 0:
                            continue
                        if pos + n > len(buf):
                            out_file.write(buf[:pos].tobytes())
                            pos = 0
                        buf[pos:pos + n] = np.asarray(ids, dtype=np.uint16)
                        pos += n
                        n_written += n
                        lang_tokens[lang] = lang_tokens.get(lang, 0) + n
                        done_langs[lang] = done_langs.get(lang, 0) + n
            out_file.write(buf[:pos].tobytes())
        size = out_path.stat().st_size
        manifest["splits"][split] = {"files": [f"data/tokens/{split}.bin"],
                                     "tokens": size // 2, "tokens_by_lang": lang_tokens,
                                     "bytes": size, "available_bytes_by_lang": lang_bytes}
        log(f"{split}: {size//2/1e6:.2f}M tokens ({size/1e6:.1f} MB) in {time.time()-t0:.0f}s")
        log(f"   mixture: " + ", ".join(f"{k}={v/1e6:.1f}M" for k, v in
                                        sorted(lang_tokens.items(), key=lambda kv: -kv[1])))
    (TOKENS / "manifest.json").write_text(json.dumps(manifest, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["download", "process", "tokenizer", "tokenize", "all"])
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--budget-scale", type=float, default=1.0,
                    help="scale the per-language token budgets for the train split")
    args = ap.parse_args()
    if args.stage in ("download", "all"):
        stage_download()
    if args.stage in ("process", "all"):
        stage_process()
    if args.stage in ("tokenizer", "all"):
        stage_tokenizer(args.vocab_size)
    if args.stage in ("tokenize", "all"):
        stage_tokenize(args.budget_scale)


if __name__ == "__main__":
    main()
