#!/usr/bin/env python3
"""Build instruction-tuning data from the permissively licensed corpus.

No external LLM is used as a teacher.  Every example is either
  (a) derived mechanically from source code that is already in our licensed
      corpus (docstrings <-> implementations, code <-> explanations), or
  (b) hand-written programming questions authored for this project.

    python scripts/build_sft_data.py --max-examples 6000
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import chatfmt  # noqa: E402

DATA = ROOT / "data"

# --------------------------------------------------------------------------- #
# Hand-written concept questions (answer text written for this project).
# --------------------------------------------------------------------------- #

CONCEPTS = [
    ("What does the `len()` function do in Python?",
     "`len(obj)` returns the number of items in a container: the number of characters in a string, "
     "elements in a list or tuple, keys in a dict, or elements in a set.\n\n"
     "```python\nlen([1, 2, 3])   # 3\nlen(\"hello\")     # 5\nlen({\"a\": 1})    # 1\n```\n\n"
     "It calls the object's `__len__` method and raises `TypeError` for objects that have no length."),
    ("How do I read a text file in Python?",
     "Use the built-in `open()` function inside a `with` block so the file is closed automatically:\n\n"
     "```python\nwith open(\"data.txt\", \"r\", encoding=\"utf-8\") as f:\n    text = f.read()\n```\n\n"
     "Use `f.readline()` or iterate over `f` to process the file line by line, which keeps memory use low."),
    ("What is a Python list comprehension?",
     "A list comprehension builds a new list from an iterable in a single expression:\n\n"
     "```python\nsquares = [x * x for x in range(10)]\nevens = [x for x in range(10) if x % 2 == 0]\n```\n\n"
     "It is equivalent to a `for` loop that appends to a list, but shorter and usually faster."),
    ("What is the difference between a list and a tuple in Python?",
     "Lists are mutable and use `[...]`; tuples are immutable and use `(...)`. "
     "Because tuples cannot change they can be used as dictionary keys and are slightly cheaper.\n\n"
     "```python\nlst = [1, 2, 3]\nlst[0] = 9            # allowed\ntup = (1, 2, 3)\n# tup[0] = 9          # TypeError\n```"),
    ("How does a `for` loop work in JavaScript?",
     "JavaScript has several loop forms. `for (let i = 0; i < n; i++)` uses an index, "
     "`for (const x of arr)` iterates values, and `for (const k in obj)` iterates keys.\n\n"
     "```js\nfor (const item of [1, 2, 3]) console.log(item);\n```"),
    ("What is a Python dictionary?",
     "A dict maps hashable keys to values and gives average O(1) lookup:\n\n"
     "```python\nd = {\"name\": \"Ada\", \"age\": 36}\nprint(d[\"name\"])       # Ada\nprint(d.get(\"job\", \"unknown\"))\nfor key, value in d.items():\n    print(key, value)\n```"),
    ("How do I handle exceptions in Python?",
     "Wrap the code in `try` and recover in `except`; use `finally` for cleanup:\n\n"
     "```python\ntry:\n    value = int(user_input)\nexcept ValueError:\n    value = 0\nfinally:\n    print(\"done\")\n```\n\n"
     "Catch the most specific exception you can, and avoid bare `except:` clauses."),
    ("What is recursion?",
     "Recursion is a function calling itself on a smaller version of the problem, with a base case "
     "that stops the recursion:\n\n"
     "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```\n\n"
     "Every recursive function needs a base case, or it will hit the recursion limit."),
    ("What is Big-O notation?",
     "Big-O describes how the running time or memory of an algorithm grows with the input size, "
     "ignoring constants. O(1) is constant, O(log n) halves the problem, O(n) scans the input, "
     "O(n log n) is a good sort, and O(n^2) is quadratic. For example, binary search is O(log n) "
     "while a single pass over a list is O(n)."),
    ("How do I reverse a string in Python?",
     "Use a slice with a step of -1:\n\n```python\ns = \"hello\"\nprint(s[::-1])   # olleh\n```\n\n"
     "`''.join(reversed(s))` gives the same result."),
    ("What is a hash table?",
     "A hash table stores key-value pairs in an array, using a hash function to turn a key into an "
     "index. Average lookup, insertion and deletion are O(1); collisions are resolved by chaining or "
     "open addressing. Python's `dict` and JavaScript's `Map` are hash tables."),
    ("How do I sort a list of dictionaries by a key in Python?",
     "Pass a `key` function to `sorted` (or `list.sort`):\n\n"
     "```python\nusers = [{\"name\": \"b\", \"age\": 30}, {\"name\": \"a\", \"age\": 25}]\n"
     "users.sort(key=lambda u: u[\"age\"])\nnewest_first = sorted(users, key=lambda u: -u[\"age\"])\n```"),
    ("What is a Python generator?",
     "A generator produces values lazily with `yield`, so memory stays small for large sequences:\n\n"
     "```python\ndef count_up(n):\n    i = 0\n    while i < n:\n        yield i\n        i += 1\n\n"
     "for x in count_up(3):\n    print(x)\n```"),
    ("What is the difference between `==` and `is` in Python?",
     "`==` compares values, `is` compares object identity. Use `is` only for singletons such as "
     "`None`, `True` and `False`.\n\n```python\na = [1, 2]\nb = [1, 2]\nprint(a == b)   # True\nprint(a is b)   # False\n```"),
    ("What is a closure in JavaScript?",
     "A closure is a function that keeps access to the variables of the scope where it was created, "
     "even after that scope has finished:\n\n"
     "```js\nfunction counter() {\n  let count = 0;\n  return () => ++count;\n}\nconst next = counter();\nnext(); // 1\nnext(); // 2\n```"),
    ("How do I create a class in Python?",
     "Use the `class` keyword and define `__init__` to initialise instances:\n\n"
     "```python\nclass Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\n"
     "    def norm(self):\n        return (self.x ** 2 + self.y ** 2) ** 0.5\n```"),
    ("What is a SQL JOIN?",
     "A JOIN combines rows from two tables using a matching condition. `INNER JOIN` keeps only "
     "matching rows, `LEFT JOIN` keeps all rows of the left table and fills NULLs otherwise:\n\n"
     "```sql\nSELECT o.id, c.name\nFROM orders AS o\nJOIN customers AS c ON c.id = o.customer_id\nWHERE o.total > 100;\n```"),
    ("How do I find the maximum value in a list in Python?",
     "Use the built-in `max()` function:\n\n```python\nprint(max([3, 7, 2]))          # 7\n"
     "print(max(words, key=len))     # longest word\n```\n\nFor an empty list supply `default=`."),
    ("What is a dictionary comprehension?",
     "It builds a dict from an iterable in one expression:\n\n"
     "```python\nsquares = {x: x * x for x in range(5)}\n```\n\n"
     "It is equivalent to a loop that assigns `d[x] = x * x`."),
    ("What does `if __name__ == \"__main__\":` do in Python?",
     "It runs the block only when the file is executed as a script, not when it is imported as a "
     "module. This lets a file be both a library and a program."),
    ("What is a mutex?",
     "A mutex (mutual exclusion lock) lets only one thread enter a critical section at a time, "
     "protecting shared data from races. In Python use `threading.Lock()`; in Go use `sync.Mutex`."),
    ("How do I remove duplicates from a list in Python?",
     "Convert to a `dict` (or `set`) when order does not matter, or when it does:\n\n"
     "```python\nnums = [1, 2, 2, 3, 1]\nunique = list(dict.fromkeys(nums))   # [1, 2, 3]\n```"),
    ("What is the difference between `let`, `const` and `var` in JavaScript?",
     "`const` creates a block-scoped binding that cannot be reassigned, `let` is block-scoped and "
     "reassignable, and `var` is function-scoped and hoisted. Prefer `const`, then `let`."),
    ("How do I write a function in Go?",
     "Use the `func` keyword, with types after the names:\n\n"
     "```go\nfunc Add(a, b int) int {\n    return a + b\n}\n```"),
    ("What is a stack data structure?",
     "A stack is a last-in-first-out container: push adds to the top, pop removes the top. "
     "In Python a list works: `append` pushes and `pop()` pops. Stacks implement undo history, "
     "expression evaluation and depth-first search."),
    ("What is a queue?",
     "A queue is first-in-first-out: items are added at the back and removed from the front. "
     "Use `collections.deque` in Python (`append` / `popleft`) for O(1) operations; a plain list "
     "with `pop(0)` is O(n)."),
    ("How do I check if a key exists in a Python dictionary?",
     "Use the `in` operator: `if \"name\" in d:` or `d.get(\"name\")`. Avoid `d[\"name\"]` when the "
     "key may be missing, since it raises `KeyError`."),
    ("What is an API?",
     "An API (application programming interface) is the contract a program exposes to other code: "
     "function signatures, request/response formats and error behaviour. A web API usually means "
     "HTTP endpoints that accept JSON and return JSON."),
    ("How do I parse JSON in Python?",
     "Use the `json` module:\n\n```python\nimport json\ndata = json.loads('{\"a\": 1}')\n"
     "text = json.dumps(data)\nwith open(\"f.json\") as fh:\n    data = json.load(fh)\n```"),
    ("What is a linked list?",
     "A linked list stores elements in nodes, each holding a value and a pointer to the next node. "
     "Insertion and deletion at a known position are O(1), but lookup is O(n) because nodes are not "
     "contiguous in memory."),
]

# --------------------------------------------------------------------------- #
# Extraction helpers
# --------------------------------------------------------------------------- #

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def docstring_examples(rec: dict, max_per_file: int = 6) -> list[dict]:
    """Turn Python functions with docstrings into write-this-function tasks."""
    out = []
    if "/test" in rec["path"].lower() or rec["path"].lower().startswith("test") \
            or rec["path"].lower().endswith("_test.py"):
        return out
    try:
        tree = ast.parse(rec["text"])
    except SyntaxError:
        return out
    lines = rec["text"].split("\n")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc or len(doc) < 40 or len(doc) > 600:
            continue
        if node.name.startswith("test") or node.name.startswith("_"):
            continue
        if ">>>" in doc:
            doc = doc.split(">>>")[0].strip()
        # keep the first paragraph only: it is the actual specification
        doc = doc.split("\n\n")[0].strip()
        if len(doc) < 40:
            continue
        if len(node.body) < 2:
            continue
        try:
            src = ast.get_source_segment(rec["text"], node)
        except Exception:
            continue
        if not src or len(src) > 1600 or len(src) < 60:
            continue
        name = node.name
        if not IDENT_RE.match(name):
            continue
        args = [a.arg for a in node.args.args if a.arg != "self"]
        sig = f"{name}({', '.join(args)})"
        doc_clean = " ".join(doc.split())
        prompt = (f"Write a Python function `{sig}` that satisfies this specification:\n\n"
                  f"{doc_clean}")
        out.append({"messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": src.strip()}],
                    "source": rec["id"]})
        # code -> explanation (the docstring is the reference explanation)
        code_block = src.strip()[:1200] if "```" not in src else None
        if code_block:
            out.append({"messages": [
                {"role": "user",
                 "content": f"Explain what this Python code does:\n\n```python\n{code_block}\n```"},
                {"role": "assistant", "content": doc_clean}],
                "source": rec["id"]})
        if len(out) >= max_per_file * 2:
            break
    return out


JS_DOC_RE = re.compile(r"/\*\*(.*?)\*/\s*(export\s+)?(async\s+)?function\s+(\w+)\s*\(([^)]*)\)",
                       re.DOTALL)
JAVA_DOC_RE = re.compile(r"/\*\*(.*?)\*/\s*(public|private|protected)[^\n{;]*?\s(\w+)\s*\(([^)]*)\)",
                         re.DOTALL)


def jsdoc_examples(rec: dict, max_examples: int = 3) -> list[dict]:
    """JSDoc / Javadoc comment above a function -> implementation task."""
    out = []
    text = rec["text"]
    pattern = JS_DOC_RE if rec["lang"] in ("javascript", "typescript") else JAVA_DOC_RE
    lang = {"javascript": "JavaScript", "typescript": "TypeScript", "java": "Java"}.get(
        rec["lang"], "Java")
    for m in pattern.finditer(text):
        doc = m.group(1)
        doc = re.sub(r"^\s*\*\s?", "", doc, flags=re.MULTILINE).strip()
        if len(doc) < 40 or len(doc) > 500:
            continue
        start = m.start()
        # capture the function body by brace matching
        i = text.find("{", m.end())
        if i == -1 or i - m.end() > 200:
            continue
        depth, j = 0, i
        while j < len(text) and j - i < 2000:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        src = text[start:j + 1]
        if len(src) < 120 or "```" in src:
            continue
        name = m.group(4) if rec["lang"] in ("javascript", "typescript") else m.group(3)
        prompt = (f"Write a {lang} function `{name}` that satisfies this specification:\n\n"
                  f"{' '.join(doc.split())}")
        out.append({"messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": src.strip()}],
                    "source": rec["id"]})
        if len(out) >= max_examples:
            break
    return out


def concept_examples() -> list[dict]:
    return [{"messages": [{"role": "user", "content": q},
                          {"role": "assistant", "content": a}], "source": "authored"}
            for q, a in CONCEPTS]


def load_train_records():
    for shard in sorted((DATA / "docs/train").glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-examples", type=int, default=9000)
    ap.add_argument("--max-python-files", type=int, default=4000)
    ap.add_argument("--max-js-files", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    examples = concept_examples()
    print(f"authored concept examples: {len(examples)}")

    py_files = js_files = 0
    last_report = -1
    for rec in load_train_records():
        if len(examples) >= args.max_examples:
            break
        if rec["lang"] == "python" and py_files < args.max_python_files:
            py_files += 1
            if rng.random() < 0.5:
                examples.extend(docstring_examples(rec))
        elif rec["lang"] in ("javascript", "typescript", "java") and js_files < args.max_js_files:
            js_files += 1
            if rng.random() < 0.15:
                examples.extend(jsdoc_examples(rec))
        if (py_files + js_files) % 1000 == 0 and (py_files + js_files) != last_report:
            last_report = py_files + js_files
            print(f"  scanned {py_files} py / {js_files} js files, {len(examples)} examples",
                  flush=True)

    # Shuffle and split deterministically (95/5).
    rng.shuffle(examples)
    out_dir = DATA / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_f = (out_dir / "train.jsonl").open("w", encoding="utf-8")
    val_f = (out_dir / "val.jsonl").open("w", encoding="utf-8")
    n_train = n_val = 0
    seen = set()
    for ex in examples:
        body = json.dumps(ex["messages"], sort_keys=True)
        h = hashlib.sha1(body.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        # never leak the same source document into both splits
        src_hash = int(hashlib.sha1(ex["source"].encode()).hexdigest(), 16)
        line = json.dumps(ex, ensure_ascii=False) + "\n"
        if src_hash % 100 < 5 and n_val < 400:
            val_f.write(line)
            n_val += 1
        else:
            train_f.write(line)
            n_train += 1
    train_f.close()
    val_f.close()
    print(f"wrote {n_train} train / {n_val} val examples to {out_dir}")

    # Sanity check the chat template against the real tokenizer
    tok_path = DATA / "tokenizer.json"
    if tok_path.exists():
        from tokenizer import BPETokenizer
        tok = BPETokenizer.load(tok_path)
        p, r = chatfmt.render_example(examples[0]["messages"])
        ids = tok.encode(p + r, allow_special=True)
        assert tok.decode(ids) == p + r
        print(f"template check ok; first example uses {len(ids)} tokens")


if __name__ == "__main__":
    main()
