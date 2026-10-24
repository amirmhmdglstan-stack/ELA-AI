import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import (  # noqa: E402
    Deduper, clean_document, contains_secret, detect_license, entropy,
    passes_quality, simhash64,
)

MIT = """MIT License

Copyright (c) 2024 Someone

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction."""

APACHE = """                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/"""

GPL = """                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007
 Copyright (C) 2007 Free Software Foundation, Inc."""

BSD3 = """Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice
2. Redistributions in binary form must reproduce the above copyright notice
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software."""


def test_license_detection():
    assert detect_license(MIT) == "MIT"
    assert detect_license(APACHE) == "Apache-2.0"
    assert detect_license(BSD3) == "BSD-3-Clause"
    assert detect_license(GPL) == "GPL-3.0"
    assert detect_license("all rights reserved, ask us nicely") == "UNKNOWN"


def test_license_priority_permissive_over_copyleft_bundle():
    bundled = MIT + "\n\nThe GNU General Public License applies to the bundled tools."
    assert detect_license(bundled) == "MIT"


def test_secret_detection():
    assert contains_secret('password = "hunter2hunter2hunter2"')
    assert contains_secret("-----BEGIN RSA PRIVATE KEY-----\nMIIE")
    assert contains_secret("key = 'AKIAABCDEFGHIJKLMNOP'")
    assert not contains_secret("password = get_password()\nx = 1")


def test_quality_filters():
    good = "def f(x):\n    return x + 1\n" * 10
    assert passes_quality(good, "python")[0]
    assert not passes_quality("x = 1", "python")[0]
    long_lines = ("x = 1\n" * 3) + "a" * 500 + "\n" * 3
    assert passes_quality(long_lines, "python")[1] == "line_too_long"
    rep = ("aaaa " * 200)
    assert not passes_quality(rep, "python")[0]
    blob = 'data = "' + "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2" * 12 + '"\n' * 10
    assert not passes_quality(blob, "python")[0]


def test_clean_document():
    assert clean_document("a\r\nb\r\n") == "a\nb"
    assert clean_document("bin\x00ary") is None
    assert clean_document("trailing   \nspace\t\n") == "trailing\nspace"


def test_entropy():
    assert entropy("ab") == pytest.approx(1.0)
    assert entropy("a" * 100) == 0.0


def test_simhash_similar_vs_different():
    a = "def foo(x):\n    return x * 2\n" * 40
    b = "def foo(x):\n    return x * 2\n" * 40 + "print('extra')\n"
    c = "class Bar:\n    def baz(self):\n        pass\n" * 40
    d_ab = bin(simhash64(a) ^ simhash64(b)).count("1")
    d_ac = bin(simhash64(a) ^ simhash64(c)).count("1")
    assert d_ab < d_ac
    # identical documents give distance 0; a small edit gives a few bits,
    # an unrelated document roughly half of the 64 bits
    assert d_ab <= 8 and d_ac >= 20


def test_deduper_flags_exact_and_near():
    d = Deduper()
    a = "def foo(x):\n    return x * 2\n" * 40
    b = "def foo(x):\n    return x * 2\n" * 40 + "print(1)\n"
    assert d.is_duplicate(a, "a") is None
    assert d.is_duplicate(a, "a2") == "exact"
    assert d.is_duplicate(b, "b") == "near"
    assert d.is_duplicate("完全不同的一段代码" * 100, "c") in (None, "near")
