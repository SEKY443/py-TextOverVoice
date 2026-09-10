#!/usr/bin/env python3
"""
Generates a per-language word -> 3-letter-code dictionary for
src/textovervoice/dictionary.py, using `wordfreq` (Apache-2.0, LuminosoInsight)
for real frequency data rather than a guessed word list.

Ranking is by *expected character savings*, not raw frequency: a dictionary
hit costs 4 codes on the wire (1 flag + 3 code letters), so a word only
saves anything if it's 5+ letters, and the savings scale with length. Pure
frequency ranking picks mostly words <=4 letters ("the", "and", "for") that
wouldn't save anything -- see the wordfreq check that motivated this
(top-30 by raw frequency were almost all short function words).

score(word) = 10**zipf_frequency(word) * (len(word) - 4)

Only alphabetic, lowercase, 5+ letter words are considered -- apostrophes/
hyphens and non-Latin scripts need different tokenization and are out of
scope for this generator (word-segmentation-free languages like Chinese/
Japanese need a different approach entirely, not just a different word list).

Usage: ./.venv/bin/python3 tools/build_dictionary.py [lang] [size]
"""
from __future__ import annotations

import json
import re
import string
import sys
from itertools import product
from pathlib import Path

from wordfreq import top_n_list, zipf_frequency

WORD_RE = re.compile(r"[a-z]{5,}")
CODE_ALPHABET = string.ascii_uppercase


def build(lang: str, size: int) -> dict:
    candidates = []
    seen = set()
    for w in top_n_list(lang, 50000):
        if not WORD_RE.fullmatch(w) or w in seen:
            continue
        seen.add(w)
        z = zipf_frequency(w, lang)
        score = (10 ** z) * (len(w) - 4)
        candidates.append((score, w))

    candidates.sort(reverse=True)
    chosen = [w for _, w in candidates[:size]]

    codes = ["".join(c) for c in product(CODE_ALPHABET, repeat=3)]
    word_to_code = {w: codes[i] for i, w in enumerate(chosen)}
    return word_to_code


def main() -> None:
    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 4096

    table = build(lang, size)

    out_dir = Path(__file__).resolve().parents[1] / "src" / "textovervoice" / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"dict_{lang}.json"
    with open(out_path, "w") as f:
        json.dump(table, f, ensure_ascii=False, indent=0, separators=(",", ":"))

    total_len = sum(len(w) for w in table)
    print(f"{lang}: {len(table)} words -> {out_path}", file=sys.stderr)
    print(f"  avg word length {total_len/len(table):.1f} chars, "
          f"avg savings per hit {total_len/len(table) - 4:.1f} codes", file=sys.stderr)


if __name__ == "__main__":
    main()
