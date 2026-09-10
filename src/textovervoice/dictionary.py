"""Word-level compression: common words become a 4-code dictionary hit
(1 flag + 3 code letters) instead of one code per character -- the same
trick 19th/20th-century commercial telegraph codebooks (e.g. Bentley's
Complete Phrase Code) used to cut cable costs.

See tools/build_dictionary.py for how the word list and codes were
generated: ranked by *expected character savings* (frequency-weighted by
word length), using real frequency data from `wordfreq` rather than a
guessed list. A dictionary hit only pays off for words 5+ letters long
(4 codes to send the flag+code vs. 5+ codes for the literal word) -- pure
frequency ranking picks mostly short function words ("the", "and") that
wouldn't save anything.

DICT_LOWER / DICT_TITLE (codes.py) cover the two cases worth a cheap flag:
the word as stored (lowercase) and sentence-initial capitalization. Any
other casing (ALL CAPS, mIxEd case) falls back to plain per-character
encoding via charset.py, as does any word not in the dictionary -- this is
a strict optional layer on top of the character-level protocol, never a
replacement for it, so nothing is lost for text the dictionary doesn't
cover.

Word segmentation here assumes space-delimited scripts (English and
similar). Languages without spaces between words (Chinese, Japanese, ...)
need real word segmentation, not just a different word list -- out of
scope for this module.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import codes
from .charset import Utf8CharDecoder, encode_text as _charset_encode_text
from .framing import TokenReader

_DATA_DIR = Path(__file__).parent / "data"
_WORD_RE = re.compile(r"[A-Za-z]+")

_tables: dict[str, dict[str, str]] = {}
_reverse_tables: dict[str, dict[str, str]] = {}


def _load(lang: str) -> dict[str, str]:
    if lang not in _tables:
        path = _DATA_DIR / f"dict_{lang}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no dictionary for language {lang!r}; generate one with "
                f"tools/build_dictionary.py {lang}"
            )
        with open(path) as f:
            _tables[lang] = json.load(f)
    return _tables[lang]


def _load_reverse(lang: str) -> dict[str, str]:
    if lang not in _reverse_tables:
        _reverse_tables[lang] = {code: word for word, code in _load(lang).items()}
    return _reverse_tables[lang]


def encode_text(text: str, lang: str = "en") -> list[int]:
    table = _load(lang)
    out: list[int] = []
    pos = 0

    for m in _WORD_RE.finditer(text):
        if m.start() > pos:
            out += _charset_encode_text(text[pos:m.start()])

        word = m.group()
        lower = word.lower()
        code = table.get(lower)

        if code is None:
            out += _charset_encode_text(word)
        elif word == lower:
            out.append(codes.DICT_LOWER)
            out += [ord(c) for c in code]
        elif word == lower.capitalize():
            out.append(codes.DICT_TITLE)
            out += [ord(c) for c in code]
        else:
            out += _charset_encode_text(word)  # ALL CAPS / odd casing: no cheap flag for it

        pos = m.end()

    if pos < len(text):
        out += _charset_encode_text(text[pos:])

    return out


def decode_codes(code_stream: list[int], lang: str = "en") -> str:
    reverse = _load_reverse(lang)
    reader = TokenReader(code_stream)
    decoder = Utf8CharDecoder()
    result: list[str] = []

    while reader:
        kind, value = reader.read()

        if kind == "flag" and value in (codes.DICT_LOWER, codes.DICT_TITLE):
            letters = []
            ok = True
            for _ in range(3):
                if not reader:
                    ok = False
                    break
                k2, v2 = reader.read()
                if k2 != "data":
                    ok = False
                    break
                letters.append(chr(v2))
            if ok:
                word = reverse.get("".join(letters))
                if word:
                    result.append(word.capitalize() if value == codes.DICT_TITLE else word)
            # malformed/unknown code: drop silently and resync at the next token,
            # same recovery philosophy as Utf8CharDecoder for corrupted input
            continue

        ch = decoder.feed(kind, value)
        if ch:
            result.append(ch)

    return "".join(result)
