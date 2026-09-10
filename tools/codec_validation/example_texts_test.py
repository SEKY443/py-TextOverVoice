#!/usr/bin/env python3
"""
Broader real-example test: runs a diverse set of actual text samples
(multiple scripts, emoji, punctuation-heavy, edge cases) through the full
CLI pipeline (protocol.build_frame -> modem.modulate -> real AMR-NB
round-trip -> modem.demodulate -> protocol.parse_frame), not just the
narrow ASCII/CJK pair realism_suite.py used.

This is meant to catch anything specific to a particular script or edge
case (RTL text, combining characters, empty input, very long input) that a
narrow test matrix might miss.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from textovervoice import modem  # noqa: E402
from textovervoice.protocol import build_frame, parse_frame  # noqa: E402
from amr_harness import roundtrip_amr  # noqa: E402

SAMPLES = {
    "empty": "",
    "single_char": "A",
    "ascii_pangram": "The quick brown fox jumps over the lazy dog",
    "punctuation_heavy": "Price: $19.99 (50% off!) -- call now @ 555-0123? #deal",
    "chinese": "你好，世界！这是一个测试。语音通道传输文字。",
    "japanese": "こんにちは世界。これはテストです。",
    "korean": "안녕하세요 세계. 이것은 테스트입니다.",
    "russian_cyrillic": "Здравствуй, мир! Это тест передачи текста по голосовому каналу.",
    "arabic_rtl": "مرحبا بالعالم، هذا اختبار",
    "emoji_only": "😀🎉🚀💻📞🔥✅",
    "mixed_script_emoji": "Mixed: 中文 + English + 123 + 🎉🚀 + Русский",
    "long_paragraph": (
        "This is a longer test message intended to stress the frame and FEC "
        "layers with more symbols than a short sentence would produce. It "
        "mixes ordinary punctuation, numbers like 42 and 2026, and enough "
        "length to matter for Reed-Solomon's correction budget once real "
        "AMR-NB compression is applied at the low end of its bitrate range."
    ),
    "newlines_and_tabs": "line one\nline two\ttabbed",
}

BITRATES = ["4.75k", "7.40k", "12.2k"]
PARITY_BYTES = 10


def run_case(text: str, bitrate: str, tmpdir: Path) -> tuple[bool, str]:
    if text == "":
        # build_frame on empty text still produces a valid (tiny) frame;
        # exercise it rather than special-casing it out of the matrix.
        pass
    frame_codes = build_frame(text, PARITY_BYTES)
    symbols = modem.bytes_to_symbols(bytes(frame_codes))
    audio_in = modem.modulate(symbols)

    audio_out = roundtrip_amr(audio_in, bitrate, tmpdir)

    n_symbols = len(symbols)  # known here; a live receiver would need frame sync instead
    dets = modem.demodulate(audio_out, n_symbols)
    decoded_symbols = [d.symbol for d in dets] + [0] * (n_symbols - len(dets))
    n_bytes = (len(decoded_symbols) * modem.BITS_PER_SYMBOL) // 8
    recv_codes = list(modem.symbols_to_bytes(decoded_symbols, n_bytes))

    result = parse_frame(recv_codes, PARITY_BYTES)
    if not result.ok:
        return False, f"FAIL ({result.reason})"
    if result.text != text:
        return False, f"MISMATCH (got {result.text!r})"
    return True, "OK"


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        results = []
        for name, text in SAMPLES.items():
            for bitrate in BITRATES:
                ok, detail = run_case(text, bitrate, tmpdir)
                results.append((name, bitrate, ok, detail))
                status = "PASS" if ok else "FAIL"
                print(f"[{status}] {name:20s} {bitrate:6s} {detail}", file=sys.stderr)

    total = len(results)
    passed = sum(1 for *_, ok, _ in results if ok)
    print(f"\n{passed}/{total} passed", file=sys.stderr)

    print("\n=== Failures by bitrate ===", file=sys.stderr)
    for bitrate in BITRATES:
        fails = [name for name, br, ok, _ in results if br == bitrate and not ok]
        print(f"  {bitrate}: {len(fails)} failed" + (f" ({', '.join(fails)})" if fails else ""),
              file=sys.stderr)


if __name__ == "__main__":
    main()
