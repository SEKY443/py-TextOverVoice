#!/usr/bin/env python3
"""
Tests the existing tone plan / modem / protocol against Opus (WhatsApp/
Discord-style VoIP), not just AMR-NB (cellular). See opus_harness.py's
docstring for what these profiles approximate and what they don't cover
(no WebRTC-style noise suppression/AEC simulation).

Mirrors realism_suite.py's structure: full-alphabet sequence accuracy, plus
full end-to-end protocol (build_frame -> modulate -> Opus -> demodulate ->
parse_frame) with real text.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from textovervoice import modem  # noqa: E402
from textovervoice.protocol import build_frame, parse_frame  # noqa: E402
from opus_harness import roundtrip_opus, OPUS_PROFILES  # noqa: E402


def test_full_alphabet(tmpdir: Path) -> None:
    print("[1/2] Full-alphabet sequence (64 symbols) through Opus", file=sys.stderr)
    symbols = list(range(64))
    for i in range(len(symbols)):  # deterministic shuffle, not random -- reproducible
        j = (i * 17 + 5) % len(symbols)
        symbols[i], symbols[j] = symbols[j], symbols[i]

    for profile in OPUS_PROFILES:
        audio_in = modem.modulate(symbols)
        audio_out = roundtrip_opus(audio_in, profile, tmpdir, modem.SR)
        dets = modem.demodulate(audio_out, len(symbols))
        errors = sum(1 for d, s in zip(dets, symbols) if d.symbol != s)
        ser = errors / len(symbols)
        target_sr, bitrate = OPUS_PROFILES[profile]
        print(f"  {profile:20s} ({target_sr}Hz, {bitrate}): SER={ser:.1%} ({errors}/{len(symbols)})",
              file=sys.stderr)


def test_full_protocol(tmpdir: Path) -> None:
    print("\n[2/2] Full protocol (frame -> modem -> Opus -> demodulate -> parse)", file=sys.stderr)
    samples = {
        "ascii": "Hello TextOverVoice, testing 1234567890!",
        "cjk": "文字轉聲音穿越語音通話測試，準確率至關重要。",
    }
    for label, text in samples.items():
        frame_codes = build_frame(text, parity_bytes=10)
        symbols = modem.bytes_to_symbols(bytes(frame_codes))
        for profile in OPUS_PROFILES:
            audio_in = modem.modulate_frame(symbols)
            audio_out = roundtrip_opus(audio_in, profile, tmpdir, modem.SR)
            result = modem.demodulate_frame(audio_out, len(symbols))
            if result is None:
                print(f"  [{label:5s}] {profile:20s} SYNC FAILED", file=sys.stderr)
                continue
            detections, sync_score = result
            decoded_symbols = [d.symbol for d in detections] + [0] * (len(symbols) - len(detections))
            n_bytes = (len(decoded_symbols) * modem.BITS_PER_SYMBOL) // 8
            recv_codes = list(modem.symbols_to_bytes(decoded_symbols, n_bytes))
            result = parse_frame(recv_codes, parity_bytes=10)
            status = "OK " if (result.ok and result.text == text) else "FAIL"
            detail = result.text if result.ok else result.reason
            print(f"  [{status}] {label:5s} {profile:20s} sync={sync_score:.2f}  {detail!r}",
                  file=sys.stderr)


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        test_full_alphabet(tmpdir)
        test_full_protocol(tmpdir)


if __name__ == "__main__":
    main()
