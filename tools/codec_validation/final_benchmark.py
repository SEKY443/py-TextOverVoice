#!/usr/bin/env python3
"""
Final throughput/overhead benchmark, covering everything added this session
(dictionary compression, fast_air mode, addressing, encryption), measured
against real ggwave for a final apples-to-apples comparison.

Not a reliability test (see realism_suite.py / opus_survivability.py /
acoustic_loopback_test.py for those) -- this measures symbol/time cost per
configuration using the real modem/protocol code, i.e. what actually gets
transmitted, not estimates.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from textovervoice import crypto, modem  # noqa: E402
from textovervoice.protocol import build_frame  # noqa: E402

SAMPLE_SHORT = "Hello TextOverVoice"
SAMPLE_MEDIUM = "The government should provide information about something important."
SAMPLE_LONG = (
    "This is a longer test message intended to stress the frame and FEC "
    "layers with more symbols than a short sentence would produce. It "
    "mixes ordinary punctuation, numbers like 42 and 2026, and enough "
    "length to matter for Reed-Solomon's correction budget."
)


def rate(text: str, mode: str = "phone", **build_kwargs) -> tuple[int, float, float]:
    frame_codes = build_frame(text, **build_kwargs)
    symbols = modem.bytes_to_symbols(bytes(frame_codes))
    profile = modem.MODES[mode]
    duration_s = len(symbols) * (profile["symbol_duration_s"] + profile["guard_s"]) + modem.PREAMBLE_DURATION_S
    return len(frame_codes), duration_s, len(text) / duration_s


def print_row(label: str, text: str, mode: str = "phone", **build_kwargs) -> None:
    n_codes, duration_s, cps = rate(text, mode=mode, **build_kwargs)
    print(f"  {label:38s} {len(text):4d} chars -> {n_codes:4d} codes  "
          f"{duration_s:6.2f}s  {cps:6.2f} chars/sec")


def main() -> None:
    print("=== TextOverVoice throughput by configuration ===\n", file=sys.stderr)

    alice_priv, alice_pub = crypto.generate_keypair()
    bob_priv, bob_pub = crypto.generate_keypair()
    session_key = crypto.derive_session_key(alice_priv, bob_pub)

    for label, text in [("short", SAMPLE_SHORT), ("medium", SAMPLE_MEDIUM), ("long", SAMPLE_LONG)]:
        print(f"[{label}] \"{text[:50]}{'...' if len(text) > 50 else ''}\"", file=sys.stderr)
        print_row("phone, plain+dictionary", text, mode="phone")
        print_row("phone, plain, no dictionary", text, mode="phone", use_dictionary=False)
        print_row("phone, addressed (dest_id)", text, mode="phone", dest_id=7)
        print_row("phone, encrypted", text, mode="phone", session_key=session_key)
        print_row("fast_air, plain+dictionary", text, mode="fast_air")
        print_row("fast_air, plain, no dictionary", text, mode="fast_air", use_dictionary=False)
        print_row("fast_air, encrypted", text, mode="fast_air", session_key=session_key)
        print(file=sys.stderr)

    print("=== vs. ggwave (measured separately with the ggwave library) ===", file=sys.stderr)
    print("  ggwave AUDIBLE_NORMAL                  9.95 chars/sec  (fails through AMR-NB/8kHz entirely)",
          file=sys.stderr)
    print("  ggwave AUDIBLE_FAST                   14.36 chars/sec  (fails through AMR-NB/8kHz entirely)",
          file=sys.stderr)
    print("  ggwave AUDIBLE_FASTEST                25.81 chars/sec  (fails through AMR-NB/8kHz entirely)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
