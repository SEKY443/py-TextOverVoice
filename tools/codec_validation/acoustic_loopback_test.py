#!/usr/bin/env python3
"""
Real acoustic loopback test: plays the modulated frame through the device
speaker and simultaneously records it back through the microphone, then
decodes what was actually captured.

This is a materially different (and harder) test than the file-based
AMR-NB round-trips in amr_harness.py/realism_suite.py -- those simulate
codec compression only. This one adds everything a real over-the-air
handset use would add: room acoustics/reverb, speaker and mic frequency
response, ambient noise, and critically, *no known sample-0 alignment* --
the recording starts at an arbitrary time relative to playback, with
device scheduling latency on top. Frame sync (modem.modulate_frame/
demodulate_frame, a chirp cross-correlated via normalized cross-correlation)
handles that; this script was originally where that sync logic was
prototyped before being promoted into modem.py as a first-class feature.

--mode selects modem.MODES: "phone" (default, AMR-validated) or "fast_air"
(faster, acoustic-only -- see modem.MODES's docstring for the real-hardware
duration sweep behind those numbers).

Requires a working microphone + speaker and OS mic permission granted to
whatever terminal/app is running this.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from textovervoice import modem  # noqa: E402
from textovervoice.protocol import build_frame, parse_frame  # noqa: E402

DEVICE_SR = 48000  # universally supported; modem.SR (8000) is resampled to/from this
PRE_SILENCE_S = 0.3  # lead-in so playback/device startup transients don't clip the preamble
POST_SILENCE_S = 0.8  # generous margin for device scheduling latency


def run(text: str, parity_bytes: int, mode: str) -> None:
    profile = modem.MODES[mode]
    symbol_duration, guard = profile["symbol_duration_s"], profile["guard_s"]

    frame_codes = build_frame(text, parity_bytes)
    symbols = modem.bytes_to_symbols(bytes(frame_codes))
    tx_8k = modem.modulate_frame(symbols, symbol_duration_s=symbol_duration, guard_s=guard)
    tx_8k = np.concatenate([np.zeros(int(PRE_SILENCE_S * modem.SR)), tx_8k,
                             np.zeros(int(POST_SILENCE_S * modem.SR))])
    tx_device = resample_poly(tx_8k, DEVICE_SR, modem.SR).astype(np.float32)

    print(f"[{mode}] Playing {len(tx_device)/DEVICE_SR:.2f}s of audio "
          f"({len(symbols)} symbols, {len(frame_codes)} frame codes) "
          f"-- speak nothing, let it play...", file=sys.stderr)

    rec_device = sd.playrec(tx_device.reshape(-1, 1), samplerate=DEVICE_SR,
                             channels=1, dtype="float32")
    sd.wait()
    rec_8k = resample_poly(rec_device[:, 0].astype(np.float64), modem.SR, DEVICE_SR)

    result = modem.demodulate_frame(rec_8k, len(symbols), symbol_duration_s=symbol_duration, guard_s=guard)
    if result is None:
        print("DECODE FAILED: preamble not found (no frame sync) -- check volume "
              "levels, mic permission, ambient noise", file=sys.stderr)
        sys.exit(1)
    detections, sync_score = result
    print(f"Frame sync score: {sync_score:.2f}", file=sys.stderr)
    if sync_score < 0.5:
        print("WARNING: low sync score -- decode below may be unreliable", file=sys.stderr)

    decoded_symbols = [d.symbol for d in detections] + [0] * (len(symbols) - len(detections))
    n_bytes = (len(decoded_symbols) * modem.BITS_PER_SYMBOL) // 8
    recv_codes = list(modem.symbols_to_bytes(decoded_symbols, n_bytes))

    result = parse_frame(recv_codes, parity_bytes)
    if not result.ok:
        print(f"DECODE FAILED: {result.reason}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDecoded: {result.text!r}", file=sys.stderr)
    print("MATCH" if result.text == text else "MISMATCH (survived FEC but wrong content)",
          file=sys.stderr)
    print(result.text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="Hello TextOverVoice 你好",
                         help="text to transmit acoustically")
    parser.add_argument("--parity-bytes", type=int, default=10)
    parser.add_argument("--mode", choices=list(modem.MODES), default="phone")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    run(args.text, args.parity_bytes, args.mode)


if __name__ == "__main__":
    main()
