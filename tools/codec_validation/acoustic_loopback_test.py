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
device scheduling latency on top.

The modem/protocol code has no preamble-based frame sync yet (documented
open item -- modem.demodulate's docstring, protocol.py's module docstring).
This script prototypes the minimum viable version of that: a linear chirp
prepended before the actual frame, located in the recording via
cross-correlation, to establish where symbol 0 actually starts. This is a
test-support utility, not a finished protocol feature -- integrating real
preamble sync into the wire format is separate, larger work.

Requires a working microphone + speaker and OS mic permission granted to
whatever terminal/app is running this.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import chirp as scipy_chirp
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from textovervoice import modem  # noqa: E402
from textovervoice.protocol import build_frame, parse_frame  # noqa: E402

DEVICE_SR = 48000  # universally supported; modem.SR (8000) is resampled to/from this
CHIRP_DURATION_S = 0.25
CHIRP_F0, CHIRP_F1 = 800, 3000  # stays inside the 300-3400Hz voice band
PRE_SILENCE_S = 0.3
POST_SILENCE_S = 0.8  # generous margin for device scheduling latency
GUARD_AFTER_CHIRP_S = 0.05


def make_sync_chirp(sr: int = modem.SR) -> np.ndarray:
    n = int(CHIRP_DURATION_S * sr)
    t = np.arange(n) / sr
    sig = scipy_chirp(t, f0=CHIRP_F0, f1=CHIRP_F1, t1=CHIRP_DURATION_S, method="linear")
    ramp_n = int(0.1 * n)
    window = np.ones(n)
    window[:ramp_n] = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_n) / ramp_n))
    window[-ramp_n:] = window[:ramp_n][::-1]
    return (0.6 * sig * window).astype(np.float64)


def find_chirp_offset(recorded: np.ndarray, reference_chirp: np.ndarray) -> tuple[int, float]:
    """Cross-correlate to find where the chirp starts in `recorded`.
    Returns (sample_offset, peak_confidence) -- confidence is the peak
    correlation normalized by the recording's overall energy, useful for
    telling a real detection apart from noise."""
    corr = np.correlate(recorded, reference_chirp, mode="valid")
    peak_idx = int(np.argmax(np.abs(corr)))
    peak_val = np.abs(corr[peak_idx])
    noise_floor = np.median(np.abs(corr)) + 1e-9
    confidence = peak_val / noise_floor
    return peak_idx, confidence


def run(text: str, parity_bytes: int, symbol_duration: float) -> None:
    frame_codes = build_frame(text, parity_bytes)
    symbols = modem.bytes_to_symbols(bytes(frame_codes))
    payload_audio = modem.modulate(symbols, symbol_duration_s=symbol_duration)

    sync_chirp = make_sync_chirp()
    guard = np.zeros(int(GUARD_AFTER_CHIRP_S * modem.SR))
    pre = np.zeros(int(PRE_SILENCE_S * modem.SR))
    post = np.zeros(int(POST_SILENCE_S * modem.SR))
    tx_audio_8k = np.concatenate([pre, sync_chirp, guard, payload_audio, post])

    tx_audio_device = resample_poly(tx_audio_8k, DEVICE_SR, modem.SR).astype(np.float32)

    print(f"Playing {len(tx_audio_device)/DEVICE_SR:.2f}s of audio "
          f"({len(symbols)} symbols, {len(frame_codes)} frame codes) "
          f"-- speak nothing, let it play...", file=sys.stderr)

    rec_device = sd.playrec(tx_audio_device.reshape(-1, 1), samplerate=DEVICE_SR,
                             channels=1, dtype="float32")
    sd.wait()
    rec_device = rec_device[:, 0].astype(np.float64)

    rec_8k = resample_poly(rec_device, modem.SR, DEVICE_SR)

    offset, confidence = find_chirp_offset(rec_8k, sync_chirp)
    print(f"Chirp detected at sample {offset} ({offset/modem.SR:.3f}s into recording), "
          f"confidence={confidence:.1f}x noise floor", file=sys.stderr)
    if confidence < 3.0:
        print("WARNING: low confidence -- chirp may not have been reliably detected "
              "(check volume levels, mic permission, ambient noise)", file=sys.stderr)

    payload_start = offset + len(sync_chirp) + len(guard)
    if payload_start >= len(rec_8k):
        print("DECODE FAILED: chirp detected too close to end of recording, "
              "no room for payload -- try increasing POST_SILENCE_S", file=sys.stderr)
        sys.exit(1)

    dets = modem.demodulate(rec_8k[payload_start:], len(symbols), symbol_duration_s=symbol_duration)
    decoded_symbols = [d.symbol for d in dets] + [0] * (len(symbols) - len(dets))
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
    parser.add_argument("--symbol-duration", type=float, default=0.06,
                         help="longer than the file-test default (0.04s) to give "
                              "some extra margin against room acoustics/echo")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    run(args.text, args.parity_bytes, args.symbol_duration)


if __name__ == "__main__":
    main()
