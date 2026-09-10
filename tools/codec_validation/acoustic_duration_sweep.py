#!/usr/bin/env python3
"""
Finds how short a symbol duration actually survives real speaker->air->mic
acoustic transmission (no AMR/codec compression at all), to pick real
numbers for modem.MODES["fast_air"] instead of guessing.

"phone" mode's 40ms floor exists because of AMR's 20ms codec frame grid --
that constraint doesn't exist here, so this sweeps down from there to find
where the *acoustic* channel itself (speaker/mic frequency response, room
reflections, FFT frequency resolution against the 150Hz-spaced tone grid)
starts to break, which is a different and likely much lower floor.

Uses the real modem.modulate_frame/demodulate_frame (preamble-based sync),
not fixed-timing -- this is also a real-hardware regression check for that
sync code, not just a duration sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from textovervoice import modem  # noqa: E402

DEVICE_SR = 48000
DURATIONS_MS = [40, 30, 20, 15, 12, 10, 8, 6]
N_SYMBOLS = 32
GUARD_RATIO = 0.2  # guard_s = symbol_duration_s * this
PRE_SILENCE_S = 0.3  # lead-in so playback/device startup transients don't clip the preamble


def run_trial(symbol_duration_s: float, guard_s: float, symbols: list[int]) -> dict:
    tx_8k = modem.modulate_frame(symbols, symbol_duration_s=symbol_duration_s, guard_s=guard_s)
    tx_8k = np.concatenate([np.zeros(int(PRE_SILENCE_S * modem.SR)), tx_8k,
                             np.zeros(int(0.5 * modem.SR))])
    tx_device = resample_poly(tx_8k, DEVICE_SR, modem.SR).astype(np.float32)

    rec_device = sd.playrec(tx_device.reshape(-1, 1), samplerate=DEVICE_SR, channels=1, dtype="float32")
    sd.wait()
    rec_8k = resample_poly(rec_device[:, 0].astype(np.float64), modem.SR, DEVICE_SR)

    result = modem.demodulate_frame(rec_8k, len(symbols), symbol_duration_s=symbol_duration_s, guard_s=guard_s)
    if result is None:
        return {"sync": False, "score": None, "errors": None, "ser": None}

    detections, score = result
    decoded = [d.symbol for d in detections]
    errors = sum(1 for a, b in zip(decoded, symbols) if a != b) + (len(symbols) - len(decoded))
    return {"sync": True, "score": score, "errors": errors, "ser": errors / len(symbols)}


def main() -> None:
    rng_symbols = [(i * 7 + 3) % 64 for i in range(N_SYMBOLS)]  # deterministic, spread across the symbol space

    print(f"Testing durations: {DURATIONS_MS} ms, {N_SYMBOLS} symbols each, "
          f"guard = {GUARD_RATIO:.0%} of symbol duration", file=sys.stderr)
    print("(let it play without background noise/talking for a clean reading)\n", file=sys.stderr)

    results = []
    for ms in DURATIONS_MS:
        dur_s = ms / 1000
        guard_s = dur_s * GUARD_RATIO
        r = run_trial(dur_s, guard_s, rng_symbols)
        results.append((ms, r))
        if not r["sync"]:
            print(f"{ms:3d}ms: SYNC FAILED (preamble not found)", file=sys.stderr)
        else:
            print(f"{ms:3d}ms: sync_score={r['score']:.2f}  SER={r['ser']:.1%}  "
                  f"({r['errors']}/{N_SYMBOLS} symbol errors)", file=sys.stderr)

    print("\n=== Summary ===", file=sys.stderr)
    clean = [ms for ms, r in results if r["sync"] and r["ser"] == 0.0]
    if clean:
        fastest_clean = min(clean)
        print(f"Fastest duration with zero errors: {fastest_clean}ms", file=sys.stderr)
        print(f"Recommend fast_air symbol_duration_s = {fastest_clean/1000:.3f} "
              f"(or one step slower for margin)", file=sys.stderr)
    else:
        print("No duration achieved zero errors in this run -- keep phone-mode "
              "defaults for fast_air, or retest in quieter conditions", file=sys.stderr)


if __name__ == "__main__":
    main()
