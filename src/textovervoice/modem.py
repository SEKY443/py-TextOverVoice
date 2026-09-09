"""
2-of-16 dual-tone MFSK modem: the physical-layer symbol encoder/decoder.

One symbol = exactly one tone from LOW_GROUP + one tone from HIGH_GROUP,
giving 8 x 8 = 64 combinations = 6 bits/symbol. This replaces an earlier
8-simultaneous-tone FDM design, which measurably failed against AMR-NB:
real ffmpeg/libopencore_amrnb round-trips showed 8-tone chords producing
more intermodulation energy than signal at AMR's low bitrate, while
dual-tone pairs stayed well clear of that failure mode.

Frequencies were chosen from a single-tone sweep: all 8 candidates in each
group individually survive AMR-NB at both 4.75k and 12.2k with <6dB loss
and <30Hz drift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 8000  # AMR-NB operates at 8kHz mono; keep the whole pipeline native to that rate

LOW_GROUP = [400, 550, 700, 850, 1000, 1150, 1300, 1450]
HIGH_GROUP = [1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200]
BITS_PER_SYMBOL = 6  # log2(8*8)

# Symbol duration needs to be at least ~40ms (2+ AMR frames) -- measured
# data showed a 20ms symbol still had 28% error rate at AMR's low bitrate
# despite matching the codec's 20ms frame grid; 40ms+ is needed regardless
# of exact grid alignment.
DEFAULT_SYMBOL_DURATION_S = 0.04
DEFAULT_GUARD_S = 0.01
RAMP_FRACTION = 0.15  # raised-cosine edge, as a fraction of symbol duration


def symbol_to_freqs(symbol: int) -> tuple[float, float]:
    if not 0 <= symbol < 64:
        raise ValueError(f"symbol {symbol} out of range 0-63")
    low_idx, high_idx = divmod(symbol, 8)
    return LOW_GROUP[low_idx], HIGH_GROUP[high_idx]


def freqs_to_symbol(low_idx: int, high_idx: int) -> int:
    return low_idx * 8 + high_idx


def synth_symbol(symbol: int, duration_s: float, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    low_f, high_f = symbol_to_freqs(symbol)
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    x = amplitude * 0.5 * (np.sin(2 * np.pi * low_f * t) + np.sin(2 * np.pi * high_f * t))

    ramp_n = max(1, int(n * RAMP_FRACTION))
    window = np.ones(n)
    window[:ramp_n] = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_n) / ramp_n))
    window[-ramp_n:] = window[:ramp_n][::-1]
    return x * window


def modulate(symbols: list[int], symbol_duration_s: float = DEFAULT_SYMBOL_DURATION_S,
             guard_s: float = DEFAULT_GUARD_S, sr: int = SR) -> np.ndarray:
    guard = np.zeros(int(guard_s * sr))
    parts = []
    for s in symbols:
        parts.append(synth_symbol(s, symbol_duration_s, sr))
        parts.append(guard)
    return np.concatenate(parts) if parts else np.array([])


@dataclass
class SymbolDetection:
    symbol: int
    low_freq: float
    high_freq: float
    low_confidence_db: float  # margin of winning low-group bin over runner-up
    high_confidence_db: float


def detect_symbol(segment: np.ndarray, sr: int = SR) -> SymbolDetection:
    n = len(segment)
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(segment * window))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    bin_hz = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    search_bins = max(int(40 / bin_hz), 1)

    def group_scores(group: list[float]) -> list[float]:
        scores = []
        for f0 in group:
            center = int(f0 / bin_hz)
            lo, hi = max(0, center - search_bins), min(len(spec), center + search_bins + 1)
            scores.append(float(spec[lo:hi].max()) if hi > lo else 0.0)
        return scores

    low_scores = group_scores(LOW_GROUP)
    high_scores = group_scores(HIGH_GROUP)

    low_idx = int(np.argmax(low_scores))
    high_idx = int(np.argmax(high_scores))

    def confidence_db(scores: list[float], winner_idx: int) -> float:
        sorted_scores = sorted(scores, reverse=True)
        top = sorted_scores[0]
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 1e-9
        return 20 * np.log10((top + 1e-12) / (runner_up + 1e-12)) if scores[winner_idx] == top else -99.0

    return SymbolDetection(
        symbol=freqs_to_symbol(low_idx, high_idx),
        low_freq=LOW_GROUP[low_idx],
        high_freq=HIGH_GROUP[high_idx],
        low_confidence_db=confidence_db(low_scores, low_idx),
        high_confidence_db=confidence_db(high_scores, high_idx),
    )


def demodulate(audio: np.ndarray, n_symbols: int, symbol_duration_s: float = DEFAULT_SYMBOL_DURATION_S,
               guard_s: float = DEFAULT_GUARD_S, sr: int = SR,
               start_offset: int = 0) -> list[SymbolDetection]:
    """Fixed-timing slicer: assumes the caller already knows where symbol 0
    starts (start_offset, in samples). A real deployment needs a preamble
    to find this offset on a live capture; that layer isn't implemented
    yet -- this function tests modem accuracy in isolation.
    """
    symbol_n = int(symbol_duration_s * sr)
    guard_n = int(guard_s * sr)
    step = symbol_n + guard_n

    detections = []
    for i in range(n_symbols):
        start = start_offset + i * step
        end = start + symbol_n
        if end > len(audio):
            break
        detections.append(detect_symbol(audio[start:end], sr))
    return detections


# --- Bit packing: 8-bit bytes <-> 6-bit symbols -----------------------------
# Same ratio as base64 (3 bytes = 24 bits = 4 six-bit groups), but we need
# raw symbol indices (0-63), not a text alphabet, so a direct bit-packer is
# used rather than repurposing base64's character mapping.

def bytes_to_symbols(data: bytes) -> list[int]:
    bits = "".join(f"{b:08b}" for b in data)
    pad = (-len(bits)) % BITS_PER_SYMBOL
    bits += "0" * pad
    return [int(bits[i:i + BITS_PER_SYMBOL], 2) for i in range(0, len(bits), BITS_PER_SYMBOL)]


def symbols_to_bytes(symbols: list[int], n_bytes: int) -> bytes:
    bits = "".join(f"{s:0{BITS_PER_SYMBOL}b}" for s in symbols)
    n_bits = n_bytes * 8
    bits = bits[:n_bits]
    return bytes(int(bits[i:i + 8], 2) for i in range(0, n_bits, 8))
