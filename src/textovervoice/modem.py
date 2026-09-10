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
from scipy.signal import chirp as _scipy_chirp
from scipy.signal import fftconvolve as _fftconvolve

SR = 8000  # AMR-NB operates at 8kHz mono; keep the whole pipeline native to that rate

LOW_GROUP = [400, 550, 700, 850, 1000, 1150, 1300, 1450]
HIGH_GROUP = [1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200]
BITS_PER_SYMBOL = 6  # log2(8*8)

# Named timing profiles. "phone" is validated through real AMR-NB (see
# tools/codec_validation/realism_results.csv); "fast_air" is validated only
# acoustically (real speaker/mic, no codec) -- it is NOT expected to survive
# AMR compression, since the AMR frame-boundary constraint that set
# "phone"'s 40ms floor doesn't apply to fast_air's shorter timing at all.
#
# fast_air's duration was picked from a real-hardware sweep (see
# tools/codec_validation/acoustic_duration_sweep.py), not guessed: 6 trials
# split between 20ms and 40ms measured comparable symbol-error rates (mean
# 1.0% vs 3.1%, both within the run's noisy baseline -- 20ms showed *no*
# measurable degradation from halving the duration). 15ms and below entered
# a clearly worse regime (12.5%+ SER), and below 10ms was catastrophic
# (50%+), consistent with 150Hz-spaced tones needing enough samples for the
# FFT to resolve adjacent bins. 20ms doubles fast_air's raw bitrate over
# phone mode (240bps vs 120bps) with real hardware evidence behind it.
MODES = {
    "phone": {"symbol_duration_s": 0.04, "guard_s": 0.01},
    "fast_air": {"symbol_duration_s": 0.02, "guard_s": 0.005},
}

DEFAULT_SYMBOL_DURATION_S = MODES["phone"]["symbol_duration_s"]
DEFAULT_GUARD_S = MODES["phone"]["guard_s"]
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


# --- Frame sync: locate symbol 0 in a live capture with no known alignment --
# A linear chirp cross-correlates sharply against noise/tones (much sharper
# peak than a single fixed tone would give), which is why it's the standard
# choice for this in real modems. Prototyped and verified in
# tools/codec_validation/acoustic_loopback_test.py (3/3 real speaker/mic
# trials, confidence 174-242x noise floor) before being promoted here as a
# first-class part of the modem rather than a one-off test script.

PREAMBLE_DURATION_S = 0.25
PREAMBLE_F0, PREAMBLE_F1 = 800, 3000  # stays inside the 300-3400Hz voice band,
                                       # so it works for "phone" mode too
PREAMBLE_GUARD_S = 0.05  # silence between preamble and payload


def generate_preamble(duration_s: float = PREAMBLE_DURATION_S, f0: float = PREAMBLE_F0,
                       f1: float = PREAMBLE_F1, sr: int = SR) -> np.ndarray:
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    sig = _scipy_chirp(t, f0=f0, f1=f1, t1=duration_s, method="linear")

    ramp_n = int(0.1 * n)
    window = np.ones(n)
    window[:ramp_n] = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_n) / ramp_n))
    window[-ramp_n:] = window[:ramp_n][::-1]
    return (0.6 * sig * window).astype(np.float64)


def find_preamble(audio: np.ndarray, reference: np.ndarray | None = None,
                   min_score: float = 0.4) -> tuple[int, float] | None:
    """Cross-correlates `audio` against the known preamble chirp, using a
    normalized cross-correlation coefficient (numerator / sqrt(local window
    energy * reference energy)) rather than a raw correlation peak. This
    matters: an earlier version compared the peak against the correlation's
    own median as a "confidence" score, which false-triggered on pure noise
    (caught by tests/test_modem_sync.py) -- with many correlation lags, the
    max of an unrelated-noise correlation can spike several times above the
    median from ordinary extreme-value statistics, even though nothing
    actually matched. Normalizing by local energy bounds the score to
    roughly [-1, 1] with a stable, physically meaningful threshold instead.

    Returns (sample_index_after_preamble, score), or None if no lag's score
    clears min_score, i.e. sync failed.

    Uses FFT-based correlation (scipy.signal.fftconvolve, O(N log N)) and a
    cumulative-sum sliding-window energy (O(N)), not np.correlate/np.convolve's
    direct O(N*M) computation -- measured directly: searching a realistic
    live-polling window (180s at 48kHz) with the direct method took 40+
    seconds per call, making real-time listening (modem.py is polled every
    ~0.3s -- see live.py) effectively non-functional. This is the same
    numerical result, just computed efficiently."""
    if reference is None:
        reference = generate_preamble()
    n = len(reference)
    if len(audio) < n:
        return None

    numerator = _fftconvolve(audio, reference[::-1], mode="valid")
    cumsum = np.concatenate(([0.0], np.cumsum(audio ** 2)))
    window_energy = cumsum[n:] - cumsum[:-n]
    ref_energy = np.sum(reference ** 2)
    denom = np.sqrt(window_energy * ref_energy) + 1e-12
    score = numerator / denom

    peak_idx = int(np.argmax(np.abs(score)))
    peak_score = float(np.abs(score[peak_idx]))

    if peak_score < min_score:
        return None
    return peak_idx + n, peak_score


def modulate_frame(symbols: list[int], symbol_duration_s: float = DEFAULT_SYMBOL_DURATION_S,
                    guard_s: float = DEFAULT_GUARD_S, sr: int = SR) -> np.ndarray:
    """modulate() with a preamble prepended, for transmission into an
    unknown-alignment channel (a live capture, not a file where symbol 0 is
    known to start at sample 0)."""
    preamble = generate_preamble(sr=sr)
    preamble_guard = np.zeros(int(PREAMBLE_GUARD_S * sr))
    payload = modulate(symbols, symbol_duration_s, guard_s, sr)
    return np.concatenate([preamble, preamble_guard, payload])


def demodulate_frame(audio: np.ndarray, n_symbols: int,
                      symbol_duration_s: float = DEFAULT_SYMBOL_DURATION_S,
                      guard_s: float = DEFAULT_GUARD_S, sr: int = SR,
                      min_score: float = 0.4) -> tuple[list[SymbolDetection], float] | None:
    """Finds the preamble, then demodulates the payload that follows it.
    Returns (detections, sync_score), or None if the preamble wasn't found
    reliably (caller should treat that as "no frame here yet", e.g. keep
    listening, rather than a decode failure)."""
    found = find_preamble(audio, min_score=min_score)
    if found is None:
        return None
    offset, score = found
    payload_start = offset + int(PREAMBLE_GUARD_S * sr)
    detections = demodulate(audio[payload_start:], n_symbols, symbol_duration_s, guard_s, sr)
    return detections, score


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
