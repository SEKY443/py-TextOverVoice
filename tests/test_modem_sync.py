import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textovervoice import modem


def test_find_preamble_locates_known_offset():
    rng = np.random.default_rng(0)
    pre_silence = rng.normal(0, 0.01, 4000)
    preamble = modem.generate_preamble()
    post = rng.normal(0, 0.01, 2000)
    audio = np.concatenate([pre_silence, preamble, post])

    found = modem.find_preamble(audio)
    assert found is not None
    offset, score = found
    assert offset == len(pre_silence) + len(preamble)
    assert score > 0.9  # near-exact match against a near-noiseless embed


def test_find_preamble_returns_none_on_noise_only():
    """Regression test: an earlier confidence metric (peak vs. local median
    of the correlation) false-triggered on pure noise, since with many
    correlation lags the max of unrelated-noise correlation naturally spikes
    above the median from ordinary extreme-value statistics. Runs several
    noise seeds and a longer buffer to make a false positive here meaningful,
    not a coin flip."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.05, 80000)
        assert modem.find_preamble(noise) is None, f"false positive on noise seed {seed}"


def test_find_preamble_returns_none_on_silence():
    assert modem.find_preamble(np.zeros(20000)) is None


def test_modulate_demodulate_frame_roundtrip_no_channel_impairment():
    symbols = [0, 15, 32, 47, 63, 7, 21]
    audio = modem.modulate_frame(symbols)
    result = modem.demodulate_frame(audio, len(symbols))
    assert result is not None
    detections, score = result
    assert [d.symbol for d in detections] == symbols
    assert score > 0.9


def test_demodulate_frame_works_with_arbitrary_leading_silence():
    """This is the actual point of frame sync: symbol 0 does NOT need to
    start at sample 0, unlike modem.demodulate's fixed-timing assumption."""
    symbols = [3, 9, 40]
    padded = np.concatenate([np.zeros(7777), modem.modulate_frame(symbols)])
    result = modem.demodulate_frame(padded, len(symbols))
    assert result is not None
    detections, _ = result
    assert [d.symbol for d in detections] == symbols


def test_modes_are_registered_and_phone_matches_legacy_defaults():
    assert "phone" in modem.MODES
    assert "fast_air" in modem.MODES
    assert modem.MODES["phone"]["symbol_duration_s"] == modem.DEFAULT_SYMBOL_DURATION_S
    assert modem.MODES["phone"]["guard_s"] == modem.DEFAULT_GUARD_S
    # fast_air must actually be faster, or it's not earning its name
    assert modem.MODES["fast_air"]["symbol_duration_s"] < modem.MODES["phone"]["symbol_duration_s"]
