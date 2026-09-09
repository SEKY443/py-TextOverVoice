import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textovervoice.protocol import build_frame, parse_frame


def test_clean_roundtrip_ascii():
    frame = build_frame("Hello, World!")
    result = parse_frame(frame)
    assert result.ok
    assert result.text == "Hello, World!"


def test_clean_roundtrip_cjk_and_emoji():
    text = "文字轉聲音穿越電話通道 \U0001F600"
    frame = build_frame(text)
    result = parse_frame(frame)
    assert result.ok
    assert result.text == text


def test_survives_corruption_within_fec_budget():
    text = "The quick brown fox jumps over the lazy dog"
    frame = build_frame(text, parity_bytes=10)
    corrupted = list(frame)
    # Flip a few payload bytes (well within the 5-byte-error budget for
    # 10 parity bytes) -- avoid indices 0-2 (SOF/LENGTH) so the frame
    # structure itself stays parseable.
    for i in (5, 12, 20):
        if i < len(corrupted):
            corrupted[i] ^= 0x01
    result = parse_frame(corrupted, parity_bytes=10)
    assert result.ok
    assert result.text == text


def test_detects_uncorrectable_corruption_instead_of_returning_garbage():
    text = "The quick brown fox jumps over the lazy dog"
    frame = build_frame(text, parity_bytes=10)
    corrupted = list(frame)
    for i in range(4, 40, 2):  # heavy corruption, over the FEC budget
        if i < len(corrupted):
            corrupted[i] ^= 0xFF
    result = parse_frame(corrupted, parity_bytes=10)
    # Must not silently hand back wrong text as if it were correct.
    assert not result.ok or result.text == text


def test_missing_parity_start_marker_reported_not_crashed():
    frame = build_frame("hi")
    truncated = frame[:5]
    result = parse_frame(truncated)
    assert not result.ok
    assert result.text is None
