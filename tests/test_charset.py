import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textovervoice.charset import decode_codes, encode_text


def roundtrip(text: str) -> str:
    return decode_codes(encode_text(text))


def test_ascii_roundtrip():
    assert roundtrip("Hello, World! 123") == "Hello, World! 123"


def test_ascii_bytes_are_not_escaped():
    codes_out = encode_text("A")
    assert codes_out == [ord("A")]


def test_cjk_roundtrip():
    assert roundtrip("文字轉聲音穿越電話通道") == "文字轉聲音穿越電話通道"


def test_emoji_4byte_utf8_roundtrip():
    assert roundtrip("hi \U0001F600 bye") == "hi \U0001F600 bye"


def test_mixed_ascii_and_multibyte():
    text = "abc文字123\U0001F600xyz"
    assert roundtrip(text) == text


def test_lost_end_flag_still_recovers_fully():
    """If UTF8_END itself is dropped, completion still happens via the
    lead byte's self-describing length (see _utf8_expected_len) -- so the
    message recovers exactly, not just partially."""
    from textovervoice import codes as C

    code_stream = encode_text("a文b")
    end_idx = code_stream.index(C.UTF8_END)
    corrupted = code_stream[:end_idx] + code_stream[end_idx + 1:]
    assert decode_codes(corrupted) == "a文b"


def test_lost_continuation_byte_drops_only_that_char():
    """If a data byte (not a flag) is lost, the character can't be
    reconstructed -- but the stream still resyncs at the next boundary
    instead of corrupting everything after it."""
    from textovervoice import codes as C

    code_stream = encode_text("a文b")
    start_idx = code_stream.index(C.UTF8_START)
    # Drop the byte immediately after START (the first UTF-8 byte of 文).
    corrupted = code_stream[:start_idx + 1] + code_stream[start_idx + 2:]
    result = decode_codes(corrupted)
    assert result.startswith("a")
    assert result.endswith("b")
    assert "文" not in result
