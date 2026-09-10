import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textovervoice import message
from textovervoice.protocol import parse_frame


def _decode_all(frames: list[list[int]], **parse_kwargs) -> str:
    reassembler = message.MessageReassembler()
    result = None
    for f in frames:
        result = reassembler.add(parse_frame(f, **parse_kwargs))
    assert result is not None and result.ok, f"reassembly failed: {result}"
    return result.text


def test_short_message_is_a_single_frame():
    frames = message.build_message("hello")
    assert len(frames) == 1
    result = parse_frame(frames[0])
    assert result.ok
    assert result.seq == 0
    assert result.more_frames is False
    assert result.text == "hello"


def test_long_message_splits_into_multiple_frames():
    text = "x" * 2500
    frames = message.build_message(text, max_frame_chars=800)
    assert len(frames) == 4  # 800+800+800+100

    results = [parse_frame(f) for f in frames]
    assert [r.seq for r in results] == [0, 1, 2, 3]
    assert [r.more_frames for r in results] == [True, True, True, False]


def test_multi_frame_roundtrip_reassembles_exactly():
    text = "The quick brown fox. " * 100  # well over 800 chars
    frames = message.build_message(text, max_frame_chars=800)
    assert len(frames) > 1
    assert _decode_all(frames) == text


def test_multi_frame_roundtrip_with_dictionary_and_cjk():
    text = ("This message repeats government information about something "
            "important. 文字轉聲音穿越語音通話測試。") * 20
    frames = message.build_message(text, max_frame_chars=500)
    assert len(frames) > 1
    assert _decode_all(frames) == text


def test_reassembler_handles_out_of_order_frames():
    text = "AAAA" * 500  # multiple frames
    frames = message.build_message(text, max_frame_chars=800)
    assert len(frames) >= 3

    reassembler = message.MessageReassembler()
    order = [1, 0, 2] + list(range(3, len(frames)))  # shuffle the first few
    result = None
    for i in order:
        result = reassembler.add(parse_frame(frames[i]))
    assert result is not None and result.ok
    assert result.text == text


def test_reassembler_reports_incomplete_before_last_frame_arrives():
    text = "y" * 2000
    frames = message.build_message(text, max_frame_chars=800)
    assert len(frames) == 3

    reassembler = message.MessageReassembler()
    r0 = reassembler.add(parse_frame(frames[0]))
    assert not r0.ok
    assert r0.frames_received == 1
    assert r0.frames_expected is None  # last frame (which reveals total count) not seen yet

    r1 = reassembler.add(parse_frame(frames[1]))
    assert not r1.ok

    r2 = reassembler.add(parse_frame(frames[2]))
    assert r2.ok
    assert r2.text == text
    assert r2.frames_received == 3
    assert r2.frames_expected == 3


def test_reassembler_reset_starts_fresh():
    reassembler = message.MessageReassembler()
    frames1 = message.build_message("first message " * 100, max_frame_chars=800)
    for f in frames1[:-1]:
        reassembler.add(parse_frame(f))  # leave incomplete

    reassembler.reset()
    frames2 = message.build_message("second, unrelated", max_frame_chars=800)
    result = reassembler.add(parse_frame(frames2[0]))
    assert result.ok
    assert result.text == "second, unrelated"


def test_multi_frame_with_encryption_and_addressing():
    from textovervoice import crypto

    alice_priv, alice_pub = crypto.generate_keypair()
    bob_priv, bob_pub = crypto.generate_keypair()
    alice_key = crypto.derive_session_key(alice_priv, bob_pub)
    bob_key = crypto.derive_session_key(bob_priv, alice_pub)

    text = "Secret plan " * 200
    frames = message.build_message(text, max_frame_chars=800, dest_id=7, session_key=alice_key)
    assert len(frames) > 1

    result = _decode_all(frames, my_id=7, session_key=bob_key)
    assert result == text


def test_empty_message_produces_one_empty_frame():
    frames = message.build_message("")
    assert len(frames) == 1
    result = parse_frame(frames[0])
    assert result.ok
    assert result.text == ""
    assert result.more_frames is False
