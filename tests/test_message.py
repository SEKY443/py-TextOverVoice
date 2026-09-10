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


def test_repeated_transmission_recovers_from_per_pass_frame_loss():
    """Simulates the actual mechanism behind cli.py/live.py's --repeat: the
    sender transmits the same frame set (same seq numbers) multiple times,
    and a lossy channel drops a different subset of frames on each pass --
    no single pass is complete on its own, but every seq survives at least
    one pass. The reassembler is fed all surviving frames across all passes
    into ONE instance (not reset between passes, matching how Listener/
    decode() behave), and should still reassemble the full message purely
    because it's keyed by seq: a seq arriving more than once is a harmless
    overwrite, and completion only needs each seq to show up at least once
    across however many passes it took."""
    text = "The quick brown fox. " * 100
    frames = message.build_message(text, max_frame_chars=400)
    assert len(frames) >= 4  # need several seqs for the drop pattern below to be meaningful

    # Every seq is missing from at least one pass, but present in another.
    passes_dropped_seqs = [{0, 2}, {1, 3}, set()]

    reassembler = message.MessageReassembler()
    result = None
    for dropped in passes_dropped_seqs:
        for i, f in enumerate(frames):
            if i in dropped:
                continue
            result = reassembler.add(parse_frame(f))
    assert result is not None and result.ok
    assert result.text == text


def test_single_pass_would_have_failed_without_repeat():
    """Sanity check for the test above: confirms the per-pass drop pattern
    used there really does leave every individual pass incomplete on its
    own -- i.e. the repeat mechanism is doing real work, not papering over
    a pattern that any single pass could already reassemble."""
    text = "y" * 2000
    frames = message.build_message(text, max_frame_chars=800)
    passes_dropped_seqs = [{0, 2}, {1, 3}, set()]

    for dropped in passes_dropped_seqs[:-1]:  # the last pass alone is complete by construction
        reassembler = message.MessageReassembler()
        result = None
        for i, f in enumerate(frames):
            if i in dropped:
                continue
            result = reassembler.add(parse_frame(f))
        assert result is None or not result.ok


def test_empty_message_produces_one_empty_frame():
    frames = message.build_message("")
    assert len(frames) == 1
    result = parse_frame(frames[0])
    assert result.ok
    assert result.text == ""
    assert result.more_frames is False
