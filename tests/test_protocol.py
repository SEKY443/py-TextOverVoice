import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textovervoice import crypto
from textovervoice.protocol import BROADCAST_ID, build_frame, parse_frame


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
    # 10 parity bytes) -- avoid indices 0-5 (SOF/DEST_ID/FLAGS/SEQ/LENGTH)
    # so the frame structure itself stays parseable.
    for i in (6, 12, 20):
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


# --- Addressing (dest_id) ---------------------------------------------------

def test_default_dest_id_is_broadcast():
    frame = build_frame("hi")
    result = parse_frame(frame, my_id=42)  # any specific id should still accept broadcast
    assert result.ok
    assert result.dest_id == BROADCAST_ID


def test_addressed_frame_accepted_by_intended_recipient():
    frame = build_frame("for bob", dest_id=7)
    result = parse_frame(frame, my_id=7)
    assert result.ok
    assert result.text == "for bob"
    assert result.dest_id == 7


def test_addressed_frame_rejected_by_other_recipient_without_running_fec():
    frame = build_frame("for bob", dest_id=7)
    result = parse_frame(frame, my_id=99)
    assert not result.ok
    assert result.dest_id == 7
    assert "not addressed to me" in result.reason


def test_no_my_id_means_everyone_accepts_regardless_of_dest_id():
    frame = build_frame("for bob", dest_id=7)
    result = parse_frame(frame)  # my_id not given -> no filtering
    assert result.ok
    assert result.text == "for bob"


def test_dest_id_out_of_range_rejected():
    import pytest
    with pytest.raises(ValueError):
        build_frame("hi", dest_id=256)


# --- Encryption --------------------------------------------------------------

def _session_keypair():
    alice_priv, alice_pub = crypto.generate_keypair()
    bob_priv, bob_pub = crypto.generate_keypair()
    alice_key = crypto.derive_session_key(alice_priv, bob_pub)
    bob_key = crypto.derive_session_key(bob_priv, alice_pub)
    assert alice_key == bob_key
    return alice_key, bob_key


def test_encrypted_roundtrip():
    alice_key, bob_key = _session_keypair()
    text = "Secret message 秘密 \U0001F600"
    frame = build_frame(text, session_key=alice_key)
    result = parse_frame(frame, session_key=bob_key)
    assert result.ok
    assert result.text == text


def test_encrypted_frame_without_session_key_fails_cleanly():
    alice_key, _ = _session_keypair()
    frame = build_frame("secret", session_key=alice_key)
    result = parse_frame(frame)  # no session_key given
    assert not result.ok
    assert "encrypted" in result.reason


def test_encrypted_frame_with_wrong_key_fails_cleanly():
    alice_key, _ = _session_keypair()
    unrelated_key, _ = _session_keypair()  # a session key from a different key exchange entirely
    frame = build_frame("secret", session_key=alice_key)
    result = parse_frame(frame, session_key=unrelated_key)
    assert not result.ok
    assert not result.text


def test_encrypted_payload_is_not_readable_plaintext_on_the_wire():
    """Sanity check that encryption is actually happening -- the frame
    codes shouldn't contain the plaintext's characters in the clear."""
    alice_key, _ = _session_keypair()
    text = "PLAINTEXT_MARKER_XYZ"
    frame = build_frame(text, session_key=alice_key)
    frame_bytes = bytes(b for b in frame if b < 256)
    assert b"PLAINTEXT_MARKER_XYZ" not in frame_bytes


def test_encryption_and_addressing_combine():
    alice_key, bob_key = _session_keypair()
    frame = build_frame("for bob, secretly", dest_id=7, session_key=alice_key)

    eve_result = parse_frame(frame, my_id=99, session_key=bob_key)
    assert not eve_result.ok  # wrong recipient, rejected before decryption is even attempted

    bob_result = parse_frame(frame, my_id=7, session_key=bob_key)
    assert bob_result.ok
    assert bob_result.text == "for bob, secretly"


# --- Sequencing (seq / more_frames) -----------------------------------------

def test_default_seq_and_more_frames():
    frame = build_frame("hi")
    result = parse_frame(frame)
    assert result.ok
    assert result.seq == 0
    assert result.more_frames is False


def test_seq_and_more_frames_roundtrip():
    frame = build_frame("part one", seq=3, more_frames=True)
    result = parse_frame(frame)
    assert result.ok
    assert result.seq == 3
    assert result.more_frames is True


def test_seq_out_of_range_rejected():
    import pytest
    with pytest.raises(ValueError):
        build_frame("hi", seq=256)


def test_seq_survives_corruption_alongside_other_header_fields():
    frame = build_frame("The quick brown fox jumps over the lazy dog",
                         parity_bytes=10, seq=5, more_frames=True)
    corrupted = list(frame)
    for i in (6, 12, 20):
        corrupted[i] ^= 0x01
    result = parse_frame(corrupted, parity_bytes=10)
    assert result.ok
    assert result.seq == 5
    assert result.more_frames is True
