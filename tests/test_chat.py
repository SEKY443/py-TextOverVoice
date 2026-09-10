import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textovervoice import chat


def test_config_starts_at_phone_by_default():
    c = chat.ChatConfig(mode_idx=chat.MODES_BY_ROBUSTNESS.index("phone"))
    assert c.mode == "phone"
    assert c.parity_bytes == chat.PARITY_STEPS[0]
    assert c.max_frame_chars == chat.FRAME_CHARS_STEPS[0]


def test_step_more_robust_escalates_mode_then_parity_then_frame_size():
    c = chat.ChatConfig(mode_idx=0, parity_idx=0, frame_chars_idx=0)  # fast_air, weakest everything

    session = chat.AdaptiveChat.__new__(chat.AdaptiveChat)  # bypass __init__ (no audio needed)
    session.config = c

    assert session._step(more_robust=True) is True
    assert c.mode == "phone"

    assert session._step(more_robust=True) is True
    assert c.parity_bytes == chat.PARITY_STEPS[1]

    # exhaust remaining parity steps before frame_chars starts moving
    while c.parity_idx < len(chat.PARITY_STEPS) - 1:
        session._step(more_robust=True)
    assert c.parity_bytes == chat.PARITY_STEPS[-1]

    assert session._step(more_robust=True) is True
    assert c.max_frame_chars == chat.FRAME_CHARS_STEPS[1]


def test_step_more_robust_returns_false_at_maximum():
    c = chat.ChatConfig(mode_idx=len(chat.MODES_BY_ROBUSTNESS) - 1,
                         parity_idx=len(chat.PARITY_STEPS) - 1,
                         frame_chars_idx=len(chat.FRAME_CHARS_STEPS) - 1)
    session = chat.AdaptiveChat.__new__(chat.AdaptiveChat)
    session.config = c
    assert session._step(more_robust=True) is False


def test_step_less_robust_returns_false_at_minimum():
    c = chat.ChatConfig(mode_idx=0, parity_idx=0, frame_chars_idx=0)
    session = chat.AdaptiveChat.__new__(chat.AdaptiveChat)
    session.config = c
    assert session._step(more_robust=False) is False


def test_recalibrate_steps_up_after_sustained_failures():
    session = chat.AdaptiveChat.__new__(chat.AdaptiveChat)
    session.config = chat.ChatConfig(mode_idx=0, parity_idx=0, frame_chars_idx=0)  # fast_air
    session._history = []
    session._resync_listener = lambda: None  # no listener object in this unit test

    for _ in range(5):
        session._record_outcome(False)  # every send fails to get ACKed

    assert session.config.mode == "phone"  # stepped up from fast_air


def test_recalibrate_steps_down_after_sustained_success():
    session = chat.AdaptiveChat.__new__(chat.AdaptiveChat)
    session.config = chat.ChatConfig(mode_idx=1, parity_idx=1, frame_chars_idx=1)  # phone, some margin
    session._history = []
    session._resync_listener = lambda: None

    for _ in range(chat.HISTORY_LEN):
        session._record_outcome(True)

    # should have stepped down exactly one knob (frame_chars first)
    assert session.config.frame_chars_idx == 0


def test_recalibrate_does_nothing_with_too_little_history():
    session = chat.AdaptiveChat.__new__(chat.AdaptiveChat)
    session.config = chat.ChatConfig(mode_idx=0, parity_idx=0, frame_chars_idx=0)
    session._history = []
    session._resync_listener = lambda: None

    session._record_outcome(False)
    session._record_outcome(False)
    assert session.config.mode == "fast_air"  # only 2 data points, no verdict yet


def test_all_candidates_covers_full_config_space():
    candidates = chat._all_candidates()
    assert len(candidates) == len(chat.MODES_BY_ROBUSTNESS) * len(chat.PARITY_STEPS)
    assert set(m for m, _ in candidates) == set(chat.MODES_BY_ROBUSTNESS)
    assert set(p for _, p in candidates) == set(chat.PARITY_STEPS)


def test_message_tag_format_roundtrip():
    msg_id = "abc123"
    text = "hello there"
    tagged = f"{chat.MSG_START}{msg_id}{chat.MSG_SEP}{text}"
    parsed_id, _, parsed_text = tagged[1:].partition(chat.MSG_SEP)
    assert parsed_id == msg_id
    assert parsed_text == text


def test_ack_prefix_is_distinguishable_from_real_messages():
    ack = f"{chat.ACK_PREFIX}abc123"
    tagged = f"{chat.MSG_START}abc123{chat.MSG_SEP}hi"
    assert ack.startswith(chat.ACK_PREFIX)
    assert not tagged.startswith(chat.ACK_PREFIX)
