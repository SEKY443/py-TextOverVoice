"""Interactive two-way live conversation with adaptive robustness
("recalibration"): tracks whether recently sent messages get acknowledged
by the peer, and steps up FEC/timing robustness when the recent failure
rate gets too high -- stepping back down for speed once things have been
clean for a while.

This is an experiment/demo feature, not a formal protocol. ACK is a
lightweight application-level convention: every real chat message is
tagged with a short random id, and the peer auto-replies with a special
text message carrying that id once it successfully decodes the tagged
message. codes.py already reserves real ACK/NACK wire codes (ACK, NACK)
and handshake opcodes (OP_SYN..OP_READY) for a proper protocol-level
version of this; building that negotiation state machine is future work,
not done here -- this reuses the existing send/listen/message machinery
instead of adding new wire format.

Because the sender's own settings change over time, but the receiver
doesn't otherwise know what changed, Listener is given every (mode,
parity_bytes) combination this module can pick as `candidates` (see
live.Listener's docstring) and tries them in order per received frame --
cheap since it's just re-parsing the same short captured audio a few times.

Two-device usage: both sides run `textovervoice chat`, each pointed at the
other's --my-id as --peer-id. Verified working end-to-end this way (real
hardware, both directions ACKed) using two AdaptiveChat instances in one
Python process.

Important, measured limitation for *testing on one machine*: running two
separate `textovervoice chat` **processes** on the same computer against
the same default mic/speaker deadlocks -- confirmed directly: `sd.play()`
in one process hangs indefinitely for as long as a *different* process has
a `sounddevice.InputStream` open, which is exactly what each chat process
needs in order to listen. This is a PortAudio/CoreAudio cross-process
device contention issue, not a bug in this module -- on two actually
separate physical devices (the real intended use), each side has its own
independent audio hardware, so this contention cannot occur there. If you
need to test two chat sessions on one machine, point them at different
`--device` indices (see `listen --list-devices`) so they don't contend for
the same physical device, or use the AdaptiveChat class directly in one
process (slower due to two listener threads sharing the GIL, but doesn't
deadlock).
"""
from __future__ import annotations

import secrets
import sys
import threading
import time
from dataclasses import dataclass

from . import live
from .message import MAX_FRAME_CHARS

ACK_PREFIX = "\x01ACK:"
MSG_START = "\x02"
MSG_SEP = "\x03"

ACK_WAIT_TIMEOUT_S = 15.0
HISTORY_LEN = 6            # rolling window of recent send outcomes considered for recalibration
ERROR_RATE_HIGH = 0.4      # recalibrate to more robust settings above this failure rate
MODES_BY_ROBUSTNESS = ["fast_air", "phone"]        # index 0 = fastest/least robust
PARITY_STEPS = [10, 20, 40]                        # index 0 = fastest/least robust
FRAME_CHARS_STEPS = [MAX_FRAME_CHARS, 400, 200]    # index 0 = fastest/least robust (fewer, larger frames)


@dataclass
class ChatConfig:
    mode_idx: int = 1     # start at "phone" (index 1) -- safer default than fast_air
    parity_idx: int = 0
    frame_chars_idx: int = 0

    @property
    def mode(self) -> str:
        return MODES_BY_ROBUSTNESS[self.mode_idx]

    @property
    def parity_bytes(self) -> int:
        return PARITY_STEPS[self.parity_idx]

    @property
    def max_frame_chars(self) -> int:
        return FRAME_CHARS_STEPS[self.frame_chars_idx]

    def describe(self) -> str:
        return f"mode={self.mode} parity={self.parity_bytes} max_frame_chars={self.max_frame_chars}"


def _all_candidates() -> list[tuple[str, int]]:
    """Every (mode, parity_bytes) combination the config space can reach,
    most-robust first -- so the receiver's first successful try is usually
    also the most likely one, minimizing wasted re-parses."""
    return [(m, p) for m in reversed(MODES_BY_ROBUSTNESS) for p in reversed(PARITY_STEPS)]


class AdaptiveChat:
    def __init__(self, my_id: int, peer_id: int, mode: str = "phone",
                 session_key: bytes | None = None, device: int | None = None):
        self.my_id = my_id
        self.peer_id = peer_id
        self.session_key = session_key
        self.device = device
        self.config = ChatConfig(mode_idx=MODES_BY_ROBUSTNESS.index(mode))

        self._history: list[bool] = []
        self._pending_acks: dict[str, bool] = {}
        self._lock = threading.Lock()
        self._stream = None

        self._listener = live.Listener(mode=self.config.mode, parity_bytes=self.config.parity_bytes,
                                        my_id=my_id, session_key=session_key, device=device,
                                        on_message=self._on_message, candidates=_all_candidates())

    def start(self) -> None:
        self._stream = self._listener.start_background()

    def stop(self) -> None:
        self._listener.stop_background()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def _on_message(self, msg_result) -> None:
        text = msg_result.text
        if text.startswith(ACK_PREFIX):
            msg_id = text[len(ACK_PREFIX):]
            with self._lock:
                if msg_id in self._pending_acks:
                    self._pending_acks[msg_id] = True
            return

        if text.startswith(MSG_START) and MSG_SEP in text:
            msg_id, _, real_text = text[1:].partition(MSG_SEP)
        else:
            msg_id, real_text = None, text  # plain 'send', not 'chat' -- no ack expected

        print(f"\n<peer {msg_result.dest_id}> {real_text}")
        if msg_id:
            live.send(f"{ACK_PREFIX}{msg_id}", mode=self.config.mode, parity_bytes=self.config.parity_bytes,
                       dest_id=self.peer_id, session_key=self.session_key, device=self.device,
                       display_text=f"(ack for {msg_id})")

    def send_message(self, text: str) -> bool:
        msg_id = secrets.token_hex(3)
        tagged = f"{MSG_START}{msg_id}{MSG_SEP}{text}"
        with self._lock:
            self._pending_acks[msg_id] = False

        live.send(tagged, mode=self.config.mode, parity_bytes=self.config.parity_bytes,
                   dest_id=self.peer_id, session_key=self.session_key, device=self.device,
                   max_frame_chars=self.config.max_frame_chars, display_text=text)

        deadline = time.time() + ACK_WAIT_TIMEOUT_S
        acked = False
        while time.time() < deadline:
            with self._lock:
                acked = self._pending_acks.get(msg_id, False)
            if acked:
                break
            time.sleep(0.2)
        with self._lock:
            self._pending_acks.pop(msg_id, None)

        print(f"[{'ACKed' if acked else 'no ACK -- timed out'}]", file=sys.stderr)
        self._record_outcome(acked)
        return acked

    def _record_outcome(self, ok: bool) -> None:
        self._history.append(ok)
        if len(self._history) > HISTORY_LEN:
            self._history.pop(0)
        self._maybe_recalibrate()

    def _maybe_recalibrate(self) -> None:
        if len(self._history) < 3:
            return
        error_rate = 1 - sum(self._history) / len(self._history)

        if error_rate > ERROR_RATE_HIGH:
            stepped = self._step(more_robust=True)
            if stepped:
                print(f"\n[calibration] error rate {error_rate:.0%} over last {len(self._history)} "
                      f"message(s) -- stepping up robustness: {self.config.describe()}", file=sys.stderr)
                self._history.clear()
                self._resync_listener()
        elif error_rate == 0.0 and len(self._history) == HISTORY_LEN:
            stepped = self._step(more_robust=False)
            if stepped:
                print(f"\n[calibration] channel clean over last {len(self._history)} message(s) "
                      f"-- trying faster settings: {self.config.describe()}", file=sys.stderr)
                self._history.clear()
                self._resync_listener()

    def _step(self, more_robust: bool) -> bool:
        """Steps exactly one knob (mode, then parity, then frame size) in
        the requested direction. Returns False if already at the extreme."""
        c = self.config
        if more_robust:
            if c.mode_idx < len(MODES_BY_ROBUSTNESS) - 1:
                c.mode_idx += 1
                return True
            if c.parity_idx < len(PARITY_STEPS) - 1:
                c.parity_idx += 1
                return True
            if c.frame_chars_idx < len(FRAME_CHARS_STEPS) - 1:
                c.frame_chars_idx += 1
                return True
            return False
        else:
            if c.frame_chars_idx > 0:
                c.frame_chars_idx -= 1
                return True
            if c.parity_idx > 0:
                c.parity_idx -= 1
                return True
            if c.mode_idx > 0:
                c.mode_idx -= 1
                return True
            return False

    def _resync_listener(self) -> None:
        """Only our own send-side settings need to change for recalibration
        to work (the listener already tries every candidate per frame --
        see _all_candidates) -- this just keeps self.config.mode's display
        value and my_id/session_key consistent if anything downstream reads
        self._listener.mode directly."""
        self._listener.mode = self.config.mode
        self._listener.parity_bytes = self.config.parity_bytes


def run(my_id: int, peer_id: int, mode: str = "phone", session_key: bytes | None = None,
        device: int | None = None) -> None:
    chat = AdaptiveChat(my_id, peer_id, mode, session_key, device)
    chat.start()
    print(f"Chat started. my_id={my_id} peer_id={peer_id} starting_config={chat.config.describe()}",
          file=sys.stderr)
    print("Type a message and press Enter to send. Ctrl+C or Ctrl+D to quit.\n", file=sys.stderr)

    try:
        while True:
            try:
                text = input("> ")
            except EOFError:
                break
            if not text.strip():
                continue
            chat.send_message(text)
    except KeyboardInterrupt:
        pass
    finally:
        chat.stop()
        print("\nChat ended.", file=sys.stderr)
