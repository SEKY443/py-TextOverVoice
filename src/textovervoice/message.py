"""Multi-frame messages: splits long text into multiple independently-synced
frames (protocol.py builds/parses one frame at a time), so a single-preamble
transmission's drift-over-time problem is bounded per-chunk instead of
compounding over an entire long message.

The problem this solves is real and measured, not theoretical: sending the
full GPLv3 text (35KB) through real AMR-NB at 4.75kbps as a single frame
produced a hard "cliff" -- the first ~15% of Reed-Solomon chunks decoded
clean, then essentially everything after failed, in one contiguous block.
A controlled test (decoding from the frame-sync-detected offset vs. a
naively assumed fixed offset) produced the *identical* failure pattern,
which rules out a bad starting sync point and points at ongoing timing
drift accumulating through the transmission -- something a single sync
point at the very start structurally cannot correct for. See protocol.py's
module docstring for the full writeup. Splitting into multiple
independently-preambled frames means each one gets a fresh sync point, so
drift can only accumulate within one frame's (much shorter) duration.

Splitting is by character count (MAX_FRAME_CHARS), not by bytes -- Python
string slicing by character index is always UTF-8-safe, so no character
gets split across a frame boundary. Encryption, dictionary compression, and
addressing are applied per-frame (independently), not to the whole message
before splitting, so each frame is a fully self-contained, independently
decodable/decryptable unit -- reassembly just concatenates already-decoded
text, never partial ciphertext or partial charset state across frames.

A message that fits in one frame is exactly what protocol.build_frame
alone would have produced (seq=0, more_frames=False) -- this module is a
strict extension, not a wire-format change for short messages.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import fec
from .protocol import BROADCAST_ID, ParseResult, build_frame

MAX_FRAME_CHARS = 800
# Conservative by design: the observed drift-cliff in the GPLv3 test set in
# around 15% into a ~35-minute phone-mode transmission (roughly 5 minutes
# of audio). Capping frames at 800 characters keeps each frame's audio
# duration well under a minute even at phone mode's slower timing -- a
# large safety margin below the one drift-onset point actually measured,
# not a value derived from a full characterization of how drift onset
# scales with bitrate/mode/duration (that would need many more real trials).


def build_message(text: str, parity_bytes: int = fec.DEFAULT_PARITY_BYTES,
                   use_dictionary: bool = True, lang: str = "en",
                   dest_id: int = BROADCAST_ID, session_key: bytes | None = None,
                   max_frame_chars: int = MAX_FRAME_CHARS) -> list[list[int]]:
    """Splits `text` into <=max_frame_chars chunks and builds one frame per
    chunk (protocol.build_frame), each tagged with its position (seq) and
    whether more frames follow (more_frames). Returns a list of frames
    (each a list of codes, as build_frame returns) -- modulating/playing
    them is the caller's job (one modem.modulate_frame call per frame, with
    a silence gap between so preambles don't run into each other)."""
    chunks = [text[i:i + max_frame_chars] for i in range(0, len(text), max_frame_chars)] or [""]
    frames = []
    for i, chunk in enumerate(chunks):
        more = i < len(chunks) - 1
        frames.append(build_frame(chunk, parity_bytes, use_dictionary, lang, dest_id, session_key,
                                   seq=i % 256, more_frames=more))
    return frames


@dataclass
class MessageResult:
    ok: bool
    text: str | None
    reason: str = ""
    dest_id: int | None = None
    frames_received: int = 0
    frames_expected: int | None = None  # unknown until the frame with more_frames=False arrives


class MessageReassembler:
    """Accumulates parsed frames (by seq) for one message-in-progress.
    Feed it every ParseResult as frames arrive (in any order -- it doesn't
    assume sequential arrival, though in practice frames are sent that
    way); once a contiguous run from seq=0 through the frame with
    more_frames=False has been seen, `add` returns the completed message.

    One instance tracks one message. A caller expecting multiple
    back-to-back messages should create a fresh instance per message (e.g.
    after a completed or abandoned reassembly), since seq numbers wrap and
    aren't globally unique across messages.
    """

    def __init__(self) -> None:
        self._parts: dict[int, str] = {}
        self._last_seq: int | None = None  # seq of the frame with more_frames=False, once seen

    def add(self, result: ParseResult) -> MessageResult:
        if not result.ok:
            return MessageResult(ok=False, text=None, reason=result.reason, dest_id=result.dest_id)

        if result.seq is None:
            # a frame parsed without seq info at all (shouldn't happen via
            # protocol.parse_frame, which always sets it on success) --
            # treat defensively as a complete standalone message.
            return MessageResult(ok=True, text=result.text, dest_id=result.dest_id,
                                  frames_received=1, frames_expected=1)

        self._parts[result.seq] = result.text
        if not result.more_frames:
            self._last_seq = result.seq

        if self._last_seq is not None and all(i in self._parts for i in range(self._last_seq + 1)):
            full_text = "".join(self._parts[i] for i in range(self._last_seq + 1))
            return MessageResult(ok=True, text=full_text, dest_id=result.dest_id,
                                  frames_received=len(self._parts), frames_expected=self._last_seq + 1)

        return MessageResult(ok=False, text=None, reason="waiting for more frames",
                              dest_id=result.dest_id, frames_received=len(self._parts),
                              frames_expected=(self._last_seq + 1) if self._last_seq is not None else None)

    def reset(self) -> None:
        self._parts = {}
        self._last_seq = None
