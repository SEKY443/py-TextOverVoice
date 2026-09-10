"""Frame assembly/parsing: ties charset + fec + framing (+ optional crypto)
together into the wire format:

    SOF | DEST_ID(1B) | FLAGS(1B) | SEQ(1B) | LENGTH(2B) | PAYLOAD |
    PARITY_START | RS PARITY (stuffed) | PARITY_END | CRC_HI |
    CRC(2B, stuffed) | CRC_LO

DEST_ID, FLAGS, and SEQ are opt-in features, all default to "off"
(DEST_ID=BROADCAST_ID, FLAGS=0/plaintext/single-frame, SEQ=0) -- addressing,
encryption, and multi-frame messages are strict additions, not
requirements, so plain single-frame broadcast usage is unaffected.

SEQ + FLAG_MORE_FRAMES exist because of a real, measured failure mode: a
single preamble at the start of a long (multi-minute) transmission can't
correct for timing drift that accumulates *during* the transmission --
tested on the full GPLv3 text (35KB) through real AMR-NB at 4.75kbps, which
produced a hard "cliff": the first ~15% of Reed-Solomon chunks decoded
clean, then essentially everything after failed, in a contiguous block, not
scattered noise. A controlled comparison (decoding from the correctly
frame-synced offset vs. a naively assumed fixed offset) produced the
*identical* failure pattern, which rules out sync-finding error and points
squarely at ongoing drift the one-time sync can't fix. message.py splits
long text into multiple independently-preambled frames so drift is bounded
per-frame instead of compounding across an entire long message -- see its
docstring for the splitting/reassembly logic built on top of this.

FEC and CRC protect the exact wire-form payload bytes (post-stuffing, as
they travel over the channel) -- not a resolved/unescaped form. This matters:
the two sides must run FEC/CRC over identical byte sequences, and the
receiver can't safely unescape before FEC correction anyway, since a
corrupted escape sequence would misparse. So the flow is: collect raw wire
codes for the payload region -> FEC-correct -> CRC-verify -> only then hand
the corrected wire codes to charset/dictionary.decode_codes (plaintext) or
crypto.decrypt (encrypted), each of which does its own escape/flag
resolution via TokenReader/unstuff_bytes.

DEST_ID/FLAGS/SEQ/LENGTH and the CRC/marker bytes themselves are not
FEC-protected in this version (only the payload+parity region is) -- a
corrupted header byte still fails that one frame outright. Splitting into
multiple frames (message.py) also bounds the cost of that: one frame's
header getting corrupted only loses that frame's chunk of text, not the
whole message, whereas before this existed it could lose everything.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import charset, codes, crypto as crypto_mod, dictionary, fec
from .framing import TokenReader, stuff, unstuff_bytes

BROADCAST_ID = 255  # dest_id value meaning "for everyone" -- the default
FLAG_ENCRYPTED = 0b00000001
FLAG_MORE_FRAMES = 0b00000010  # message continues in a subsequent frame


@dataclass
class ParseResult:
    ok: bool
    text: str | None
    reason: str = ""
    dest_id: int | None = None
    seq: int | None = None
    more_frames: bool | None = None


def build_frame(text: str, parity_bytes: int = fec.DEFAULT_PARITY_BYTES,
                 use_dictionary: bool = True, lang: str = "en",
                 dest_id: int = BROADCAST_ID,
                 session_key: bytes | None = None,
                 seq: int = 0, more_frames: bool = False) -> list[int]:
    """dest_id: optional addressee, 0-254 for a specific recipient or
    BROADCAST_ID (default) for everyone -- addressing is opt-in.

    session_key: if given (see crypto.derive_session_key), the payload is
    ChaCha20-Poly1305-encrypted instead of charset/dictionary-encoded text
    -- also opt-in, off by default. When set, use_dictionary/lang are
    ignored (encrypted bytes aren't text, there's nothing to compress or
    UTF-8-flag).

    seq/more_frames: normally set by message.build_message, not called
    directly -- seq is this frame's 0-indexed position (wraps past 255),
    more_frames says whether another frame follows to complete the message."""
    if not 0 <= dest_id <= 255:
        raise ValueError(f"dest_id {dest_id} out of range 0-255")
    if not 0 <= seq <= 255:
        raise ValueError(f"seq {seq} out of range 0-255")

    if session_key is not None:
        ciphertext = crypto_mod.encrypt(session_key, text.encode("utf-8"))
        payload_codes = stuff(ciphertext)
        flags = FLAG_ENCRYPTED
    else:
        payload_codes = dictionary.encode_text(text, lang) if use_dictionary else charset.encode_text(text)
        flags = 0
    if more_frames:
        flags |= FLAG_MORE_FRAMES
    payload_bytes = bytes(payload_codes)  # wire form: flags + stuffed data, already escaped

    parity = fec.protect(payload_bytes, parity_bytes)
    crc = fec.crc16_ccitt(payload_bytes)

    frame: list[int] = [codes.FEC_SOF]
    frame += stuff(bytes([dest_id]))
    frame += stuff(bytes([flags]))
    frame += stuff(bytes([seq]))
    frame += stuff(len(payload_bytes).to_bytes(2, "big"))
    frame += payload_codes
    frame.append(codes.FEC_PARITY_START)
    frame += stuff(parity)
    frame.append(codes.FEC_PARITY_END)
    frame.append(codes.CRC_MARKER_HI)
    frame += stuff(crc.to_bytes(2, "big"))
    frame.append(codes.CRC_MARKER_LO)
    return frame


def _find_bare_flag(wire: list[int], start: int, target: int) -> int:
    """Raw-code boundary scan: skips escape pairs (2 codes) without
    resolving them, since we need the boundary index, not the value.
    A bare reserved code that isn't `target` (e.g. a UTF-8 flag inside the
    payload) is just payload content at this layer and is skipped as
    ordinary data. Returns -1 if `target` never appears bare."""
    i = start
    n = len(wire)
    while i < n:
        c = wire[i]
        if c == codes.ESCAPE:
            i += 2
            continue
        if c == target:
            return i
        i += 1
    return -1


def frame_wire_length(wire: list[int], start: int = 0) -> int | None:
    """Scans one frame starting at `start` and returns the total number of
    wire codes it occupies (SOF through the final CRC_LO's data,
    inclusive), or None if the structure can't even be walked (too short,
    markers missing). Exists for multi-frame scanning (cli.py, live.py):
    a caller demodulating a generous, over-sized audio window needs to know
    exactly where THIS frame's real content ends, so the next preamble
    search starts precisely there -- not a guessed window that might
    contain zero, one, or several subsequent preambles (a guessed window
    was a real, measured bug: scanning the full remaining file after each
    frame made modem.find_preamble return whichever preamble correlated
    globally strongest in that window, which is not necessarily the
    nearest one, causing frames to be found wildly out of sequence)."""
    reader = TokenReader(wire[start:])
    try:
        kind, v = reader.read()
        if kind != "flag" or v != codes.FEC_SOF:
            return None
        reader.read()  # dest_id
        reader.read()  # flags
        reader.read()  # seq
        reader.read()  # length byte 1
        reader.read()  # length byte 2
    except (ValueError, EOFError):
        return None

    payload_start = reader.pos
    idx = _find_bare_flag(wire[start:], payload_start, codes.FEC_PARITY_START)
    if idx == -1:
        return None
    reader.pos = idx

    try:
        reader.read()  # PARITY_START
    except (ValueError, EOFError):
        return None

    parity_start2 = reader.pos
    idx2 = _find_bare_flag(wire[start:], parity_start2, codes.FEC_PARITY_END)
    if idx2 == -1:
        return None
    reader.pos = idx2

    try:
        reader.read()  # PARITY_END
        reader.read()  # CRC_HI
        reader.read()  # crc byte 1
        reader.read()  # crc byte 2
        reader.read()  # CRC_LO
    except (ValueError, EOFError):
        return None

    return start + reader.pos


def parse_frame(received: list[int], parity_bytes: int = fec.DEFAULT_PARITY_BYTES,
                 use_dictionary: bool = True, lang: str = "en",
                 my_id: int | None = None,
                 session_key: bytes | None = None) -> ParseResult:
    """use_dictionary/lang must match what build_frame used for plaintext
    frames, or dictionary hits will be misread (see dictionary.py).

    my_id: if given, a frame not addressed to this id (and not BROADCAST_ID)
    is rejected immediately after reading DEST_ID -- before FEC/CRC even
    run -- both for the "only the intended receiver responds" demo effect
    and as a real efficiency win (skip decoding entirely for frames that
    aren't yours).

    session_key: required to decrypt a frame with FLAG_ENCRYPTED set;
    without it such a frame is reported as failed, not silently ignored."""
    reader = TokenReader(received)

    def expect_flag(name: str, value: int) -> None:
        kind, v = reader.read()
        if kind != "flag" or v != value:
            raise ValueError(f"expected {name} ({value}), got {kind}={v}")

    def read_data_bytes(n: int) -> bytes:
        out = bytearray()
        for _ in range(n):
            kind, v = reader.read()
            if kind != "data":
                raise ValueError(f"expected data byte, got flag={v}")
            out.append(v)
        return bytes(out)

    try:
        expect_flag("SOF", codes.FEC_SOF)
        dest_id = read_data_bytes(1)[0]

        if my_id is not None and dest_id != BROADCAST_ID and dest_id != my_id:
            return ParseResult(ok=False, text=None,
                                reason=f"not addressed to me (dest_id={dest_id})", dest_id=dest_id)

        flags = read_data_bytes(1)[0]
        seq = read_data_bytes(1)[0]
        more_frames = bool(flags & FLAG_MORE_FRAMES)
        length_bytes = read_data_bytes(2)
        declared_len = int.from_bytes(length_bytes, "big")

        payload_start = reader.pos
        parity_start_idx = _find_bare_flag(received, payload_start, codes.FEC_PARITY_START)
        if parity_start_idx == -1:
            return ParseResult(ok=False, text=None, reason="PARITY_START marker not found",
                                dest_id=dest_id, seq=seq, more_frames=more_frames)
        payload_wire = received[payload_start:parity_start_idx]

        reader.pos = parity_start_idx
        expect_flag("PARITY_START", codes.FEC_PARITY_START)

        parity_start2 = reader.pos
        parity_end_idx = _find_bare_flag(received, parity_start2, codes.FEC_PARITY_END)
        if parity_end_idx == -1:
            return ParseResult(ok=False, text=None, reason="PARITY_END marker not found",
                                dest_id=dest_id, seq=seq, more_frames=more_frames)
        parity_wire = received[parity_start2:parity_end_idx]

        reader.pos = parity_end_idx
        expect_flag("PARITY_END", codes.FEC_PARITY_END)
        expect_flag("CRC_HI", codes.CRC_MARKER_HI)
        crc_bytes = read_data_bytes(2)
        expect_flag("CRC_LO", codes.CRC_MARKER_LO)
        received_crc = int.from_bytes(crc_bytes, "big")

    except (ValueError, EOFError) as e:
        return ParseResult(ok=False, text=None, reason=f"frame structure error: {e}")

    if len(payload_wire) != declared_len:
        return ParseResult(ok=False, text=None, dest_id=dest_id, seq=seq, more_frames=more_frames,
                            reason=f"length mismatch: declared {declared_len}, got {len(payload_wire)}")

    # parity_wire is stuffed on the wire; unstuff it back to raw parity bytes
    # before handing it to reedsolo (parity bytes were never meant to be
    # interpreted as codes, only escaped for safe transport).
    parity_bytes_val = unstuff_bytes(parity_wire)

    corrected = fec.recover(bytes(payload_wire), parity_bytes_val, parity_bytes)
    if corrected is None:
        return ParseResult(ok=False, text=None, reason="FEC uncorrectable",
                            dest_id=dest_id, seq=seq, more_frames=more_frames)

    if fec.crc16_ccitt(corrected) != received_crc:
        return ParseResult(ok=False, text=None, reason="CRC mismatch after FEC",
                            dest_id=dest_id, seq=seq, more_frames=more_frames)

    if flags & FLAG_ENCRYPTED:
        if session_key is None:
            return ParseResult(ok=False, text=None, dest_id=dest_id, seq=seq, more_frames=more_frames,
                                reason="frame is encrypted but no session_key given")
        raw_ciphertext = unstuff_bytes(list(corrected))
        plaintext = crypto_mod.decrypt(session_key, raw_ciphertext)
        if plaintext is None:
            return ParseResult(ok=False, text=None, dest_id=dest_id, seq=seq, more_frames=more_frames,
                                reason="decryption failed (wrong key or tampered)")
        text = plaintext.decode("utf-8", errors="replace")
    elif use_dictionary:
        text = dictionary.decode_codes(list(corrected), lang)
    else:
        text = charset.decode_codes(list(corrected))

    return ParseResult(ok=True, text=text, dest_id=dest_id, seq=seq, more_frames=more_frames)
