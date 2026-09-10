"""ASCII/UTF-8 text <-> code-stream conversion.

Plain ASCII (0-127) is sent as a bare code -- it never collides with the
reserved range (129+), so it needs no escaping. Any character outside ASCII
is sent as its UTF-8 byte sequence wrapped in START/CONT/END flags, with
each raw UTF-8 byte individually stuffed (see framing.py) since those bytes
land in 128-255 and routinely collide with reserved codes.

Wire shape per non-ASCII character (n UTF-8 bytes):
    129(START) <byte1> 141(CONT) <byte2> [141 <byte3>] [141 <byte4>] 143(END)

The flags are kept even though UTF-8's own lead byte is self-describing,
because explicit boundaries let the receiver resync after a single lost or
corrupted code instead of relying on implicit length inference, which is
fragile on a lossy channel.
"""
from __future__ import annotations

from . import codes
from .framing import TokenReader, stuff


def encode_text(text: str) -> list[int]:
    out: list[int] = []
    for ch in text:
        raw = ch.encode("utf-8")
        if len(raw) == 1 and raw[0] < 128:
            out.append(raw[0])
            continue
        stuffed_bytes = [stuff(bytes([b])) for b in raw]
        out.append(codes.UTF8_START)
        out.extend(stuffed_bytes[0])
        for sb in stuffed_bytes[1:]:
            out.append(codes.UTF8_CONT)
            out.extend(sb)
        out.append(codes.UTF8_END)
    return out


class DecodeError(Exception):
    pass


def _utf8_expected_len(lead_byte: int) -> int:
    """Byte count implied by a UTF-8 lead byte's own bit pattern, or 0 if
    it isn't a valid multi-byte lead. Used as a resync fallback: if the
    UTF8_END flag itself is the thing that gets lost/corrupted, the START
    flag alone still can't be trusted to end the sequence -- but the lead
    byte tells us exactly how many bytes to expect regardless."""
    if lead_byte & 0b11100000 == 0b11000000:
        return 2
    if lead_byte & 0b11110000 == 0b11100000:
        return 3
    if lead_byte & 0b11111000 == 0b11110000:
        return 4
    return 0


class Utf8CharDecoder:
    """Incremental IDLE / UTF8_ACTIVE state machine: feed it one (kind,
    value) token at a time (as produced by framing.TokenReader.read()) and
    it returns a completed character string when one finishes, or None
    otherwise. Pulled out of decode_codes() as its own class so a caller
    that has *other* flags to recognize in the same token stream (see
    dictionary.py's DICT_LOWER/DICT_TITLE) can interleave its own handling
    with this one, one token at a time, instead of duplicating the whole
    state machine.

    A malformed sequence (unexpected flag, escape at end of stream) aborts
    only the character in progress and resumes at the next clean boundary,
    so a single corrupted code doesn't take down the rest of the message.

    Completion is driven primarily by reaching the lead byte's own expected
    length (_utf8_expected_len), not by waiting for UTF8_END -- testing
    showed that relying on END alone means a single lost END flag causes
    every following byte to be swallowed into the buffer forever, since
    nothing else ever signals "character over." Using the self-describing
    length as the primary trigger and END as a redundant confirmation
    recovers cleanly even when END itself is the corrupted part.
    """

    def __init__(self) -> None:
        self.buf = bytearray()
        self.expected_len = 0
        self.in_char = False

    def _flush(self) -> str | None:
        try:
            text = bytes(self.buf).decode("utf-8")
        except UnicodeDecodeError:
            text = None  # drop the malformed character, keep going
        self.buf = bytearray()
        return text

    def feed(self, kind: str, value: int) -> str | None:
        if not self.in_char:
            if kind == "data":
                return chr(value)
            elif value == codes.UTF8_START:
                self.in_char = True
                self.buf = bytearray()
                self.expected_len = 0
            # stray CONT/END/other flag while idle: nothing to recover, skip
            return None

        # in_char == True
        if kind == "data":
            if not self.buf:
                self.expected_len = _utf8_expected_len(value)
                if self.expected_len == 0:
                    self.in_char = False  # corrupted lead byte, abandon char
                    return None
            self.buf.append(value)
            if len(self.buf) >= self.expected_len:
                self.in_char = False
                return self._flush()
            return None
        elif value == codes.UTF8_CONT:
            return None  # just a separator; next token is the data byte
        elif value == codes.UTF8_END:
            self.in_char = False
            return self._flush() if self.buf else None
        elif value == codes.UTF8_START:
            # previous character was truncated by an error; restart clean
            self.buf = bytearray()
            self.expected_len = 0
            return None
        else:
            # foreign flag leaking into char data: abort this char
            self.in_char = False
            self.buf = bytearray()
            return None


def decode_codes(code_stream: list[int]) -> str:
    reader = TokenReader(code_stream)
    decoder = Utf8CharDecoder()
    result: list[str] = []
    while reader:
        ch = decoder.feed(*reader.read())
        if ch:
            result.append(ch)
    return "".join(result)
