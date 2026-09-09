"""Byte-stuffing for the shared reserved-code space.

Any literal data byte (UTF-8 continuation bytes, RS parity, CRC bytes) that
numerically collides with a reserved control code (see codes.py) must be
escaped, or the receiver cannot tell data from framing. This isn't a rare
edge case: UTF-8 continuation bytes are always in range 128-191, which fully
contains 129/141/143 and half of 144-157, so real multi-byte text collides
with reserved codes routinely, not just as an occasional parity-byte
accident.
"""
from __future__ import annotations

from . import codes


def stuff(data: bytes) -> list[int]:
    """Encode raw bytes into the code stream, escaping any byte that would
    otherwise be indistinguishable from a reserved control code."""
    out: list[int] = []
    for b in data:
        if b in codes.RESERVED:
            out.append(codes.ESCAPE)
            out.append(b ^ codes.ESCAPE_MASK)
        else:
            out.append(b)
    return out


class TokenReader:
    """Scans a received code stream, transparently resolving escape
    sequences. `read()` returns (kind, value): kind is "data" for any
    literal byte (whether it arrived bare or via an escape pair) and "flag"
    for a bare reserved code, which the caller (charset/protocol layer)
    interprets according to its own state machine.
    """

    def __init__(self, codes_stream: list[int]):
        self._codes = codes_stream
        self._i = 0

    def __bool__(self) -> bool:
        return self._i < len(self._codes)

    @property
    def pos(self) -> int:
        return self._i

    @pos.setter
    def pos(self, value: int) -> None:
        self._i = value

    def read(self) -> tuple[str, int]:
        if self._i >= len(self._codes):
            raise EOFError("no more codes")
        c = self._codes[self._i]
        self._i += 1
        if c == codes.ESCAPE:
            if self._i >= len(self._codes):
                raise ValueError("escape code at end of stream with no follower")
            literal = self._codes[self._i] ^ codes.ESCAPE_MASK
            self._i += 1
            return "data", literal
        if c in codes.RESERVED:
            return "flag", c
        return "data", c
