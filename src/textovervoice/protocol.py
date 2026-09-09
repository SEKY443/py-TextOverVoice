"""Frame assembly/parsing: ties charset + fec + framing together into the
wire format:

    SOF | LENGTH(2B) | PAYLOAD (charset-encoded, stuffed) | PARITY_START |
    RS PARITY (stuffed) | PARITY_END | CRC_HI | CRC(2B, stuffed) | CRC_LO

FEC and CRC protect the exact wire-form payload bytes (post-stuffing, as
they travel over the channel) -- not a resolved/unescaped form. This matters:
the two sides must run FEC/CRC over identical byte sequences, and the
receiver can't safely unescape before FEC correction anyway, since a
corrupted escape sequence would misparse. So the flow is: collect raw wire
codes for the payload region -> FEC-correct -> CRC-verify -> only then hand
the corrected wire codes to charset.decode_codes, which does its own
escape/flag resolution via TokenReader.

LENGTH and the CRC/marker bytes themselves are not FEC-protected in this
version (only the payload+parity region is) -- a corrupted header byte still
fails the frame outright. That, and the fixed-timing-only demod (no
preamble/frame sync yet -- see modem.py's demodulate docstring), are the
main open gaps before this could run against a live call.

Multi-frame handshake/ACK/NACK sequencing isn't implemented here yet --
this module only builds/parses a single frame's codes; modem.py turns codes
into audio.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import codes, fec
from .charset import decode_codes, encode_text
from .framing import TokenReader, stuff


@dataclass
class ParseResult:
    ok: bool
    text: str | None
    reason: str = ""


def build_frame(text: str, parity_bytes: int = fec.DEFAULT_PARITY_BYTES) -> list[int]:
    payload_codes = encode_text(text)  # wire form: flags + stuffed data, already escaped
    payload_bytes = bytes(payload_codes)

    parity = fec.protect(payload_bytes, parity_bytes)
    crc = fec.crc16_ccitt(payload_bytes)

    frame: list[int] = [codes.FEC_SOF]
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


def parse_frame(received: list[int], parity_bytes: int = fec.DEFAULT_PARITY_BYTES) -> ParseResult:
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
        length_bytes = read_data_bytes(2)
        declared_len = int.from_bytes(length_bytes, "big")

        payload_start = reader.pos
        parity_start_idx = _find_bare_flag(received, payload_start, codes.FEC_PARITY_START)
        if parity_start_idx == -1:
            return ParseResult(ok=False, text=None, reason="PARITY_START marker not found")
        payload_wire = received[payload_start:parity_start_idx]

        reader.pos = parity_start_idx
        expect_flag("PARITY_START", codes.FEC_PARITY_START)

        parity_start2 = reader.pos
        parity_end_idx = _find_bare_flag(received, parity_start2, codes.FEC_PARITY_END)
        if parity_end_idx == -1:
            return ParseResult(ok=False, text=None, reason="PARITY_END marker not found")
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
        return ParseResult(ok=False, text=None,
                            reason=f"length mismatch: declared {declared_len}, got {len(payload_wire)}")

    # parity_wire is stuffed on the wire; unstuff it back to raw parity bytes
    # before handing it to reedsolo (parity bytes were never meant to be
    # interpreted as codes, only escaped for safe transport).
    parity_reader = TokenReader(parity_wire)
    parity_bytes_val = bytearray()
    while parity_reader:
        _, v = parity_reader.read()
        parity_bytes_val.append(v)

    combined = bytes(payload_wire) + bytes(parity_bytes_val)
    corrected = fec.recover(combined, parity_bytes)
    if corrected is None:
        return ParseResult(ok=False, text=None, reason="FEC uncorrectable")

    if fec.crc16_ccitt(corrected) != received_crc:
        return ParseResult(ok=False, text=None, reason="CRC mismatch after FEC")

    text = decode_codes(list(corrected))
    return ParseResult(ok=True, text=text)
