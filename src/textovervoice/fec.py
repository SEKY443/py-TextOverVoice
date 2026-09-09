"""Reed-Solomon FEC + CRC-16 integrity check.

RS correction and CRC verification are independent: RS corrects what it can,
CRC confirms the result is actually right. RS can silently mis-correct
beyond its guaranteed bound, so a frame that fails CRC is treated as failed
even if RS reported success.

Reed-Solomon uses `reedsolo` (widely used, MIT-licensed, pure-Python RS
implementation) rather than a hand-rolled codec -- correctness of an
error-correcting code is exactly the kind of thing to reuse a maintained
library for rather than reimplement.

Validated empirically in tools/codec_validation/realism_suite.py: at AMR-NB's
worst bitrate (4.75k), plain payloads corrupted (1-3 byte errors out of
40-76 bytes) and failed exact recovery; the same payloads with RS(n,10)
parity recovered exactly every time in that test run. 10 parity bytes is a
starting point, not a validated final parameter -- real burst-length
statistics from actual calls are needed to size this properly.

GF(256) Reed-Solomon caps a single codeword at 255 symbols total, so data
longer than (255 - parity_bytes) bytes is split into multiple chunks here,
each with its own parity_bytes of parity. A first version didn't do this
explicitly and instead let `reedsolo` silently chunk internally on
`.encode()`, then mishandled the result as if it were one contiguous
parity block -- broke silently (not a channel-noise artifact; reproduced
with zero audio/codec involved) on any payload past ~245 bytes, e.g. a
~65-character message in a 2-byte-per-char script like Cyrillic. Chunking
explicitly here, with the caller passing the original data length so both
sides derive identical chunk boundaries, fixes it and makes the boundary
condition a normal, tested code path instead of an implicit library detail.
"""
from __future__ import annotations

import reedsolo

DEFAULT_PARITY_BYTES = 10
MAX_RS_BLOCK = 255  # GF(256) codeword length ceiling (data + parity)


def _chunk(data: bytes, chunk_size: int) -> list[bytes]:
    # range(0, 0, n) is already empty, so b"" naturally yields zero chunks --
    # matching reedsolo's own treatment of empty input as a no-op (0 bytes
    # of parity, not parity_bytes of parity for a padded empty chunk).
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def protect(data: bytes, parity_bytes: int = DEFAULT_PARITY_BYTES) -> bytes:
    """Returns only the parity bytes (one parity_bytes-sized block per
    chunk, concatenated) -- not the full RS codeword. The frame format
    (protocol.py) transmits PAYLOAD and PARITY as separate labeled regions,
    so the caller keeps data and parity apart and passes both to recover()."""
    chunk_size = MAX_RS_BLOCK - parity_bytes
    rsc = reedsolo.RSCodec(parity_bytes)
    parity_out = bytearray()
    for chunk in _chunk(data, chunk_size):
        full = rsc.encode(chunk)
        parity_out += full[len(chunk):]
    return bytes(parity_out)


def recover(data: bytes, parity: bytes, parity_bytes: int = DEFAULT_PARITY_BYTES) -> bytes | None:
    """`parity` must be exactly what protect(data, parity_bytes) produced
    (chunk boundaries are derived from len(data), so both sides need to
    agree on that length -- protocol.py gets it from the frame's LENGTH
    field). Returns the corrected data, or None if RS could not correct it
    (caller should treat None the same as a CRC mismatch -> NACK)."""
    chunk_size = MAX_RS_BLOCK - parity_bytes
    rsc = reedsolo.RSCodec(parity_bytes)
    out = bytearray()
    pi = 0
    for chunk in _chunk(data, chunk_size):
        chunk_parity = parity[pi:pi + parity_bytes]
        pi += parity_bytes
        if len(chunk_parity) != parity_bytes:
            return None  # truncated parity region, can't have come from protect()
        try:
            decoded, _, _ = rsc.decode(bytearray(chunk + chunk_parity))
            out += decoded
        except reedsolo.ReedSolomonError:
            return None
    return bytes(out)


_CRC16_CCITT_POLY = 0x1021
_CRC16_CCITT_INIT = 0xFFFF  # CRC-16/CCITT-FALSE


def crc16_ccitt(data: bytes) -> int:
    crc = _CRC16_CCITT_INIT
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ _CRC16_CCITT_POLY) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc
