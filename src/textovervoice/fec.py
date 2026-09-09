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
"""
from __future__ import annotations

import reedsolo

DEFAULT_PARITY_BYTES = 10


def protect(data: bytes, parity_bytes: int = DEFAULT_PARITY_BYTES) -> bytes:
    """Returns only the parity suffix, not the full RS codeword -- the
    frame format (protocol.py) transmits PAYLOAD and PARITY as separate
    labeled regions, so the caller reconstructs data+parity before calling
    recover(); reedsolo's own .encode() returns them already concatenated,
    which would double the payload if returned as-is here."""
    full = reedsolo.RSCodec(parity_bytes).encode(data)
    return bytes(full[len(data):])


def recover(codeword: bytes, parity_bytes: int = DEFAULT_PARITY_BYTES) -> bytes | None:
    """`codeword` is data+parity concatenated (i.e. data + protect(data)).
    Returns the corrected data, or None if RS could not correct it (caller
    should treat None the same as a CRC mismatch -> NACK)."""
    try:
        decoded, _, _ = reedsolo.RSCodec(parity_bytes).decode(bytearray(codeword))
        return bytes(decoded)
    except reedsolo.ReedSolomonError:
        return None


_CRC16_CCITT_POLY = 0x1021
_CRC16_CCITT_INIT = 0xFFFF  # CRC-16/CCITT-FALSE


def crc16_ccitt(data: bytes) -> int:
    crc = _CRC16_CCITT_INIT
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ _CRC16_CCITT_POLY) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc
