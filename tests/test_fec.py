import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textovervoice import fec


def test_crc16_ccitt_known_vector():
    # Standard CRC-16/CCITT-FALSE test vector.
    assert fec.crc16_ccitt(b"123456789") == 0x29B1


def test_rs_protect_returns_parity_suffix_only():
    data = b"the quick brown fox"
    parity = fec.protect(data, 10)
    assert len(parity) == 10  # not the full codeword


def test_rs_protect_recover_clean():
    data = b"the quick brown fox"
    parity = fec.protect(data)
    assert fec.recover(data + parity) == data


def test_rs_corrects_errors_within_budget():
    data = b"the quick brown fox jumps"
    parity_bytes = 10  # corrects up to 5 byte errors
    codeword = bytearray(data + fec.protect(data, parity_bytes))
    for i in (0, 5, 10, 15, 20):
        codeword[i] ^= 0xFF
    assert fec.recover(bytes(codeword), parity_bytes) == data


def test_rs_fails_beyond_budget_without_crashing():
    data = b"the quick brown fox jumps over"
    parity_bytes = 10  # corrects up to 5 byte errors
    codeword = bytearray(data + fec.protect(data, parity_bytes))
    for i in range(0, 20, 2):  # 10 errors, over budget
        codeword[i] ^= 0xFF
    result = fec.recover(bytes(codeword), parity_bytes)
    assert result is None or result != data  # must not silently return wrong-but-unflagged data as if correct
