import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textovervoice import fec


def test_crc16_ccitt_known_vector():
    # Standard CRC-16/CCITT-FALSE test vector.
    assert fec.crc16_ccitt(b"123456789") == 0x29B1


def test_rs_protect_returns_parity_only_sized_to_parity_bytes():
    data = b"the quick brown fox"
    parity = fec.protect(data, 10)
    assert len(parity) == 10  # single chunk: not the full codeword


def test_rs_protect_recover_clean():
    data = b"the quick brown fox"
    parity = fec.protect(data)
    assert fec.recover(data, parity) == data


def test_rs_corrects_errors_within_budget():
    data = b"the quick brown fox jumps"
    parity_bytes = 10  # corrects up to 5 byte errors
    parity = bytearray(fec.protect(data, parity_bytes))
    codeword = bytearray(data) + parity
    for i in (0, 5, 10, 15, 20):
        codeword[i] ^= 0xFF
    corrupted_data, corrupted_parity = bytes(codeword[:len(data)]), bytes(codeword[len(data):])
    assert fec.recover(corrupted_data, corrupted_parity, parity_bytes) == data


def test_rs_fails_beyond_budget_without_crashing():
    data = b"the quick brown fox jumps over"
    parity_bytes = 10  # corrects up to 5 byte errors
    parity = fec.protect(data, parity_bytes)
    codeword = bytearray(data) + bytearray(parity)
    for i in range(0, 20, 2):  # 10 errors, over budget
        codeword[i] ^= 0xFF
    corrupted_data, corrupted_parity = bytes(codeword[:len(data)]), bytes(codeword[len(data):])
    result = fec.recover(corrupted_data, corrupted_parity, parity_bytes)
    assert result is None or result != data  # must not silently return wrong-but-unflagged data as if correct


def test_rs_handles_data_longer_than_single_gf256_block():
    """Regression test: GF(256) RS caps a codeword at 255 symbols. A first
    implementation let reedsolo silently chunk internally on encode() and
    then mishandled the result as one contiguous parity block, which broke
    on any payload past ~245 bytes (found via a ~65-character Cyrillic
    message, whose 2-byte-per-char UTF-8 encoding pushed the frame payload
    past that threshold). This exercises the same boundary explicitly."""
    data = bytes(range(256)) * 2  # 512 bytes, spans multiple RS blocks
    parity_bytes = 10
    parity = fec.protect(data, parity_bytes)
    assert fec.recover(data, parity) == data

    # and it should still actually correct errors within each chunk's budget
    codeword = bytearray(data) + bytearray(parity)
    codeword[0] ^= 0xFF     # error in chunk 1
    codeword[300] ^= 0xFF   # error in chunk 2 (chunk_size = 245, so byte 300 is in chunk 2)
    corrupted_data, corrupted_parity = bytes(codeword[:len(data)]), bytes(codeword[len(data):])
    assert fec.recover(corrupted_data, corrupted_parity, parity_bytes) == data


def test_rs_empty_data():
    parity = fec.protect(b"", 10)
    assert fec.recover(b"", parity, 10) == b""
