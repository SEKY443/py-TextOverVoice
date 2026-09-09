import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textovervoice import codes
from textovervoice.framing import TokenReader, stuff


def test_stuff_roundtrip_all_byte_values():
    for b in range(256):
        wire = stuff(bytes([b]))
        reader = TokenReader(wire)
        kind, v = reader.read()
        assert kind == "data"
        assert v == b, f"byte {b} did not round-trip"
        assert not reader  # exactly one token consumed


def test_non_reserved_bytes_pass_through_unescaped():
    for b in range(129):  # 0-128 never collide with RESERVED (min is 129)
        wire = stuff(bytes([b]))
        assert wire == [b], f"byte {b} should not have been escaped"


def test_reserved_bytes_are_escaped():
    for b in codes.RESERVED:
        wire = stuff(bytes([b]))
        assert len(wire) == 2
        assert wire[0] == codes.ESCAPE
        assert wire[1] != b  # masked


def test_escape_mask_never_produces_another_reserved_code():
    for b in codes.RESERVED:
        masked = b ^ codes.ESCAPE_MASK
        assert masked not in codes.RESERVED
