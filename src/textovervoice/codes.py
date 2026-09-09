"""Reserved control-code constants shared by every protocol layer.

129/141/143 mark UTF-8 multi-byte character boundaries (charset.py).
144-157 is a block reserved for frame control / FEC / CRC / ACK-NACK
(protocol.py, fec.py). ESCAPE (147) drives byte-stuffing (framing.py) so
that literal data bytes landing in this range can't be confused with the
control codes themselves.
"""

# UTF-8 multi-byte boundary flags
UTF8_START = 129
UTF8_CONT = 141
UTF8_END = 143

# 144-157: frame control / FEC / CRC / ACK-NACK reserved block
FEC_SOF = 144
FEC_SCHEME_ID_LO = 145
FEC_SCHEME_ID_HI = 146
ESCAPE = 147
CRC_MARKER_HI = 148
CRC_MARKER_LO = 149
FEC_PARITY_START = 150
FEC_PARITY_END = 152
FEC_SOFT_SUCCESS = 153
FEC_UNCORRECTABLE = 154
FEC_RESERVED = 155
ACK = 156
NACK = 157

# Handshake opcodes
OP_SYN = 1
OP_SYN_ACK = 2
OP_ACK_EST = 3
OP_CAL_START = 4
OP_CAL_RESULT = 5
OP_PARAM = 6
OP_READY = 7

RESERVED = frozenset({UTF8_START, UTF8_CONT, UTF8_END} | set(range(FEC_SOF, NACK + 1)))

ESCAPE_MASK = 0x20  # XOR mask for byte-stuffing; see framing.py

assert ESCAPE not in RESERVED - {ESCAPE}  # ESCAPE is itself reserved but self-consistent
for _code in RESERVED:
    assert (_code ^ ESCAPE_MASK) not in RESERVED, (
        f"escape mask collides for reserved code {_code}"
    )
