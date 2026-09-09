"""Offline CLI: text <-> WAV using the real modem/framing/FEC stack.

This produces and consumes the actual over-the-air waveform (2-of-16 MFSK),
but doesn't yet do preamble-based frame sync (decode assumes symbol 0 starts
at sample 0 of the file) or talk to a live phone call -- both are open
items. Use tools/codec_validation/ scripts to see how this waveform holds
up after a real AMR pass.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.io import wavfile

from . import modem
from .fec import DEFAULT_PARITY_BYTES
from .protocol import build_frame, parse_frame


def encode(text: str, out_path: str, parity_bytes: int, symbol_duration: float) -> None:
    frame_codes = build_frame(text, parity_bytes)
    symbols = modem.bytes_to_symbols(bytes(frame_codes))
    audio = modem.modulate(symbols, symbol_duration_s=symbol_duration)
    wavfile.write(out_path, modem.SR, (audio * 32767).astype(np.int16))
    duration_s = len(audio) / modem.SR
    print(f"Encoded {len(text)} chars -> {len(frame_codes)} frame codes -> "
          f"{len(symbols)} symbols -> {out_path} ({duration_s:.2f}s)", file=sys.stderr)


def decode(in_path: str, parity_bytes: int, symbol_duration: float, n_frame_bytes: int) -> None:
    sr, audio = wavfile.read(in_path)
    if sr != modem.SR:
        print(f"warning: file is {sr}Hz, expected {modem.SR}Hz", file=sys.stderr)
    audio = audio.astype(np.float64) / 32767.0

    n_symbols = len(audio) // int((symbol_duration + modem.DEFAULT_GUARD_S) * modem.SR)
    dets = modem.demodulate(audio, n_symbols, symbol_duration_s=symbol_duration)
    symbols = [d.symbol for d in dets]
    n_bytes = (len(symbols) * modem.BITS_PER_SYMBOL) // 8
    frame_codes = list(modem.symbols_to_bytes(symbols, n_bytes))

    result = parse_frame(frame_codes, parity_bytes)
    if not result.ok:
        print(f"DECODE FAILED: {result.reason}", file=sys.stderr)
        sys.exit(1)
    print(result.text)


def main() -> None:
    parser = argparse.ArgumentParser(prog="textovervoice")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="text -> WAV")
    enc.add_argument("text")
    enc.add_argument("out_wav")
    enc.add_argument("--parity-bytes", type=int, default=DEFAULT_PARITY_BYTES)
    enc.add_argument("--symbol-duration", type=float, default=modem.DEFAULT_SYMBOL_DURATION_S,
                      help="seconds per symbol; keep at least 0.04 -- see "
                           "tools/codec_validation/realism_suite.py's duration-sensitivity "
                           "results for why shorter durations fail badly at low AMR bitrates")

    dec = sub.add_parser("decode", help="WAV -> text")
    dec.add_argument("in_wav")
    dec.add_argument("--parity-bytes", type=int, default=DEFAULT_PARITY_BYTES)
    dec.add_argument("--symbol-duration", type=float, default=modem.DEFAULT_SYMBOL_DURATION_S)
    dec.add_argument("--max-frame-bytes", type=int, default=4096,
                      help="upper bound used to size the symbol scan; unused beyond that")

    args = parser.parse_args()
    if args.cmd == "encode":
        encode(args.text, args.out_wav, args.parity_bytes, args.symbol_duration)
    elif args.cmd == "decode":
        decode(args.in_wav, args.parity_bytes, args.symbol_duration, args.max_frame_bytes)


if __name__ == "__main__":
    main()
