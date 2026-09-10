"""Exhaustive offline decode of a 48kHz WAV file, using the same sample
rate live.py's Listener captures at (DEVICE_SR=48000) -- companion to
record_and_compare.py. cli.py's decode() can't be reused directly for
this: it hardcodes modem.SR (8000), a different rate that goes through a
different CoreAudio resampling path and isn't equivalent for diagnosing a
live-only failure. This mirrors cli.py's decode()/_scan_for_preamble logic
with sr threaded through as 48000 instead.

Usage: python decode_48k.py <in.wav> [mode] [parity_bytes]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from textovervoice import message, modem  # noqa: E402
from textovervoice.protocol import frame_wire_length, parse_frame  # noqa: E402

SR = 48000
SEARCH_WINDOW_S = 0.6
PREAMBLE_BACKOFF_S = 0.1
MAX_SYMBOLS_PER_FRAME = 2000


def scan_for_preamble(audio, start, reference):
    search_window_n = int(SEARCH_WINDOW_S * SR)
    preamble_len_n = int(modem.PREAMBLE_DURATION_S * SR)
    pos = start
    while pos < len(audio):
        found = modem.find_preamble(audio[pos:pos + search_window_n], reference=reference)
        if found is not None:
            offset, score = found
            return pos + offset, score
        window_actually_searched = min(search_window_n, len(audio) - pos)
        if window_actually_searched <= preamble_len_n:
            return None
        pos += window_actually_searched - preamble_len_n
    return None


def main():
    in_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "phone"
    parity_bytes = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    profile = modem.MODES[mode]
    symbol_duration, guard = profile["symbol_duration_s"], profile["guard_s"]

    sr, audio = wavfile.read(in_path)
    print(f"file sr={sr}", file=sys.stderr)
    audio = audio.astype(np.float64) / 32767.0

    reference = modem.generate_preamble(sr=SR)
    step_n = int((symbol_duration + guard) * SR)
    preamble_len_n = int(modem.PREAMBLE_DURATION_S * SR)
    reassembler = message.MessageReassembler()
    search_start = 0
    frame_num = 0

    while search_start < len(audio):
        found = scan_for_preamble(audio, search_start, reference)
        if found is None:
            break
        abs_offset, score = found
        payload_start = abs_offset + int(modem.PREAMBLE_GUARD_S * SR)
        available = max(len(audio) - payload_start, 0)
        n_symbols = min(available // step_n, MAX_SYMBOLS_PER_FRAME)

        detections = modem.demodulate(audio[payload_start:], n_symbols,
                                       symbol_duration_s=symbol_duration, guard_s=guard, sr=SR)
        symbols = [d.symbol for d in detections]
        n_bytes = (len(symbols) * modem.BITS_PER_SYMBOL) // 8
        frame_codes = list(modem.symbols_to_bytes(symbols, n_bytes))

        frame_num += 1
        result = parse_frame(frame_codes, parity_bytes)
        print(f"[frame {frame_num}] sync={score:.2f} "
              f"{'ok, seq=' + str(result.seq) if result.ok else 'FAILED: ' + result.reason}",
              file=sys.stderr)

        msg_result = reassembler.add(result)
        if msg_result.ok:
            print(msg_result.text)
            return

        consumed_codes = frame_wire_length(frame_codes)
        if consumed_codes is not None:
            consumed_symbols = -(-consumed_codes * 8 // 6)
            exact_end = payload_start + consumed_symbols * step_n
            search_start = max(payload_start, exact_end - int(PREAMBLE_BACKOFF_S * SR))
        else:
            search_start = abs_offset + preamble_len_n

    print("DECODE FAILED: message incomplete", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
