"""Offline CLI: text <-> WAV using the real modem/framing/FEC stack.

Uses modem.modulate_frame/demodulate_frame, which prepend/locate a chirp
preamble via cross-correlation -- so decode does NOT need to assume symbol 0
starts at sample 0 of the file (arbitrary leading silence/padding is fine).
Still doesn't talk to a live call; use tools/codec_validation/ scripts to
see how this waveform holds up after a real AMR pass or real speaker/mic
acoustic transmission.

--mode selects a named timing profile (modem.MODES): "phone" (default,
validated through real AMR-NB) or "fast_air" (roughly 2x faster, validated
only acoustically with no codec involved -- NOT expected to survive AMR
compression; see modem.MODES's docstring for the real-hardware sweep behind
these numbers).

Addressing (--dest-id/--my-id) and encryption (--my-privkey/--peer-pubkey)
are both opt-in -- see protocol.py's module docstring and crypto.py for
what they do and don't protect against.

Long text is automatically split into multiple independently-synced frames
(message.py, --max-frame-chars) rather than sent as one giant frame --
tested on the full GPLv3 license (35KB) through real AMR-NB, a single frame
suffers a hard timing-drift "cliff" partway through at low bitrate; multiple
shorter frames each get their own fresh preamble sync, bounding how far
drift can accumulate before it's corrected.

'send'/'listen' do the same thing live, through the speaker/microphone
(live.py) instead of a WAV file -- needs `sounddevice`
(pip install -e ".[acoustic-test]").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from . import crypto, message, modem
from .fec import DEFAULT_PARITY_BYTES
from .protocol import BROADCAST_ID, frame_wire_length, parse_frame

MAX_SYMBOLS_PER_FRAME = 2000  # generous cap; real frame length always comes
                               # from the LENGTH field, this just bounds how
                               # much audio one demodulate() call chews through
PREAMBLE_BACKOFF_S = 1.0  # safety margin subtracted from the calculated next-frame
                           # boundary before searching, so a small sample-count
                           # mismatch (real audio doesn't preserve exact counts)
                           # can't cause the search to start inside the next
                           # preamble instead of right before it
SEARCH_WINDOW_S = 20  # bounds each individual preamble search -- kept deliberately
                       # small (not "comfortably larger than one frame", which an
                       # earlier version tried at 180s and still had this bug).
                       # modem.find_preamble returns the single GLOBALLY strongest
                       # correlation peak in whatever it's given, so ANY window
                       # that can contain more than one preamble risks jumping
                       # straight to whichever one correlates best overall,
                       # skipping nearer frames out of sequence entirely (this
                       # was an actual bug, caught by testing the full GPLv3
                       # text: decode found frame seq=37 before seq=0, and later,
                       # even after adding frame_wire_length's exact-jump logic,
                       # still occasionally skipped frames whenever that jump
                       # fell back to the cruder preamble-length-only skip). A
                       # small window can't contain two frames' preambles at
                       # once; _scan_for_preamble below advances through the
                       # file in overlapping small windows when a given one
                       # comes up empty, rather than using one large window as
                       # the primary search mechanism.


def keygen(priv_path: str, pub_path: str) -> None:
    priv, pub = crypto.generate_keypair()
    priv_hex = crypto.serialize_private_key(priv).hex()
    pub_hex = crypto.serialize_public_key(pub).hex()
    Path(priv_path).write_text(priv_hex + "\n")
    Path(pub_path).write_text(pub_hex + "\n")
    print(f"Private key written to {priv_path} (keep secret, never share)", file=sys.stderr)
    print(f"Public key written to {pub_path}", file=sys.stderr)
    print(f"\nShare this public key with your peer:\n{pub_hex}", file=sys.stderr)


def _load_private_key(path: str) -> crypto.X25519PrivateKey:
    return crypto.load_private_key(bytes.fromhex(Path(path).read_text().strip()))


def _load_peer_public_key(path_or_hex: str) -> crypto.X25519PublicKey:
    p = Path(path_or_hex)
    hex_str = p.read_text().strip() if p.exists() else path_or_hex.strip()
    return crypto.load_public_key(bytes.fromhex(hex_str))


def _resolve_session_key(args: argparse.Namespace) -> bytes | None:
    if args.my_privkey is None and args.peer_pubkey is None:
        return None
    if args.my_privkey is None or args.peer_pubkey is None:
        print("error: --my-privkey and --peer-pubkey must be given together", file=sys.stderr)
        sys.exit(2)
    my_priv = _load_private_key(args.my_privkey)
    peer_pub = _load_peer_public_key(args.peer_pubkey)
    return crypto.derive_session_key(my_priv, peer_pub)


def encode(text: str, out_path: str, parity_bytes: int, symbol_duration: float, guard: float,
           dest_id: int, session_key: bytes | None, max_frame_chars: int) -> None:
    frames = message.build_message(text, parity_bytes, dest_id=dest_id, session_key=session_key,
                                    max_frame_chars=max_frame_chars)
    inter_frame_silence = np.zeros(int(0.3 * modem.SR))  # keeps preambles from running together

    audio_parts = []
    for frame_codes in frames:
        symbols = modem.bytes_to_symbols(bytes(frame_codes))
        audio_parts.append(modem.modulate_frame(symbols, symbol_duration_s=symbol_duration, guard_s=guard))
        audio_parts.append(inter_frame_silence)
    audio = np.concatenate(audio_parts)

    wavfile.write(out_path, modem.SR, (audio * 32767).astype(np.int16))
    extras = []
    if dest_id != BROADCAST_ID:
        extras.append(f"dest_id={dest_id}")
    if session_key is not None:
        extras.append("encrypted")
    extra_str = f" [{', '.join(extras)}]" if extras else ""
    print(f"Encoded {len(text)} chars -> {len(frames)} frame(s) -> {out_path} "
          f"({len(audio)/modem.SR:.2f}s total){extra_str}", file=sys.stderr)


def _scan_for_preamble(audio: np.ndarray, start: int) -> tuple[int, float] | None:
    """Finds the next preamble at or after `start`, advancing through the
    file in small (SEARCH_WINDOW_S), overlapping windows rather than
    searching one huge span at once -- see SEARCH_WINDOW_S's comment for
    why a large window is a correctness bug here, not just a performance
    concern. Returns (absolute_sample_offset, score), or None if no
    preamble is found before the end of the file."""
    search_window_n = SEARCH_WINDOW_S * modem.SR
    preamble_len_n = int(modem.PREAMBLE_DURATION_S * modem.SR)
    pos = start
    while pos < len(audio):
        found = modem.find_preamble(audio[pos:pos + search_window_n])
        if found is not None:
            offset, score = found
            return pos + offset, score
        window_actually_searched = min(search_window_n, len(audio) - pos)
        if window_actually_searched <= preamble_len_n:
            return None  # not enough audio left for even one more preamble
        pos += window_actually_searched - preamble_len_n  # overlap so a chirp
        # straddling this window's trailing edge isn't split across two scans
    return None


def decode(in_path: str, parity_bytes: int, symbol_duration: float, guard: float,
           my_id: int | None, session_key: bytes | None) -> None:
    sr, audio = wavfile.read(in_path)
    if sr != modem.SR:
        print(f"warning: file is {sr}Hz, expected {modem.SR}Hz", file=sys.stderr)
    audio = audio.astype(np.float64) / 32767.0

    step_n = int((symbol_duration + guard) * modem.SR)
    preamble_len_n = int(modem.PREAMBLE_DURATION_S * modem.SR)
    reassembler = message.MessageReassembler()
    search_start = 0
    frame_num = 0

    while search_start < len(audio):
        found = _scan_for_preamble(audio, search_start)
        if found is None:
            break
        abs_offset, score = found
        payload_start = abs_offset + int(modem.PREAMBLE_GUARD_S * modem.SR)
        available = max(len(audio) - payload_start, 0)
        n_symbols = min(available // step_n, MAX_SYMBOLS_PER_FRAME)

        detections = modem.demodulate(audio[payload_start:], n_symbols,
                                       symbol_duration_s=symbol_duration, guard_s=guard)
        symbols = [d.symbol for d in detections]
        n_bytes = (len(symbols) * modem.BITS_PER_SYMBOL) // 8
        frame_codes = list(modem.symbols_to_bytes(symbols, n_bytes))

        frame_num += 1
        result = parse_frame(frame_codes, parity_bytes, my_id=my_id, session_key=session_key)
        print(f"[frame {frame_num}] sync={score:.2f} "
              f"{'ok, seq=' + str(result.seq) if result.ok else 'FAILED: ' + result.reason}",
              file=sys.stderr)

        msg_result = reassembler.add(result)
        if msg_result.ok:
            print(msg_result.text)
            return

        consumed_codes = frame_wire_length(frame_codes)
        if consumed_codes is not None:
            # Exact boundary known -- jump straight to where the next
            # preamble should start instead of relying on
            # _scan_for_preamble's incremental crawl (which is the
            # correctness safety net, not the primary mechanism -- this
            # jump is just faster when the header decoded cleanly enough
            # to compute it). Back off by a small margin first: real audio
            # (post-AMR, etc.) doesn't preserve sample counts exactly, so
            # landing precisely on the calculated boundary can overshoot
            # into the middle of the next preamble, weakening its
            # correlation. Backing off re-includes a bit of
            # already-processed silence, which is harmless.
            consumed_symbols = -(-consumed_codes * 8 // 6)  # ceil division, matches bytes_to_symbols padding
            exact_end = payload_start + consumed_symbols * step_n
            search_start = max(payload_start, exact_end - int(PREAMBLE_BACKOFF_S * modem.SR))
        else:
            # Header too corrupted to even find the frame's own boundary --
            # skip just the preamble and let _scan_for_preamble's
            # incremental crawl (not a single wide search) find the next
            # one correctly from here.
            search_start = abs_offset + preamble_len_n

    print("DECODE FAILED: message incomplete (ran out of audio or preambles)", file=sys.stderr)
    sys.exit(1)


def _resolve_timing(args: argparse.Namespace) -> tuple[float, float]:
    profile = modem.MODES[args.mode]
    symbol_duration = args.symbol_duration if args.symbol_duration is not None else profile["symbol_duration_s"]
    guard = args.guard if args.guard is not None else profile["guard_s"]
    return symbol_duration, guard


def _add_crypto_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--my-privkey", default=None, help="path to your private key file (see 'keygen')")
    p.add_argument("--peer-pubkey", default=None,
                    help="peer's public key: a file path, or the hex string directly")


def main() -> None:
    parser = argparse.ArgumentParser(prog="textovervoice")
    sub = parser.add_subparsers(dest="cmd", required=True)

    kg = sub.add_parser("keygen", help="generate an X25519 keypair for encryption")
    kg.add_argument("priv_out", help="output path for the private key (keep secret)")
    kg.add_argument("pub_out", help="output path for the public key (share with your peer)")

    enc = sub.add_parser("encode", help="text -> WAV")
    enc.add_argument("text")
    enc.add_argument("out_wav")
    enc.add_argument("--parity-bytes", type=int, default=DEFAULT_PARITY_BYTES)
    enc.add_argument("--mode", choices=list(modem.MODES), default="phone",
                      help="timing profile: 'phone' (AMR-validated) or 'fast_air' "
                           "(~2x faster, acoustic-only, not AMR-safe)")
    enc.add_argument("--symbol-duration", type=float, default=None,
                      help="override --mode's symbol duration (seconds)")
    enc.add_argument("--guard", type=float, default=None,
                      help="override --mode's guard interval (seconds)")
    enc.add_argument("--dest-id", type=int, default=BROADCAST_ID,
                      help="addressee (0-254), default broadcast (everyone accepts)")
    enc.add_argument("--max-frame-chars", type=int, default=message.MAX_FRAME_CHARS,
                      help="split text longer than this into multiple independently-synced frames")
    _add_crypto_args(enc)

    dec = sub.add_parser("decode", help="WAV -> text")
    dec.add_argument("in_wav")
    dec.add_argument("--parity-bytes", type=int, default=DEFAULT_PARITY_BYTES)
    dec.add_argument("--mode", choices=list(modem.MODES), default="phone")
    dec.add_argument("--symbol-duration", type=float, default=None)
    dec.add_argument("--guard", type=float, default=None)
    dec.add_argument("--my-id", type=int, default=None,
                      help="if given, reject frames not addressed to this id "
                           "(broadcast frames are always accepted)")
    _add_crypto_args(dec)

    snd = sub.add_parser("send", help="text -> speaker (live)")
    snd.add_argument("text")
    snd.add_argument("--parity-bytes", type=int, default=DEFAULT_PARITY_BYTES)
    snd.add_argument("--mode", choices=list(modem.MODES), default="phone")
    snd.add_argument("--dest-id", type=int, default=BROADCAST_ID)
    snd.add_argument("--max-frame-chars", type=int, default=message.MAX_FRAME_CHARS)
    snd.add_argument("--device", type=int, default=None, help="output device index")
    _add_crypto_args(snd)

    lst = sub.add_parser("listen", help="microphone -> text (live)")
    lst.add_argument("--parity-bytes", type=int, default=DEFAULT_PARITY_BYTES)
    lst.add_argument("--mode", choices=list(modem.MODES), default="phone")
    lst.add_argument("--my-id", type=int, default=None)
    lst.add_argument("--device", type=int, default=None, help="input device index")
    lst.add_argument("--list-devices", action="store_true")
    _add_crypto_args(lst)

    cht = sub.add_parser("chat", help="two-way live conversation with adaptive robustness")
    cht.add_argument("--my-id", type=int, required=True)
    cht.add_argument("--peer-id", type=int, required=True)
    cht.add_argument("--mode", choices=list(modem.MODES), default="phone",
                      help="starting timing profile; recalibration may move away from it")
    cht.add_argument("--device", type=int, default=None)
    _add_crypto_args(cht)

    args = parser.parse_args()

    if args.cmd == "keygen":
        keygen(args.priv_out, args.pub_out)
        return

    if args.cmd == "send":
        from . import live
        session_key = _resolve_session_key(args)
        live.send(args.text, args.mode, args.parity_bytes, args.dest_id, session_key,
                  args.device, args.max_frame_chars)
        return

    if args.cmd == "listen":
        if args.list_devices:
            import sounddevice as sd
            print(sd.query_devices())
            return
        from . import live
        session_key = _resolve_session_key(args)
        live.listen(args.mode, args.parity_bytes, args.my_id, session_key, args.device)
        return

    if args.cmd == "chat":
        from . import chat
        session_key = _resolve_session_key(args)
        chat.run(args.my_id, args.peer_id, args.mode, session_key, args.device)
        return

    symbol_duration, guard = _resolve_timing(args)
    session_key = _resolve_session_key(args)
    if args.cmd == "encode":
        encode(args.text, args.out_wav, args.parity_bytes, symbol_duration, guard,
               args.dest_id, session_key, args.max_frame_chars)
    elif args.cmd == "decode":
        decode(args.in_wav, args.parity_bytes, symbol_duration, guard, args.my_id, session_key)


if __name__ == "__main__":
    main()
