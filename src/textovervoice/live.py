"""Live (real-time) speaker/microphone send and receive, for interactive
demos -- as opposed to cli.py's encode/decode, which round-trip through a
WAV file. Needs `sounddevice` (pip install -e ".[acoustic-test]"), imported
lazily here so the rest of the package doesn't require it just to use
encode/decode/keygen.

Runs the whole pipeline natively at DEVICE_SR (48kHz, universally
supported) rather than resampling to/from modem.SR (8kHz) -- every
modem.py function already accepts an `sr` parameter, so there's no need to
downsample streamed audio chunk-by-chunk, which would risk boundary
artifacts at each callback's edge (each independent resample_poly call
on a short chunk is not equivalent to resampling the whole stream at once).

listen() runs a background rolling-buffer scan: sounddevice streams
microphone audio into a buffer via a callback (real-time, can't block), a
polling loop periodically scans the unconsumed tail of that buffer for a
preamble (modem.find_preamble) and, once found, retries a full decode
(modem.demodulate + protocol.parse_frame) as more audio accumulates, up to
a timeout. This is a simplification appropriate for a demo, not a
production streaming receiver -- a real implementation would decode
incrementally as symbols arrive rather than re-attempting a growing window
on every poll.

Long text is split into multiple independently-synced frames (message.py)
rather than sent as one giant frame -- see message.py's module docstring
for why (a single preamble can't correct for timing drift that accumulates
over a long transmission; this was measured, not theoretical). Each
frame's result feeds a message.MessageReassembler so listen() only prints
once the full message is back together.

Listener accepts a `candidates` list of (mode, parity_bytes) pairs to try
per detected preamble, instead of one fixed setting -- this exists for
chat.py's adaptive recalibration: if the sender changes its own mode/parity
based on channel conditions, a receiver with fixed settings would simply
fail to decode from then on, with no way to know what changed. Trying a
handful of candidates per frame is cheap (it's just re-parsing the same
short, already-captured audio a few times) and sidesteps needing a formal
negotiation handshake for this experiment.

IMPORTANT, hard-won operational note: `sd.play()`/`OutputStream.write()`
can hang indefinitely -- not just run slowly -- if macOS puts the audio
subsystem into its "DarkWakeSilenceBuffers" idle power state, which the
system log showed happening after ~30s with no display/keyboard/mouse
activity, independent of whether something else (e.g. a `caffeinate`
process) is already preventing full system sleep. This is a real macOS
power-management behavior, not a bug in this module, PortAudio, or
sounddevice -- confirmed by watching `log show --predicate 'process ==
"coreaudiod"'` during a hang and by having `caffeinate -u` (which asserts
the user is active, unlike a plain `caffeinate` that only blocks idle
*system* sleep) immediately unstick it. **Run any of send/listen/chat
under `caffeinate -u` if the machine's display might otherwise go idle**
(e.g. during a live demo where nobody's touching the keyboard) --
otherwise audio can silently stop working partway through with no error,
which is a confusing failure mode to hit live.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

from . import message, modem
from .protocol import BROADCAST_ID, ParseResult, frame_wire_length, parse_frame

if TYPE_CHECKING:
    import sounddevice as sd


@contextmanager
def _prevent_display_sleep():
    """Runs `caffeinate -u` for the duration of the block (macOS only, a
    no-op elsewhere) -- see this module's docstring for why: without it,
    audio playback can silently hang after ~30s of no keyboard/mouse
    activity, which is exactly the situation during a live demo or an
    unattended listen/chat session.

    The brief sleep after spawning matters: caffeinate's assertion isn't
    registered with the system instantaneously, and starting audio I/O
    before it lands reproduces the exact same hang this exists to prevent
    (measured directly -- without this delay the hang still happened even
    with caffeinate already launched moments earlier)."""
    if sys.platform != "darwin":
        yield
        return
    proc = subprocess.Popen(["caffeinate", "-u"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    try:
        yield
    finally:
        proc.terminate()

DEVICE_SR = 48000
MAX_SYMBOLS = 2000         # generous cap on one frame's length
PREAMBLE_TIMEOUT_S = 8.0   # give up on a detected preamble if it never resolves
POLL_INTERVAL_S = 0.3
SEARCH_WINDOW_S = 15       # each poll's search window -- kept small for real-time
                            # responsiveness (measured ~10ms at this size, post-FFT-fix;
                            # the old direct-convolution find_preamble took 40+ SECONDS
                            # at cli.py's 180s bulk-file window size, which made live
                            # polling effectively non-functional). Safe to keep small
                            # because _try_decode's _scan_position advances through
                            # unscanned audio incrementally rather than needing one big
                            # window to contain everything -- a smaller window here also
                            # can't jump to a wrong out-of-sequence preamble the way an
                            # oversized one could (see cli.py's SEARCH_WINDOW_S for that
                            # bug's full story).
PREAMBLE_BACKOFF_S = 1.0   # see cli.py's PREAMBLE_BACKOFF_S


def send(text: str, mode: str = "phone", parity_bytes: int = 10,
         dest_id: int = BROADCAST_ID, session_key: bytes | None = None,
         device: int | None = None, max_frame_chars: int = message.MAX_FRAME_CHARS,
         display_text: str | None = None) -> None:
    """display_text overrides what's logged for the ">> Sending" line --
    for callers (chat.py) that wrap `text` in non-printing control
    characters (a hidden message-id tag) which would otherwise show up
    concatenated into the visible log with no separator."""
    import sounddevice as sd

    profile = modem.MODES[mode]
    frames = message.build_message(text, parity_bytes, dest_id=dest_id, session_key=session_key,
                                    max_frame_chars=max_frame_chars)
    inter_frame_silence = np.zeros(int(0.3 * DEVICE_SR))

    audio_parts = []
    for frame_codes in frames:
        symbols = modem.bytes_to_symbols(bytes(frame_codes))
        audio_parts.append(modem.modulate_frame(symbols, symbol_duration_s=profile["symbol_duration_s"],
                                                  guard_s=profile["guard_s"], sr=DEVICE_SR))
        audio_parts.append(inter_frame_silence)
    audio = np.concatenate(audio_parts)

    extras = []
    if dest_id != BROADCAST_ID:
        extras.append(f"dest_id={dest_id}")
    if session_key is not None:
        extras.append("encrypted")
    extra_str = f" [{', '.join(extras)}]" if extras else ""
    print(f">> Sending ({len(audio)/DEVICE_SR:.2f}s, mode={mode}, {len(frames)} frame(s)){extra_str}: "
          f"{display_text if display_text is not None else text}", file=sys.stderr)

    with _prevent_display_sleep():
        sd.play(audio.astype(np.float32), samplerate=DEVICE_SR, device=device, blocking=True)
    print(">> Sent.", file=sys.stderr)


class Listener:
    def __init__(self, mode: str = "phone", parity_bytes: int = 10, my_id: int | None = None,
                 session_key: bytes | None = None, device: int | None = None,
                 on_message=None, candidates: list[tuple[str, int]] | None = None):
        self.mode = mode
        self.parity_bytes = parity_bytes
        self.my_id = my_id
        self.session_key = session_key
        self.device = device
        self.on_message = on_message  # optional callback(MessageResult), called on full-message completion
        self.candidates = candidates or [(mode, parity_bytes)]
        self._preamble_ref = modem.generate_preamble(sr=DEVICE_SR)

        self._buffer = np.zeros(0, dtype=np.float64)
        self._lock = threading.Lock()
        self._processed_until = 0   # confirmed/resolved up to here -- safe to trim before this
        self._scan_position = 0     # scanned-with-nothing-found up to here -- see _try_decode
        self._pending_since: float | None = None
        self._reassembler = message.MessageReassembler()

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        chunk = indata[:, 0].astype(np.float64)
        with self._lock:
            self._buffer = np.concatenate([self._buffer, chunk])
            headroom = DEVICE_SR * 2  # keep 2s before processed_until as safety margin
            if self._processed_until > headroom:
                trim = self._processed_until - headroom
                self._buffer = self._buffer[trim:]
                self._processed_until -= trim

    def _finish(self, abs_offset: int, exact_end: int | None = None) -> None:
        """Marks this preamble occurrence as resolved (success, definitive
        reject, or timeout) so the next scan looks past it rather than
        re-detecting the same chirp forever. `exact_end` (from
        frame_wire_length, in samples) jumps precisely past this frame's
        real content when known -- falling back to just skipping the
        preamble itself risks the next bounded search window containing
        more than one subsequent preamble (see protocol.frame_wire_length's
        docstring)."""
        preamble_end = abs_offset + int(modem.PREAMBLE_DURATION_S * DEVICE_SR)
        if exact_end is not None:
            # Back off a bit before the calculated boundary -- real audio
            # doesn't preserve exact sample counts, so landing precisely on
            # it can overshoot into the next preamble and weaken its
            # correlation (see cli.py's PREAMBLE_BACKOFF_S for the full story).
            next_start = max(preamble_end, exact_end - int(PREAMBLE_BACKOFF_S * DEVICE_SR))
        else:
            next_start = preamble_end
        with self._lock:
            self._processed_until = max(self._processed_until, next_start)
            self._scan_position = max(self._scan_position, self._processed_until)
        self._pending_since = None

    def _try_candidate(self, payload_start: int, cand_mode: str,
                        cand_parity: int) -> tuple[ParseResult | None, int | None]:
        profile = modem.MODES[cand_mode]
        step_n = int((profile["symbol_duration_s"] + profile["guard_s"]) * DEVICE_SR)
        with self._lock:
            available = len(self._buffer) - payload_start
        n_symbols = min(MAX_SYMBOLS, max(available // step_n, 0))
        if n_symbols < 4:
            return None, None

        with self._lock:
            payload_audio = self._buffer[payload_start:payload_start + n_symbols * step_n].copy()
        detections = modem.demodulate(payload_audio, n_symbols, symbol_duration_s=profile["symbol_duration_s"],
                                       guard_s=profile["guard_s"], sr=DEVICE_SR)
        symbols = [d.symbol for d in detections]
        n_bytes = (len(symbols) * modem.BITS_PER_SYMBOL) // 8
        frame_codes = list(modem.symbols_to_bytes(symbols, n_bytes))
        result = parse_frame(frame_codes, cand_parity, my_id=self.my_id, session_key=self.session_key)

        exact_end = None
        consumed_codes = frame_wire_length(frame_codes)
        if consumed_codes is not None:
            consumed_symbols = -(-consumed_codes * 8 // 6)  # ceil division
            exact_end = payload_start + consumed_symbols * step_n
        return result, exact_end

    def _try_decode(self) -> None:
        search_window_n = SEARCH_WINDOW_S * DEVICE_SR
        preamble_len_n = int(modem.PREAMBLE_DURATION_S * DEVICE_SR)
        with self._lock:
            scan_start = max(self._processed_until, self._scan_position)
            tail = self._buffer[scan_start:scan_start + search_window_n]

        found = modem.find_preamble(tail, reference=self._preamble_ref)
        if found is None:
            # Nothing in this window -- advance the scan position so the
            # next poll looks at newly-arrived audio instead of re-scanning
            # the same stale window forever. Without this, a fixed window
            # anchored at processed_until never moves without a resolved
            # detection, so audio arriving beyond it would be silently
            # unreachable (a real bug this fixes -- e.g. the user just
            # taking a while to type their next message). Leave a
            # preamble-length overlap so a chirp straddling this window's
            # trailing edge isn't split across two separate scans.
            if len(tail) > preamble_len_n:
                with self._lock:
                    self._scan_position = max(self._scan_position, scan_start + len(tail) - preamble_len_n)
            return
        offset, score = found
        abs_offset = scan_start + offset

        if self._pending_since is None:
            self._pending_since = time.time()
            print(f"\n[sync] preamble detected (score={score:.2f})...", file=sys.stderr)

        payload_start = abs_offset + int(modem.PREAMBLE_GUARD_S * DEVICE_SR)

        result, exact_end, last_result, last_exact_end = None, None, None, None
        for cand_mode, cand_parity in self.candidates:
            r, e = self._try_candidate(payload_start, cand_mode, cand_parity)
            last_result, last_exact_end = r, e
            if r is not None and r.ok:
                result, exact_end = r, e
                break
        if result is None:
            result, exact_end = last_result, last_exact_end

        timed_out = time.time() - self._pending_since > PREAMBLE_TIMEOUT_S

        if result is not None and result.ok:
            more = " (more frames coming...)" if result.more_frames else ""
            print(f"[frame seq={result.seq}] dest_id={result.dest_id}: {result.text!r}{more}",
                  file=sys.stderr)
            msg_result = self._reassembler.add(result)
            if msg_result.ok:
                print(f"[received, {msg_result.frames_received} frame(s)]: {msg_result.text}",
                      file=sys.stderr)
                if self.on_message is not None:
                    self.on_message(msg_result)
                else:
                    print(msg_result.text)
                self._reassembler.reset()
            self._finish(abs_offset, exact_end)
        elif result is not None and result.reason and "not addressed to me" in result.reason:
            print(f"[ignored] frame for dest_id={result.dest_id} (not me)", file=sys.stderr)
            self._finish(abs_offset, exact_end)
        elif timed_out:
            reason = result.reason if result is not None else "insufficient audio captured"
            print(f"[sync] gave up on this preamble ({reason})", file=sys.stderr)
            self._finish(abs_offset, exact_end)
        # else: not resolved yet and still within timeout -- wait for more
        # audio and retry the same window (possibly larger) next poll.

    def run(self) -> None:
        import sounddevice as sd

        my_id_str = str(self.my_id) if self.my_id is not None else "any (broadcast + all addresses)"
        print(f"Listening (mode={self.mode}, my_id={my_id_str})... Ctrl+C to stop.", file=sys.stderr)
        with _prevent_display_sleep(), \
             sd.InputStream(samplerate=DEVICE_SR, channels=1, dtype="float32",
                             device=self.device, callback=self._audio_callback):
            try:
                while True:
                    self._try_decode()
                    time.sleep(POLL_INTERVAL_S)
            except KeyboardInterrupt:
                print("\nStopped.", file=sys.stderr)

    def start_background(self) -> "sd.InputStream":
        """Non-blocking variant of run(), for chat.py: starts the input
        stream and a polling thread, returns immediately. Call
        stop_background() (not just stream.stop()) to actually stop the
        polling thread and release the display-sleep-prevention assertion
        started here -- it's a daemon thread so it won't block process
        exit either way, but stop_background() ends it cleanly rather than
        leaving it spinning on stale buffer state until then."""
        import sounddevice as sd

        self._caffeinate = _prevent_display_sleep()
        self._caffeinate.__enter__()

        stream = sd.InputStream(samplerate=DEVICE_SR, channels=1, dtype="float32",
                                 device=self.device, callback=self._audio_callback)
        stream.start()

        self._stop_event = threading.Event()

        def _poll_loop():
            while not self._stop_event.is_set():
                self._try_decode()
                time.sleep(POLL_INTERVAL_S)

        thread = threading.Thread(target=_poll_loop, daemon=True)
        thread.start()
        return stream

    def stop_background(self) -> None:
        """Stops the polling thread started by start_background() and
        releases its display-sleep-prevention assertion."""
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if hasattr(self, "_caffeinate"):
            self._caffeinate.__exit__(None, None, None)


def listen(mode: str = "phone", parity_bytes: int = 10, my_id: int | None = None,
           session_key: bytes | None = None, device: int | None = None) -> None:
    Listener(mode, parity_bytes, my_id, session_key, device).run()
