"""Records raw microphone audio for a fixed duration while the caller plays
a known message on the same machine, then saves it to a WAV file so it can
be decoded offline with decode_48k.py (an exhaustive, non-real-time
scanner running at the same 48kHz live.py's Listener uses) -- isolating
whether a live acoustic test's frame loss comes from the real-time
Listener's incremental processing or from the acoustic channel itself
(genuinely mangled tones). Run this concurrently with `textovervoice
send`, not instead of it.

This is how a real, two-bug root cause was actually found (see README's
"Real acoustic hardware" section): a raw recording of a transmission that
the live Listener had only decoded 3-4 of 17 frames from decoded 17/17,
byte-exact, offline -- proving the channel and scanner were both fine, and
pointing squarely at live.py's own real-time processing instead of
acoustics/hardware/background noise (all of which had been suspected
first and ruled out this way).
"""
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from textovervoice import live  # noqa: E402  (for DEVICE_SR -- record at the SAME
                                  # rate live.py's Listener actually captures at,
                                  # 48000Hz, not modem.SR's 8000Hz; those go through
                                  # different CoreAudio resampling paths and aren't
                                  # equivalent for diagnosing a live-only failure)


def main() -> None:
    duration_s = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/recorded_live.wav"
    sr = live.DEVICE_SR
    print(f"Recording {duration_s}s from default input at {sr}Hz to {out_path} ...", file=sys.stderr)
    audio = sd.rec(int(duration_s * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio = audio[:, 0]
    wavfile.write(out_path, sr, (audio * 32767).astype(np.int16))
    print(f"Saved {len(audio)/sr:.2f}s of audio.", file=sys.stderr)


if __name__ == "__main__":
    main()
