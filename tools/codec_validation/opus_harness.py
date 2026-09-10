"""Shared Opus round-trip helper (ffmpeg + libopus), mirroring amr_harness.py
but for VoIP-style calling apps (WhatsApp, Discord, etc.) instead of
cellular AMR-NB.

Unlike AMR-NB, Opus is not fixed at 8kHz -- WebRTC-based apps typically run
it wideband-to-fullband (16-48kHz) with adaptive bitrate. Neither this
project nor its author has access to WhatsApp/Discord's actual encoder
configuration, so these are realistic *approximations* based on commonly
cited operating points, not confirmed exact parameters:

  - WhatsApp-like: 16kHz (wideband), ~16-24kbps voice-optimized
  - Discord-like: 48kHz (fullband), ~64kbps default

This tests Opus compression only -- it does NOT include the WebRTC-style
noise suppression / AEC layer these apps also run, which is a separate and
likely more aggressive risk for tone-based signals (see modem docstrings'
notes on AEC suppressing steady tones). No good offline way to simulate
that layer without the app's actual audio pipeline.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from amr_harness import FFMPEG  # reuse the same ffmpeg-full binary discovery

# (label, sample_rate, bitrate) -- approximations, see module docstring
OPUS_PROFILES = {
    "whatsapp_like": (16000, "16k"),
    "whatsapp_like_hq": (16000, "24k"),
    "discord_like": (48000, "64k"),
    "discord_like_low": (48000, "32k"),
}


def roundtrip_opus(x: np.ndarray, profile: str, tmpdir: Path, source_sr: int) -> np.ndarray:
    target_sr, bitrate = OPUS_PROFILES[profile]

    wav_in = tmpdir / "opus_in.wav"
    opus = tmpdir / "opus.opus"
    wav_out = tmpdir / "opus_out.wav"

    wavfile.write(wav_in, source_sr, np.clip(x * 32767, -32768, 32767).astype(np.int16))

    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", str(wav_in),
         "-ar", str(target_sr), "-ac", "1", "-c:a", "libopus", "-b:a", bitrate, str(opus)],
        check=True,
    )
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", str(opus),
         "-ar", str(source_sr), str(wav_out)],
        check=True,
    )

    sr_out, y = wavfile.read(wav_out)
    assert sr_out == source_sr
    return y.astype(np.float64) / 32767.0
