"""Shared AMR-NB round-trip helper (ffmpeg + libopencore_amrnb) used by all
codec-validation scripts in this directory. Kept in one place so the
encode/decode subprocess plumbing isn't duplicated per test script.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

FFMPEG_CANDIDATES = [
    "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
    "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
]

AMR_NB_BITRATES = ["4.75k", "5.15k", "5.90k", "6.70k", "7.40k", "7.95k", "10.2k", "12.2k"]


def find_ffmpeg() -> str:
    for candidate in FFMPEG_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    found = subprocess.run(["which", "ffmpeg-full"], capture_output=True, text=True).stdout.strip()
    if found:
        return found
    raise RuntimeError(
        "ffmpeg-full not found (need libopencore_amrnb encoder support). "
        "Install with: brew install ffmpeg-full"
    )


FFMPEG = find_ffmpeg()


def roundtrip_amr(x: np.ndarray, bitrate: str, tmpdir: Path, sr: int = 8000,
                   passes: int = 1) -> np.ndarray:
    """Encode+decode through AMR-NB `passes` times in a row (passes > 1
    simulates tandem transcoding across multiple network legs, which happens
    whenever a call isn't routed transcoder-free end to end).
    """
    wav_in = tmpdir / "hop_in.wav"
    amr = tmpdir / "hop.amr"
    wav_out = tmpdir / "hop_out.wav"

    y = x
    for _ in range(passes):
        wavfile.write(wav_in, sr, np.clip(y * 32767, -32768, 32767).astype(np.int16))
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(wav_in),
             "-ar", str(sr), "-ac", "1", "-b:a", bitrate, str(amr)],
            check=True,
        )
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(amr), str(wav_out)],
            check=True,
        )
        sr_out, y = wavfile.read(wav_out)
        assert sr_out == sr
        y = y.astype(np.float64) / 32767.0

    return y
