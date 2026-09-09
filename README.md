# TextOverVoice

Text transport over voice-grade audio channels (phone calls) — designed to
survive a 300–3400Hz bandlimited channel, destructive AMR compression, and
phone-side AEC/noise suppression.

## Design

- **Modulation**: 2-of-16 dual-tone MFSK (`src/textovervoice/modem.py`) —
  one symbol = one tone from an 8-frequency low group + one from an
  8-frequency high group (64 combinations, 6 bits/symbol). An earlier
  8-simultaneous-tone design was dropped after real AMR-NB round-trips
  showed it produces more intermodulation distortion than signal at low
  bitrate; dual-tone stays clear of that failure mode.
- **Framing**: SOF/LENGTH/PAYLOAD/PARITY/CRC/EOF wire format
  (`protocol.py`), with byte-stuffing (`framing.py`) so literal data bytes
  can't be confused with reserved control codes.
- **UTF-8 handling**: non-ASCII characters are wrapped in explicit
  START/CONT/END boundary flags (`charset.py`) for resync after channel
  errors, with UTF-8's own self-describing lead-byte length as a fallback
  completion trigger.
- **Error correction**: Reed-Solomon (via `reedsolo`) + CRC-16 (`fec.py`) —
  RS corrects burst errors typical of codec/channel loss, CRC independently
  confirms the result is actually right.

Verified end-to-end through a real AMR-NB encode/decode pass (`ffmpeg` +
`libopencore_amrnb`) at the worst-case 4.75kbps bitrate: text survives
byte-exact with FEC; the same payload without FEC does not. See
`tools/codec_validation/realism_results.csv` for the full test matrix
(symbol accuracy, noise robustness, symbol-duration sensitivity, tandem
transcoding).

**Known gaps**: no preamble-based frame sync yet (decode assumes symbol 0
starts at sample 0 of the file — fine for the offline CLI, not for a live
capture), no live call audio I/O, no handshake/calibration state machine,
and tandem AMR transcoding (a call routed through multiple network legs)
degrades sharply at low bitrate (0%→28% symbol error over 1→3 passes) —
addressing that needs either the sync layer or adaptive FEC.

## Setup

```
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Codec validation scripts additionally need `ffmpeg-full` (for the
`libopencore_amrnb` encoder, which the stock `ffmpeg` Homebrew formula
doesn't include):

```
brew install ffmpeg-full
```

## Usage

```
./.venv/bin/textovervoice encode "Hello, 你好" out.wav
./.venv/bin/textovervoice decode out.wav
```

This is the offline modem/framing/FEC round-trip only (file in, file out) —
no live call audio I/O yet.

## Tests

```
./.venv/bin/pytest tests/                                    # framing/charset/fec/protocol logic
./.venv/bin/python3 tools/codec_validation/realism_suite.py   # full modem realism suite vs real AMR-NB
```

## Layout

```
src/textovervoice/       modem.py, framing.py, charset.py, fec.py, protocol.py, cli.py
tests/                    unit tests (no audio, fast)
tools/codec_validation/   AMR-NB round-trip test scripts + CSV results
```
