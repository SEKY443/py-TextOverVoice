# TextOverVoice

Text transport over voice-grade audio channels — cellular phone calls (AMR)
and VoIP calling apps like WhatsApp/Discord (Opus) alike — designed to
survive destructive codec compression and phone/WebRTC-side AEC/noise
suppression.

## Design

- **Modulation**: 2-of-16 dual-tone MFSK (`src/textovervoice/modem.py`) —
  one symbol = one tone from an 8-frequency low group + one from an
  8-frequency high group (64 combinations, 6 bits/symbol), inside the
  300–3400Hz voice band. An earlier 8-simultaneous-tone design was dropped
  after real AMR-NB round-trips showed it produces more intermodulation
  distortion than signal at low bitrate; dual-tone stays clear of that
  failure mode.
- **Timing profiles** (`modem.MODES`): `phone` (40ms/symbol, validated
  through real AMR-NB down to its worst 4.75kbps mode) and `fast_air`
  (20ms/symbol, ~2x faster, acoustic-only — validated on real speaker/mic
  hardware via a duration sweep, not guessed; not expected to survive AMR's
  20ms codec-frame quantization the way `phone` does).
- **Frame sync**: a linear chirp preamble, located via normalized
  cross-correlation (`modem.generate_preamble`/`find_preamble`/
  `modulate_frame`/`demodulate_frame`), so a receiver doesn't need to know
  where symbol 0 starts in a live capture. Verified on real hardware.
- **Multi-frame messages** (`message.py`): long text is split into multiple
  independently-preambled frames rather than sent as one giant frame. This
  exists because of a measured failure mode, not a theoretical one: sending
  the full GPLv3 text (35KB) as a single frame through real AMR-NB at
  4.75kbps produced a hard timing-drift "cliff" partway through — a single
  preamble at the very start can't correct for drift accumulating over a
  30+ minute transmission. Splitting into shorter frames, each with its own
  fresh sync point, bounds how far drift can accumulate before it's
  corrected, and also bounds the blast radius of a corrupted frame header
  to just that frame instead of the whole message. The scanner that finds
  each frame's preamble in a long file was itself a source of bugs before
  it was fixed: `modem.find_preamble` returns the single globally-strongest
  correlation match in whatever span it's given, so a search window wide
  enough to contain more than one frame's preamble could jump straight to
  a distant one and skip nearer frames out of sequence (caught directly by
  the GPLv3 test: decode found frame seq=37 before seq=0). Fixed by
  scanning in small, non-overlapping-content windows that structurally
  can't contain two preambles, advancing incrementally rather than
  searching one large span up front — verified on the full GPLv3 text:
  all 44 frames now found in exact sequence, byte-exact reassembly at
  AMR-NB 12.2kbps.
- **Framing**: SOF/DEST_ID/FLAGS/SEQ/LENGTH/PAYLOAD/PARITY/CRC wire format
  (`protocol.py`), with byte-stuffing (`framing.py`) so literal data bytes
  can't be confused with reserved control codes.
- **Addressing**: an optional DEST_ID field (`protocol.build_frame`'s
  `dest_id`) lets a receiver (`my_id`) ignore frames meant for someone else
  without running FEC/CRC on them at all — default is broadcast (everyone
  accepts).
- **Encryption**: optional X25519 key agreement + ChaCha20-Poly1305 AEAD
  (`crypto.py`), not RSA-style direct encryption — a raw asymmetric
  ciphertext (190+ bytes minimum) would dominate transmission time on this
  channel, so the asymmetric step only derives a shared session key once,
  and a fast symmetric cipher handles each message (28 bytes of fixed
  overhead: 12-byte nonce + 16-byte auth tag). Tamper detection and
  wrong-key rejection are both verified in tests. No built-in peer identity
  verification (no PKI, no Signal-style safety-number check) — a live key
  exchange over an open channel is in principle interceptable; treat this
  as "encrypted in transit," not "provably secure against an active
  attacker," unless you add out-of-band key verification yourself.
- **UTF-8 handling**: non-ASCII characters are wrapped in explicit
  START/CONT/END boundary flags (`charset.py`) for resync after channel
  errors, with UTF-8's own self-describing lead-byte length as a fallback
  completion trigger.
- **Dictionary compression** (`dictionary.py`): common English words (5+
  letters, chosen by *expected character savings* — frequency × (length−4)
  via `wordfreq`, not raw frequency, since short common words like "the"
  wouldn't save anything) substitute for a 4-code dictionary hit instead of
  one code per character. ~35% frame-size reduction measured on ordinary
  prose. Strictly optional/additive — anything not in the dictionary falls
  back to per-character encoding unchanged.
- **Error correction**: Reed-Solomon (via `reedsolo`) + CRC-16 (`fec.py`) —
  RS corrects burst errors typical of codec/channel loss (chunked to handle
  payloads past GF(256)'s 255-symbol codeword limit), CRC independently
  confirms the result is actually right.
- **Live two-way conversation with adaptive recalibration** (`chat.py`):
  an experiment, not a formal protocol. Every chat message is tagged with a
  short id; the peer auto-replies with a lightweight ACK on successful
  decode. The sender tracks a rolling window of ACK/timeout outcomes and
  "recalibrates" — stepping to a more robust `(mode, parity_bytes,
  max_frame_chars)` combination when the recent failure rate is high, and
  back to a faster one once the channel's been clean for a while. Since the
  receiver has no other way to know the sender changed settings, the
  listener just tries every reachable `(mode, parity_bytes)` combination
  per received frame (cheap — it's re-parsing the same short captured audio
  a few times) instead of requiring a formal negotiation handshake.
  Verified working end-to-end on real hardware, both directions, using two
  `AdaptiveChat` instances in one process (slower than real deployment due
  to GIL contention between two listener threads, but correctness-proving).
- **A hard-won, non-obvious fix that matters for any live audio use**:
  `sd.play()`/`OutputStream.write()` can hang *indefinitely* (not just
  slowly) if macOS puts the audio subsystem into its
  "DarkWakeSilenceBuffers" idle power state, which was observed kicking in
  after ~30s of no keyboard/mouse activity — exactly the situation during
  an unattended live demo or a long-running `listen`/`chat` session.
  Confirmed via `log show --predicate 'process == "coreaudiod"'` during a
  live hang, and fixed by asserting `caffeinate -u` (not a plain
  `caffeinate`, which only blocks *system* sleep, not this). `send`,
  `listen`, and `chat` all do this automatically now
  (`live._prevent_display_sleep`); it's macOS-only and a no-op elsewhere.

### Validated against

- **AMR-NB** (cellular): real `ffmpeg` + `libopencore_amrnb` round-trips
  across all 8 bitrate modes. Byte-exact recovery with FEC at the worst
  4.75kbps mode; without FEC it fails. See
  `tools/codec_validation/realism_results.csv` (symbol accuracy, noise
  robustness, symbol-duration sensitivity, tandem transcoding).
- **Opus** (WhatsApp/Discord-style VoIP): real `ffmpeg` + `libopus`
  round-trips at approximated WhatsApp-like (16kHz/16-24kbps) and
  Discord-like (48kHz/32-64kbps) operating points. 8/8 full-protocol tests
  passed (ASCII + CJK), with much lower symbol error rates than AMR-NB's
  worst case (0-1.6% vs. up to 28%) — Opus is measurably less destructive
  to this signal than AMR-NB. See `tools/codec_validation/opus_survivability.py`.
  **Not yet tested**: the WebRTC-style noise suppression/AEC layer these
  apps also run on top of Opus, which specifically targets steady tonal
  content and could be a real risk independent of codec compression.
- **Real acoustic hardware** (speaker → room air → mic): `phone`/`fast_air`
  single-frame send/decode, the full two-way `chat` flow (both directions,
  as two `AdaptiveChat` instances in one process), and addressing across
  two genuinely separate OS processes (a `send` and a `listen` process — a
  wrongly-addressed listener correctly ignores a frame without attempting
  to decode it) all verified working. Reliability is visibly
  room-noise-dependent, as expected for any acoustic system. See
  live.py's docstring for a hard-won, unrelated finding from this testing:
  macOS can silently hang audio playback after ~30s of no keyboard/mouse
  activity (a power-saving state, not a bug in this project), which
  `send`/`listen`/`chat` now work around automatically.
- **Scale**: the full GPLv3 license text (35,148 chars), split into 44
  frames, round-trips byte-exact through real AMR-NB at 12.2kbps (best
  bitrate) with every frame found in correct sequence. At AMR's worst
  bitrate (4.75kbps), 29/44 frames decode correctly and 15 fail on genuine
  per-frame FEC/channel errors (not a scanning bug — confirmed by the
  scanner finding and correctly sequencing all 44 preambles before any
  per-frame decode is attempted); recovering the rest needs either
  stronger FEC or frame retransmission, neither implemented yet.

### Comparison to ggwave

ggwave (a similar data-over-sound library) has higher raw throughput at its
fastest setting, but its default protocols use 2013-6098Hz — 66% of that
energy sits outside the 300-3400Hz telephone band. Tested empirically: it
failed to decode in all 6 AMR-NB round-trip trials, including at AMR's best
quality setting, and even failed from 8kHz resampling alone with no
compression involved. It's built for open-air device-to-device pairing,
not for fitting inside a phone call's channel — a different problem than
this project targets. With `fast_air` mode + dictionary compression,
TextOverVoice's open-air throughput on longer text (30+ chars/sec) matches
or beats ggwave's fastest mode too.

**Known gaps**: no handshake/calibration wire-protocol (chat.py's
recalibration is an application-level convention, not the opcodes reserved
in `codes.py`), no frame retransmission (a lost frame in a multi-frame
message is just gone — recalibration can prevent future losses but can't
recover one already missed), and the LENGTH/marker header bytes in a frame
still aren't FEC-protected (a corrupted header byte fails that frame
outright even if its payload was recoverable) — multi-frame splitting
bounds the damage to one frame instead of a whole message, but doesn't
eliminate it.

## Setup

```
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Codec validation scripts need `ffmpeg-full` (for the `libopencore_amrnb`
encoder and `libopus`; the stock `ffmpeg` Homebrew formula doesn't include
the AMR encoder):

```
brew install ffmpeg-full
```

Live audio (`send`/`listen`/`chat`) needs:

```
./.venv/bin/pip install -e ".[acoustic-test]"
```

## Usage

Offline, file-based (no live audio needed):

```
./.venv/bin/textovervoice encode "Hello, 你好" out.wav              # phone mode (default)
./.venv/bin/textovervoice encode "Hello, 你好" out.wav --mode fast_air
./.venv/bin/textovervoice decode out.wav
```

Live, through your actual speaker/microphone:

```
./.venv/bin/textovervoice listen                       # one-shot receiver
./.venv/bin/textovervoice send "Hello!"                 # one-shot sender

# two-way conversation with adaptive recalibration (run one per device)
./.venv/bin/textovervoice chat --my-id 1 --peer-id 2
./.venv/bin/textovervoice chat --my-id 2 --peer-id 1
```

Addressing and encryption (opt-in, composable with any of the above):

```
./.venv/bin/textovervoice keygen alice.key alice.pub
./.venv/bin/textovervoice send "For device 7 only" --dest-id 7
./.venv/bin/textovervoice send "Secret" --dest-id 7 --my-privkey alice.key --peer-pubkey bob.pub
```

## Tests

```
./.venv/bin/pytest tests/                                              # framing/charset/fec/protocol/sync/crypto/message/chat logic
./.venv/bin/python3 tools/codec_validation/realism_suite.py            # full modem realism suite vs real AMR-NB
./.venv/bin/python3 tools/codec_validation/opus_survivability.py       # vs real Opus (WhatsApp/Discord-style)
./.venv/bin/python3 tools/codec_validation/acoustic_loopback_test.py   # real speaker/mic loopback
./.venv/bin/python3 tools/codec_validation/acoustic_duration_sweep.py  # real-hardware timing-floor sweep
```

## Layout

```
src/textovervoice/       modem.py, framing.py, charset.py, dictionary.py, fec.py,
                          crypto.py, protocol.py, message.py, live.py, chat.py, cli.py
src/textovervoice/data/  generated word-frequency dictionaries (tools/build_dictionary.py)
tests/                    unit tests (no audio, fast)
tools/                    build_dictionary.py
tools/codec_validation/   AMR-NB / Opus / acoustic round-trip test scripts + CSV results
```
