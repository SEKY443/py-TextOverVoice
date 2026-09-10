# TextOverVoice

Text transport over voice-grade audio channels — cellular phone calls (AMR)
and VoIP calling apps like WhatsApp/Discord (Opus) alike — designed to
survive destructive codec compression and phone/WebRTC-side AEC/noise
suppression.

## Project status

**This is a prototype, not a production implementation.** It exists to
validate the underlying approach (modulation scheme, FEC strategy, framing)
against real codecs and real hardware before committing to a production
build. The current codebase is Python, chosen for fast iteration during
this experimentation phase (numpy/scipy for DSP, quick to reshape as the
design changed); a rewrite in another language is planned once the design
has settled, for the performance and deployment characteristics a
prototype-stage Python implementation isn't optimized for. Treat the
numbers and findings in this document as validated *results*, not as
commitments about the shape of the eventual production code.

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

## Experimental results

All numbers below come from real codec round-trips (`ffmpeg` +
`libopencore_amrnb`/`libopus`) or real speaker/microphone hardware — not
simulation of channel behavior. Raw data and the scripts that produced it
are in `tools/codec_validation/`.

### AMR-NB (cellular) — symbol accuracy across all 8 bitrate modes

| Test | 4.75kbps (worst) | 7.40kbps (mid) | 12.2kbps (best) |
|---|---|---|---|
| Isolated symbol accuracy (64 symbols) | 1.6% SER | 0% SER | 0% SER |
| Full-alphabet sequence (64 symbols, one clip) | 4.7% SER | 0% SER | 0% SER |
| Noise robustness (AWGN, worst tested SNR 0dB) | 6.2% SER | — | 0% SER |
| Tandem transcoding, 1 → 3 AMR passes | 0% → 12.5% → 28.1% SER | — | 0% throughout |
| End-to-end text, no FEC | fails (byte errors) | passes | passes |
| End-to-end text, with RS(n,10) FEC | exact recovery | exact recovery | exact recovery |

Symbol duration matters more than the "align to the 20ms codec frame" rule
alone suggested: at 4.75kbps, a 20ms symbol (exactly one AMR frame) still
had **28.1% SER**; 40ms+ dropped to ≤6% regardless of exact alignment. This
is why `phone` mode uses 40ms.

### Opus (WhatsApp/Discord-style VoIP)

| Profile | Sample rate / bitrate | Symbol error rate | Full-protocol (ASCII+CJK) |
|---|---|---|---|
| WhatsApp-like | 16kHz / 16kbps | 1.6% (1/64) | 8/8 pass |
| WhatsApp-like HQ | 16kHz / 24kbps | 0% | pass |
| Discord-like | 48kHz / 64kbps | 0% | pass |
| Discord-like (low) | 48kHz / 32kbps | 0% | pass |

Opus is measurably less destructive to this signal than AMR-NB's worst
case (0–1.6% vs. up to 28% SER). **Not tested**: the WebRTC-style noise
suppression/AEC layer these apps also run on top of Opus, which
specifically targets steady tonal content and could be a real risk
independent of codec compression alone.

### Throughput (chars/sec), measured via the real modem/protocol code

| Configuration | 19 chars | 68 chars | 251 chars |
|---|---|---|---|
| `phone`, plain + dictionary | 6.67 | 16.00 | 15.94 |
| `phone`, plain, no dictionary | 6.23 | 10.97 | 13.18 |
| `phone`, encrypted | 3.80 | 7.95 | 11.23 |
| `fast_air`, plain + dictionary | 12.26 | 30.22 | 31.38 |
| `fast_air`, plain, no dictionary | 11.52 | 21.09 | 26.01 |
| `fast_air`, encrypted | 7.04 | 15.11 | 22.11 |
| ggwave AUDIBLE_NORMAL | 9.95 | — | — |
| ggwave AUDIBLE_FAST | 14.36 | — | — |
| ggwave AUDIBLE_FASTEST | 25.81 | — | — |

Dictionary compression's gain scales with how much of the text is long
common English words: +43% throughput on the 68-char sample (dense with
words like "government", "information"), only +6% on the 19-char greeting.
Encryption costs 30–45% throughput (ciphertext isn't dictionary-compressible,
plus 28 bytes of fixed AEAD overhead) — `fast_air` recovers most of that
back. See `tools/codec_validation/final_benchmark.py`.

### Scale test: full GPLv3 license text (35,148 characters, 44 frames)

| Bitrate | Result |
|---|---|
| AMR-NB 12.2kbps (best) | **44/44 frames, byte-exact reassembly** |
| AMR-NB 4.75kbps (worst) | 29/44 frames decode correctly; 15 fail on genuine per-frame FEC/channel errors |

The 4.75kbps failures are confirmed to be real channel-error limits, not a
scanning bug: the preamble scanner correctly finds and sequences all 44
frames before any per-frame decode is even attempted. Recovering the
remaining 15 needs stronger FEC or frame retransmission (neither
implemented yet). This test is also what originally exposed two real bugs
that are now fixed: a Reed-Solomon chunking error past ~245 bytes, and a
preamble scanner that could jump to a distant, more-strongly-correlated
frame instead of the nearest one.

### Real acoustic hardware (speaker → room air → mic)

`phone`/`fast_air` single-frame send/decode, the full two-way `chat` flow
(both directions, as two `AdaptiveChat` instances in one process), and
addressing across two genuinely separate OS processes (a wrongly-addressed
listener correctly ignores a frame without attempting to decode it) all
verified working. A real-hardware duration sweep found `fast_air`'s 20ms
symbols perform statistically indistinguishably from `phone`'s 40ms in
clean acoustic conditions (mean SER ~1–5% either way over repeated trials)
but degrade sharply below ~15ms (SER jumps to 8–15%) and catastrophically
below 10ms (40–90%) — this is what set `fast_air`'s timing, not a guess.

One hard-won, unrelated finding from this testing: macOS can silently hang
audio playback indefinitely after ~30s of no keyboard/mouse activity (a
power-saving state, not a bug in this project, confirmed via `log show
--predicate 'process == "coreaudiod"'` during a live hang). `send`,
`listen`, and `chat` now prevent this automatically.

### Comparison to ggwave

ggwave (a similar data-over-sound library) has higher raw throughput at its
fastest setting, but its default protocols use 2013-6098Hz — 66% of that
energy sits outside the 300-3400Hz telephone band. Tested empirically: it
failed to decode in **all 6** AMR-NB round-trip trials (3 protocols × 2
bitrates), including at AMR's best quality setting, and even failed from
plain 8kHz resampling alone with no compression involved. It's built for
open-air device-to-device pairing, not for fitting inside a phone call's
channel — a different problem than this project targets. With `fast_air`
mode + dictionary compression, TextOverVoice's open-air throughput on
longer text (30+ chars/sec) matches or beats ggwave's fastest mode too
(see throughput table above).

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

## License

MIT — see [LICENSE](LICENSE).
