# TextOverVoice

Text transport over voice-grade audio channels — cellular calls (AMR) and
VoIP apps like WhatsApp/Discord (Opus) — designed to survive destructive
codec compression and phone/WebRTC-side AEC/noise suppression.

## Project status

**Prototype, not production.** Exists to validate the approach (modulation,
FEC, framing) against real codecs and hardware before committing to a
production build. Python was chosen for fast iteration (numpy/scipy for
DSP); a rewrite in another language is planned once the design settles.
Treat the numbers here as validated *results*, not commitments about the
eventual production code's shape.

## Design

- **Modulation**: 2-of-16 dual-tone MFSK (`modem.py`) — one symbol = one
  low-group tone + one high-group tone (64 combos, 6 bits/symbol), inside
  the 300–3400Hz voice band. An earlier 8-simultaneous-tone design was
  dropped after real AMR-NB tests showed too much intermodulation
  distortion at low bitrate.
- **Timing profiles** (`modem.MODES`): `phone` (40ms/symbol, AMR-NB
  validated down to 4.75kbps) and `fast_air` (20ms/symbol, ~2x faster,
  acoustic-only — not expected to survive AMR's 20ms frame quantization).
- **Frame sync**: a linear chirp preamble located via normalized
  cross-correlation, so a receiver doesn't need to know where symbol 0
  starts in a live capture.
- **Multi-frame messages** (`message.py`): long text splits into multiple
  independently-preambled frames instead of one giant frame, so timing
  drift and header corruption can't compound across the whole message
  (measured, not theoretical — see GPLv3 scale test below).
- **Framing**: SOF/DEST_ID/FLAGS/SEQ/LENGTH/PAYLOAD/PARITY/CRC wire format
  (`protocol.py`), with byte-stuffing so data bytes can't collide with
  control codes.
- **Addressing**: an optional DEST_ID lets a receiver ignore frames meant
  for someone else without running FEC/CRC on them — default is broadcast.
- **Encryption**: X25519 key agreement + ChaCha20-Poly1305 AEAD
  (`crypto.py`) rather than direct RSA — one asymmetric handshake derives
  a session key, then a cheap symmetric cipher handles each message (28
  bytes fixed overhead). No peer identity verification (no PKI/safety
  numbers) — "encrypted in transit," not proof against an active MITM,
  unless you add out-of-band key verification yourself.
- **Full UTF-8 support**: any codepoint round-trips independently via
  explicit START/CONT/END boundary flags (`charset.py`) plus UTF-8's own
  lead-byte length as a fallback, so multi-codepoint sequences (ZWJ emoji,
  flags, combining marks) work by construction. See the extreme UTF-8 test
  below for a measured stress test.
- **Dictionary compression** (`dictionary.py`): common English words (5+
  letters, ranked by frequency × expected character savings) substitute
  for a 4-code hit instead of one code per character — ~35% frame-size
  reduction on ordinary prose. Optional/additive.
- **Error correction**: Reed-Solomon + CRC-16 (`fec.py`) — RS corrects
  burst errors, CRC independently confirms the result.
- **Live two-way chat with adaptive recalibration** (`chat.py`): an
  experiment, not a formal protocol. ACKs drive a rolling failure-rate
  window that steps the sender to a more (or less) robust
  `(mode, parity_bytes, max_frame_chars)` combo; the receiver just tries
  every reachable combo per frame instead of needing a negotiation
  handshake. Verified end-to-end on real hardware.
- **Hard-won fix for any live audio use**: `sd.play()` can hang
  *indefinitely* if macOS puts audio into its "DarkWakeSilenceBuffers"
  idle state after ~30s of no keyboard/mouse activity. Fixed by asserting
  `caffeinate -u` automatically in `send`/`listen`/`chat`
  (macOS-only, no-op elsewhere).

## Experimental results

All numbers come from real codec round-trips (`ffmpeg` +
`libopencore_amrnb`/`libopus`) or real speaker/microphone hardware, not
simulation. Raw data and scripts are in `tools/codec_validation/`.

### AMR-NB (cellular) — symbol accuracy across all 8 bitrate modes

| Test | 4.75kbps (worst) | 7.40kbps (mid) | 12.2kbps (best) |
|---|---|---|---|
| Isolated symbol accuracy (64 symbols) | 1.6% SER | 0% SER | 0% SER |
| Full-alphabet sequence (64 symbols, one clip) | 4.7% SER | 0% SER | 0% SER |
| Noise robustness (AWGN, worst tested SNR 0dB) | 6.2% SER | — | 0% SER |
| Tandem transcoding, 1 → 3 AMR passes | 0% → 12.5% → 28.1% SER | — | 0% throughout |
| End-to-end text, no FEC | fails (byte errors) | passes | passes |
| End-to-end text, with RS(n,10) FEC | exact recovery | exact recovery | exact recovery |

At 4.75kbps, a 20ms symbol (one AMR frame exactly) still had 28.1% SER;
40ms+ dropped to ≤6% regardless of alignment — why `phone` mode uses 40ms.

### Opus (WhatsApp/Discord-style VoIP)

| Profile | Sample rate / bitrate | Symbol error rate | Full-protocol (ASCII+CJK) |
|---|---|---|---|
| WhatsApp-like | 16kHz / 16kbps | 1.6% (1/64) | 8/8 pass |
| WhatsApp-like HQ | 16kHz / 24kbps | 0% | pass |
| Discord-like | 48kHz / 64kbps | 0% | pass |
| Discord-like (low) | 48kHz / 32kbps | 0% | pass |

Opus is measurably less destructive than AMR-NB's worst case. **Not
tested**: WebRTC-style noise suppression/AEC on top of Opus, which
specifically targets steady tonal content.

### Throughput (chars/sec), real modem/protocol code

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

Dictionary compression's gain tracks how much text is long common words
(+43% on a word-dense 68-char sample, +6% on a short greeting). Encryption
costs 30–45% throughput (ciphertext isn't compressible, plus fixed AEAD
overhead) — `fast_air` recovers most of that back.

### Scale test: full GPLv3 license text (35,148 characters, 44 frames)

| Bitrate | Result |
|---|---|
| AMR-NB 12.2kbps (best) | **44/44 frames, byte-exact reassembly** |
| AMR-NB 4.75kbps (worst) | 29/44 frames decode correctly; 15 fail on genuine per-frame FEC/channel errors |

The 4.75kbps failures are real channel-error limits, not a scanning bug —
the preamble scanner correctly sequences all 44 frames before any per-frame
decode is attempted. This test also exposed and fixed two real bugs: a
Reed-Solomon chunking error past ~245 bytes, and a preamble scanner that
could jump to a distant, more-strongly-correlated frame instead of the
nearest one (`find_preamble` returns the single globally-strongest
correlation match in whatever span it's given, so too-wide a search window
could skip frames out of sequence — fixed by scanning small windows that
structurally can't contain two preambles).

### Extreme UTF-8 test

One 337-codepoint / 517-byte string combining CJK, right-to-left Arabic,
Cyrillic, standalone combining diacritics, currency/math symbols, a
4-codepoint ZWJ family emoji, a skin-tone modifier, flag sequences, rare
4-byte CJK Extension-B characters, and a bare zero-width joiner.

| Test | Result |
|---|---|
| Clean digital round-trip | exact match |
| Real AMR-NB 4.75kbps (worst bitrate) round-trip | exact match |

A shorter variant was sent **live through this machine's actual speaker
and microphone** and received byte-exact. The full 337-character version
(67s single-frame) exceeds the live listener's 8s `PREAMBLE_TIMEOUT_S` —
a real live-listening constraint, not a UTF-8 bug (no issue on the
file-based/AMR path, which has no timeout).

Chunking to work around that timeout (`--max-frame-chars`) surfaced three
more real bugs in live listening, all fixed:

1. **A second instance of the scanner window-sizing bug** above — the
   fix had been sized against long `phone`-mode frames, but `fast_air`'s
   short frames pack several preamble cycles into the same window
   (reproduced with zero channel noise, only 5/12 frames decoded). Fixed
   by shrinking the window further, below any realistic frame length.
2. **O(N²) buffer growth in the live listener.** `Listener._audio_callback`
   rebuilt its *entire* capture buffer on every audio callback, so
   real-time capture fell further behind the longer a session ran (59.5%
   CPU mid-session, down to 1–3% after batching appends). Found by
   recording raw mic audio in parallel and decoding it offline at
   100% — proving the acoustic channel was never the problem.
3. **A live-only race in the incremental scanner** — a buffer read
   truncated only because real-time audio hadn't fully arrived yet could
   be mistaken for "nothing here" and permanently skip a preamble
   still mid-transmission. Fixed by only advancing on a full, untruncated
   read.

All three fixes are verified (tests pass, CPU and recovery both measurably
improved), but live capture of long multi-frame messages is better, not
solved — resolving one short frame requires waiting for its whole payload
to physically arrive over the mic, leaving little slack before the next
frame's preamble arrives. `--repeat N` (blind whole-message retransmission,
exploiting that the receiver already tolerates a repeated seq for free)
helps but isn't sufficient alone at this loss rate. An open, honest gap.

### Real acoustic hardware (speaker → room air → mic)

`phone`/`fast_air` send/decode, two-way `chat`, and addressing across
separate OS processes all verified working. A duration sweep found
`fast_air`'s 20ms symbols match `phone`'s 40ms in clean conditions (~1–5%
SER either way) but degrade sharply below 15ms (8–15%) and catastrophically
below 10ms (40–90%) — what set `fast_air`'s timing.

### Comparison to ggwave

ggwave has higher raw throughput at its fastest setting, but its default
protocols put 66% of their energy outside the 300–3400Hz telephone band —
it failed to decode in all 6 AMR-NB round-trip trials tested, even at
AMR's best quality. It targets open-air device pairing, a different
problem. `fast_air` + dictionary compression matches or beats ggwave's
fastest mode on longer text (see throughput table).

**Known gaps**: no handshake/calibration wire-protocol (chat.py's
recalibration is an app-level convention, not reserved opcodes), no
selective/ACK-driven retransmission (`--repeat` resends the whole message
blindly), and frame header bytes aren't FEC-protected (a corrupted header
fails that whole frame even if the payload was recoverable).

## Setup

```
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Codec validation needs `ffmpeg-full` (stock `ffmpeg` lacks the AMR encoder):

```
brew install ffmpeg-full
```

Live audio (`send`/`listen`/`chat`) needs:

```
./.venv/bin/pip install -e ".[acoustic-test]"
```

## Usage

Offline, file-based:

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
