#!/usr/bin/env python3
"""
Real-world usability test suite for the TextOverVoice 2-of-16 MFSK modem
(src/textovervoice/modem.py), run against a real AMR-NB encoder/decoder.

Exercises the actual production modem functions (modulate/demodulate, bit
packing) under conditions closer to a real call:

  1. Isolated symbol accuracy   -- all 64 symbols x 3 AMR bitrates
  2. Full-alphabet sequence     -- all 64 symbols back-to-back in one clip
  3. Symbol duration sensitivity -- how short can a symbol be before AMR
                                    compression starts corrupting it?
  4. Noise robustness           -- AWGN before encoding, 30dB down to 0dB SNR
  5. Tandem transcoding         -- AMR encoded/decoded 2-3x in a row
                                    (simulates a call crossing multiple
                                    non-TrFO network legs)
  6. End-to-end text            -- real ASCII + CJK strings, with and without
                                    Reed-Solomon FEC (reedsolo), byte accuracy

Results are written to realism_results.csv; a human-readable summary prints
at the end.
"""
from __future__ import annotations

import csv
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import reedsolo

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from textovervoice import modem  # noqa: E402
from amr_harness import roundtrip_amr, AMR_NB_BITRATES  # noqa: E402

RNG = random.Random(1234)
REPRESENTATIVE_BITRATES = ["4.75k", "7.40k", "12.2k"]  # worst / mid / best AMR-NB modes

rows: list[dict] = []


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


_np_rng = np.random.default_rng(1234)


def add_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(x ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = _np_rng.normal(0, np.sqrt(noise_power), size=len(x))
    return x + noise


# --- 1. Isolated symbol accuracy --------------------------------------------

def test_isolated_symbols(tmpdir: Path) -> None:
    log("[1/6] Isolated symbol accuracy (64 symbols x 3 bitrates)")
    pad_s = 0.02
    dur = modem.DEFAULT_SYMBOL_DURATION_S
    pad_n = int(pad_s * modem.SR)
    sym_n = int(dur * modem.SR)

    for bitrate in REPRESENTATIVE_BITRATES:
        errors = 0
        low_errors = 0
        high_errors = 0
        for symbol in range(64):
            tone = modem.synth_symbol(symbol, dur)
            padded = np.concatenate([np.zeros(pad_n), tone, np.zeros(pad_n)])
            out = roundtrip_amr(padded, bitrate, tmpdir)
            segment = out[pad_n:pad_n + sym_n] if len(out) >= pad_n + sym_n else out[pad_n:]
            det = modem.detect_symbol(segment)

            exp_low, exp_high = modem.symbol_to_freqs(symbol)
            if det.symbol != symbol:
                errors += 1
            if det.low_freq != exp_low:
                low_errors += 1
            if det.high_freq != exp_high:
                high_errors += 1

        ser = errors / 64
        rows.append({"test": "isolated_symbol", "bitrate": bitrate, "n": 64,
                      "errors": errors, "rate": f"{ser:.4f}",
                      "detail": f"low_group_errors={low_errors} high_group_errors={high_errors}"})
        log(f"  {bitrate}: SER={ser:.1%} ({errors}/64)  low_errs={low_errors} high_errs={high_errors}")


# --- 2. Full-alphabet sequence ----------------------------------------------

def test_full_alphabet_sequence(tmpdir: Path) -> None:
    log("[2/6] Full-alphabet sequence (all 64 symbols, one continuous clip)")
    symbols = list(range(64))
    RNG.shuffle(symbols)

    for bitrate in REPRESENTATIVE_BITRATES:
        audio_in = modem.modulate(symbols)
        audio_out = roundtrip_amr(audio_in, bitrate, tmpdir)
        dets = modem.demodulate(audio_out, len(symbols))
        errors = sum(1 for d, s in zip(dets, symbols) if d.symbol != s)
        ser = errors / len(symbols)
        rows.append({"test": "sequence_full_alphabet", "bitrate": bitrate, "n": len(symbols),
                      "errors": errors, "rate": f"{ser:.4f}", "detail": ""})
        log(f"  {bitrate}: SER={ser:.1%} ({errors}/{len(symbols)}) decoded {len(dets)}/{len(symbols)} symbols")


# --- 3. Symbol duration sensitivity -----------------------------------------

def test_symbol_duration(tmpdir: Path) -> None:
    log("[3/6] Symbol duration sensitivity (20ms-multiple rule)")
    durations = [0.02, 0.03, 0.04, 0.06, 0.08, 0.10]
    symbols = [RNG.randrange(64) for _ in range(32)]

    for bitrate in ["4.75k", "12.2k"]:
        for dur in durations:
            audio_in = modem.modulate(symbols, symbol_duration_s=dur)
            audio_out = roundtrip_amr(audio_in, bitrate, tmpdir)
            dets = modem.demodulate(audio_out, len(symbols), symbol_duration_s=dur)
            errors = sum(1 for d, s in zip(dets, symbols) if d.symbol != s)
            errors += len(symbols) - len(dets)  # truncated clip counts as errors
            ser = errors / len(symbols)
            aligned = "aligned" if abs((dur * 1000) % 20) < 1e-6 else "misaligned"
            rows.append({"test": "symbol_duration", "bitrate": bitrate, "n": len(symbols),
                          "errors": errors, "rate": f"{ser:.4f}",
                          "detail": f"duration_ms={dur*1000:.0f} ({aligned} to 20ms AMR frame)"})
            log(f"  {bitrate} dur={dur*1000:.0f}ms ({aligned}): SER={ser:.1%} ({errors}/{len(symbols)})")


# --- 4. Noise robustness -----------------------------------------------------

def test_noise_robustness(tmpdir: Path) -> None:
    log("[4/6] Noise robustness (AWGN before encoding)")
    snr_levels = [30, 20, 15, 10, 5, 0]
    repeats = 3

    for bitrate in ["4.75k", "12.2k"]:
        for snr_db in snr_levels:
            total_errors = 0
            total_symbols = 0
            for rep in range(repeats):
                symbols = [RNG.randrange(64) for _ in range(32)]
                audio_in = modem.modulate(symbols)
                noisy = add_noise(audio_in, snr_db)
                audio_out = roundtrip_amr(noisy, bitrate, tmpdir)
                dets = modem.demodulate(audio_out, len(symbols))
                errors = sum(1 for d, s in zip(dets, symbols) if d.symbol != s)
                errors += len(symbols) - len(dets)
                total_errors += errors
                total_symbols += len(symbols)
            ser = total_errors / total_symbols
            rows.append({"test": "noise_robustness", "bitrate": bitrate, "n": total_symbols,
                          "errors": total_errors, "rate": f"{ser:.4f}",
                          "detail": f"snr_db={snr_db} repeats={repeats}"})
            log(f"  {bitrate} SNR={snr_db:3d}dB: SER={ser:.1%} ({total_errors}/{total_symbols})")


# --- 5. Tandem transcoding ---------------------------------------------------

def test_tandem_codec(tmpdir: Path) -> None:
    log("[5/6] Tandem transcoding (multiple AMR encode/decode passes)")
    symbols = [RNG.randrange(64) for _ in range(32)]

    for bitrate in ["4.75k", "12.2k"]:
        for passes in [1, 2, 3]:
            audio_in = modem.modulate(symbols)
            audio_out = roundtrip_amr(audio_in, bitrate, tmpdir, passes=passes)
            dets = modem.demodulate(audio_out, len(symbols))
            errors = sum(1 for d, s in zip(dets, symbols) if d.symbol != s)
            errors += len(symbols) - len(dets)
            ser = errors / len(symbols)
            rows.append({"test": "tandem_codec", "bitrate": bitrate, "n": len(symbols),
                          "errors": errors, "rate": f"{ser:.4f}", "detail": f"passes={passes}"})
            log(f"  {bitrate} x{passes} pass(es): SER={ser:.1%} ({errors}/{len(symbols)})")


# --- 6. End-to-end text, with/without Reed-Solomon FEC ----------------------

def rs_protect(data: bytes, parity_bytes: int = 10) -> bytes:
    rsc = reedsolo.RSCodec(parity_bytes)
    return bytes(rsc.encode(data))


def rs_recover(data: bytes, parity_bytes: int = 10) -> bytes | None:
    rsc = reedsolo.RSCodec(parity_bytes)
    try:
        decoded, _, _ = rsc.decode(data)
        return bytes(decoded)
    except reedsolo.ReedSolomonError:
        return None


def test_end_to_end_text(tmpdir: Path) -> None:
    log("[6/6] End-to-end text (ASCII + CJK, with/without RS FEC)")
    samples = {
        "ascii": "Hello TextOverVoice, testing 1234567890!",
        "cjk": "文字轉聲音穿越電話通道測試，準確率至關重要。",
    }

    for label, text in samples.items():
        payload = text.encode("utf-8")
        for bitrate in ["4.75k", "12.2k"]:
            for use_fec in [False, True]:
                data = rs_protect(payload) if use_fec else payload
                symbols = modem.bytes_to_symbols(data)
                audio_in = modem.modulate(symbols)
                audio_out = roundtrip_amr(audio_in, bitrate, tmpdir)
                dets = modem.demodulate(audio_out, len(symbols))
                decoded_symbols = [d.symbol for d in dets] + [0] * (len(symbols) - len(dets))
                decoded_data = modem.symbols_to_bytes(decoded_symbols, len(data))

                if use_fec:
                    recovered = rs_recover(decoded_data)
                    ok = recovered == payload
                    text_out = None
                    if recovered is not None:
                        try:
                            text_out = recovered.decode("utf-8")
                        except UnicodeDecodeError:
                            text_out = None
                else:
                    ok = decoded_data == payload
                    try:
                        text_out = decoded_data.decode("utf-8")
                    except UnicodeDecodeError:
                        text_out = None

                byte_errors = sum(a != b for a, b in zip(decoded_data, data))
                rows.append({
                    "test": "end_to_end_text", "bitrate": bitrate, "n": len(data),
                    "errors": byte_errors, "rate": f"{byte_errors/max(len(data),1):.4f}",
                    "detail": f"sample={label} fec={'RS10' if use_fec else 'none'} "
                              f"exact_match={ok} text_matches={text_out == text}",
                })
                status = "OK " if ok else "FAIL"
                log(f"  [{status}] {label:6s} {bitrate:6s} fec={'RS10' if use_fec else 'none':5s} "
                    f"byte_errors={byte_errors}/{len(data)} exact_recovery={ok}")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        test_isolated_symbols(tmpdir)
        test_full_alphabet_sequence(tmpdir)
        test_symbol_duration(tmpdir)
        test_noise_robustness(tmpdir)
        test_tandem_codec(tmpdir)
        test_end_to_end_text(tmpdir)

    out_csv = Path(__file__).parent / "realism_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "bitrate", "n", "errors", "rate", "detail"])
        w.writeheader()
        w.writerows(rows)
    log(f"\nWrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
