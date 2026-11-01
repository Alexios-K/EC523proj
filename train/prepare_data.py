#!/usr/bin/env python3
"""Verify the 16 kHz VoiceBank-DEMAND pairs and write a speaker-grouped split.

Checks every pair for: 16 kHz int16 mono, equal length, finite, nonsilent
clean audio, zero lag at the cross-correlation peak, and a measured SNR
consistent with the corpus log. Writes data/manifest.csv and mirrors the dev
split as symlink trees (data/dev/{clean,noisy}) for the official
enhance.py -> evaluate.py chain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path("/projectnb/ec523kb/projects/DeNoise")
# One male and one female held-out speaker, mirroring the two unseen speakers
# (one of each) in the standard VoiceBank-DEMAND test set.
DEV_SPEAKERS = ("p226", "p230")
MAX_LAG = 160  # 10 ms


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pair(uid: str, clean_dir: Path, noisy_dir: Path) -> dict:
    row = {"id": uid}
    cr, clean = wavfile.read(clean_dir / f"{uid}.wav")
    nr, noisy = wavfile.read(noisy_dir / f"{uid}.wav")
    problems = []
    if cr != 16_000 or nr != 16_000:
        problems.append(f"rate {cr}/{nr}")
    if clean.dtype != np.int16 or noisy.dtype != np.int16:
        problems.append(f"dtype {clean.dtype}/{noisy.dtype}")
    if clean.ndim != 1 or noisy.ndim != 1:
        problems.append("not mono")
    if clean.shape != noisy.shape:
        problems.append(f"length {clean.shape}/{noisy.shape}")
    if problems:
        row["problems"] = "; ".join(problems)
        return row
    y = clean.astype(np.float64) / 32768.0
    x = noisy.astype(np.float64) / 32768.0
    n = x - y
    clean_energy = float(np.dot(y, y))
    if clean_energy < 1e-6:
        problems.append("silent clean")
    # Lag of the clean/noisy cross-correlation peak within +-10 ms.
    size = 1 << int(np.ceil(np.log2(2 * len(y))))
    xc = np.fft.irfft(np.fft.rfft(x, size) * np.conj(np.fft.rfft(y, size)), size)
    lags = np.concatenate([xc[: MAX_LAG + 1], xc[-MAX_LAG:]])
    lag = int(np.argmax(lags))
    lag = lag if lag <= MAX_LAG else lag - (2 * MAX_LAG + 1)
    row.update(
        samples=len(y),
        measured_snr_db=round(10 * np.log10(clean_energy / max(float(np.dot(n, n)), 1e-12)), 3),
        lag=lag,
        clean_peak=round(float(np.abs(y).max()), 5),
        noisy_peak=round(float(np.abs(x).max()), 5),
        clean_sha256=sha256(clean_dir / f"{uid}.wav"),
        noisy_sha256=sha256(noisy_dir / f"{uid}.wav"),
        problems="; ".join(problems),
    )
    return row


def link_tree(rows: list[dict], src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for row in rows:
        link = dst / f"{row['id']}.wav"
        if not link.exists():
            link.symlink_to(src / f"{row['id']}.wav")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    clean_dir = args.root / "16khz/clean_trainset_28spk_wav"
    noisy_dir = args.root / "16khz/noisy_trainset_28spk_wav"
    log = {}
    for line in (args.root / "log_trainset_28spk.txt").read_text().split("\n"):
        if line.strip():
            uid, noise, snr = line.split()
            log[uid] = (noise, float(snr))

    clean_ids = {p.stem for p in clean_dir.glob("*.wav")}
    noisy_ids = {p.stem for p in noisy_dir.glob("*.wav")}
    if clean_ids != noisy_ids:
        raise SystemExit(f"unpaired files: {sorted(clean_ids ^ noisy_ids)[:10]}")
    missing_log = clean_ids - log.keys()
    if missing_log:
        raise SystemExit(f"files absent from noise log: {sorted(missing_log)[:10]}")

    ids = sorted(clean_ids)
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda u: check_pair(u, clean_dir, noisy_dir), ids))

    for row in rows:
        speaker = row["id"].split("_")[0]
        row["speaker"] = speaker
        row["split"] = "dev" if speaker in DEV_SPEAKERS else "train"
        row["noise"], row["snr_db"] = log[row["id"]]
        if not row.get("problems") and row["lag"] != 0:
            row["problems"] = f"lag {row['lag']}"

    bad = [r for r in rows if r.get("problems")]
    good = [r for r in rows if not r.get("problems")]
    fields = ["id", "speaker", "split", "noise", "snr_db", "samples", "measured_snr_db",
              "lag", "clean_peak", "noisy_peak", "clean_sha256", "noisy_sha256"]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(good)
    if bad:
        with (args.out / "rejected.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, ["id", "problems"], extrasaction="ignore")
            w.writeheader()
            w.writerows(bad)

    dev = [r for r in good if r["split"] == "dev"]
    link_tree(dev, clean_dir, args.out / "dev/clean")
    link_tree(dev, noisy_dir, args.out / "dev/noisy")

    logged = np.array([r["snr_db"] for r in good])
    measured = np.array([r["measured_snr_db"] for r in good])
    hours = sum(r["samples"] for r in good) / 16_000 / 3600
    print(f"pairs: {len(rows)}  accepted: {len(good)}  rejected: {len(bad)}  ({hours:.2f} h)")
    for split in ("train", "dev"):
        sel = [r for r in good if r["split"] == split]
        print(f"{split}: {len(sel)} utterances, {len({r['speaker'] for r in sel})} speakers")
    print("lag != 0:", sum(r.get("lag", 0) != 0 for r in rows))
    print(f"corr(logged SNR, measured SNR) = {np.corrcoef(logged, measured)[0, 1]:.3f}")
    for snr in sorted(set(logged)):
        m = measured[logged == snr]
        print(f"  logged {snr:4.0f} dB: measured mean {m.mean():6.2f}, min {m.min():6.2f}, max {m.max():6.2f}")


if __name__ == "__main__":
    main()
