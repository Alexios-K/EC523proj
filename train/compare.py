#!/usr/bin/env python3
"""Compare dev-split evaluations of several systems against a reference system.

Each system is NAME=DIR, where DIR holds the noisy_report.json and
clean_report.json written by jobs/eval_dev.sh. The first system is the
reference (the seed-paper baseline). Writes a Markdown report with:

  1. main table: noisy input and every system, PESQ / STOI / SI-SDR / A(D),
     clean-input A(D), and the leaderboard formula on dev;
  2. paired bootstrap 95% intervals over utterance IDs for each system minus
     the reference, plus the share of utterances each system improves;
  3. breakdowns by SNR and by noise type;
  4. metric-disagreement utterances (PESQ up but SI-SDR down, and the
     reverse) and the largest wins/losses, as a shortlist for listening.

  python train/compare.py dccrn=runs/dccrn_seed0/dev_eval \\
      comp_c03=runs/dccrn_comp_c03_seed0/dev_eval comp_c1=runs/dccrn_comp_c1_seed0/dev_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cli"))
from metrics import leaderboard, normalize_pesq, normalize_si_sdr, normalize_stoi  # noqa: E402

METRICS = ("pesq", "stoi", "si_sdr", "a")


def per_utterance(report: Path) -> dict[str, dict[str, float]]:
    """id -> raw metrics plus the per-utterance composite a_i (A(D) is their mean)."""
    rows = {}
    for f in json.loads(report.read_text())["files"]:
        a = (0.40 * normalize_pesq(f["pesq"]) + 0.30 * normalize_stoi(f["stoi"])
             + 0.30 * normalize_si_sdr(f["si_sdr"]))
        rows[f["id"][:-4] if f["id"].endswith(".wav") else f["id"]] = {"pesq": f["pesq"], "stoi": f["stoi"],
                                              "si_sdr": f["si_sdr"], "a": a}
    return rows


def matrix(rows: dict, ids: list[str]) -> dict[str, np.ndarray]:
    return {m: np.array([rows[i][m] for i in ids]) for m in METRICS}


def bootstrap(diff: np.ndarray, draws: np.ndarray) -> tuple[float, float, float]:
    means = diff[draws].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", help="NAME=DIR; the first is the reference")
    ap.add_argument("--noisy-input", type=Path, default=REPO / "runs/noisy_input_dev_report.json")
    ap.add_argument("--manifest", type=Path, default=REPO / "data/manifest.csv")
    ap.add_argument("--out", type=Path, default=REPO / "runs/comparison.md")
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--listen", type=int, default=5, help="utterances per listening list")
    args = ap.parse_args()

    systems = []
    for spec in args.systems:
        name, _, path = spec.partition("=")
        systems.append((name, Path(path)))
    with args.manifest.open() as f:
        meta = {r["id"]: r for r in csv.DictReader(f) if r["split"] == "dev"}
    ids = sorted(meta)

    noisy_in = per_utterance(args.noisy_input)
    noisy, clean = {}, {}
    for name, path in systems:
        noisy[name] = per_utterance(path / "noisy_report.json")
        clean[name] = per_utterance(path / "clean_report.json")
        for label, rows in (("noisy", noisy[name]), ("clean", clean[name])):
            if set(rows) != set(ids):
                raise SystemExit(f"{name} {label} report covers {len(rows)} utterances, "
                                 f"not the {len(ids)} dev IDs in the manifest; different split?")
    if set(noisy_in) != set(ids):
        raise SystemExit("noisy-input report does not match the dev manifest")

    N = {name: matrix(rows, ids) for name, rows in noisy.items()}
    C = {name: matrix(rows, ids) for name, rows in clean.items()}
    X = matrix(noisy_in, ids)
    ref = systems[0][0]
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(ids), size=(args.resamples, len(ids)))
    out = []

    out.append(f"# Dev comparison ({len(ids)} utterances, speakers "
               f"{', '.join(sorted({meta[i]['speaker'] for i in ids}))})\n")
    out.append("## 1. Main table\n")
    out.append("Noisy-in columns score enhanced noisy speech; clean-in A(D) scores the "
               "model run on clean speech (clean preservation). Score = 100(0.85 A_noisy + "
               "0.15 A_clean), the leaderboard formula applied to dev.\n")
    out.append("| System | PESQ | STOI | SI-SDR (dB) | A(D) noisy-in | A(D) clean-in | Score |")
    out.append("|---|---|---|---|---|---|---|")
    out.append(f"| noisy input | {X['pesq'].mean():.3f} | {X['stoi'].mean():.4f} | "
               f"{X['si_sdr'].mean():.2f} | {X['a'].mean():.4f} | (identity) | |")
    for name, _ in systems:
        n, c = N[name], C[name]
        score = leaderboard(n["a"].mean(), c["a"].mean())
        out.append(f"| {name} | {n['pesq'].mean():.3f} | {n['stoi'].mean():.4f} | "
                   f"{n['si_sdr'].mean():.2f} | {n['a'].mean():.4f} | {c['a'].mean():.4f} | "
                   f"{score:.2f} |")

    out.append(f"\n## 2. Paired differences vs `{ref}` (95% bootstrap CI over utterance IDs, "
               f"{args.resamples} resamples)\n")
    out.append("An interval that excludes 0 means the difference is unlikely to be "
               "utterance-sampling noise. It says nothing about seed-to-seed variation.\n")
    out.append("| System | ΔPESQ | ΔSTOI | ΔSI-SDR (dB) | ΔA noisy-in | ΔA clean-in | ΔScore | "
               "better A on |")
    out.append("|---|---|---|---|---|---|---|---|")
    for name, _ in systems[1:]:
        cells = []
        for m in METRICS:
            mean, lo, hi = bootstrap(N[name][m] - N[ref][m], draws)
            fmt = ".4f" if m in ("stoi", "a") else (".2f" if m == "si_sdr" else ".3f")
            cells.append(f"{mean:+{fmt}} [{lo:+{fmt}}, {hi:+{fmt}}]")
        mean, lo, hi = bootstrap(C[name]["a"] - C[ref]["a"], draws)
        cells.append(f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        # Score is linear in per-utterance composites over the same IDs.
        score_diff = 100 * (0.85 * (N[name]["a"] - N[ref]["a"]) + 0.15 * (C[name]["a"] - C[ref]["a"]))
        mean, lo, hi = bootstrap(score_diff, draws)
        cells.append(f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        cells.append(f"{(N[name]['a'] > N[ref]['a']).mean():.0%}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")

    for part, key, title in (("3a", "snr_db", "SNR (dB, as logged)"), ("3b", "noise", "noise type")):
        groups = sorted({meta[i][key] for i in ids}, key=lambda g: (float(g) if key == "snr_db" else g))
        out.append(f"\n## {part}. By {title}: mean PESQ / A(D) noisy-in\n")
        out.append("| " + key + " | n | noisy input | " + " | ".join(n for n, _ in systems)
                   + " | " + " | ".join(f"Δ{n} A" for n, _ in systems[1:]) + " |")
        out.append("|" + "---|" * (3 + 2 * len(systems) - 1))
        for g in groups:
            sel = np.array([meta[i][key] == g for i in ids])
            cells = [f"{X['pesq'][sel].mean():.2f} / {X['a'][sel].mean():.3f}"]
            cells += [f"{N[n]['pesq'][sel].mean():.2f} / {N[n]['a'][sel].mean():.3f}" for n, _ in systems]
            for n, _ in systems[1:]:
                d = N[n]["a"][sel] - N[ref]["a"][sel]
                means = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
                cells.append(f"{d.mean():+.4f} [{np.percentile(means, 2.5):+.4f}, "
                             f"{np.percentile(means, 97.5):+.4f}]")
            out.append(f"| {g} | {int(sel.sum())} | " + " | ".join(cells) + " |")

    out.append(f"\n## 4. Listening shortlist vs `{ref}`\n")
    out.append("Listen to these in data/dev/{clean,noisy} and each system's "
               "dev_eval/enhanced_noisy, and look at spectrograms. Disagreements show where "
               "the metrics trade off; extremes show the clearest wins and failures.\n")
    for name, _ in systems[1:]:
        dp = N[name]["pesq"] - N[ref]["pesq"]
        dr = N[name]["si_sdr"] - N[ref]["si_sdr"]
        da = N[name]["a"] - N[ref]["a"]

        def fmt(k):
            i = ids[k]
            return (f"{i} ({meta[i]['noise']}, {meta[i]['snr_db']} dB): ΔPESQ {dp[k]:+.2f}, "
                    f"ΔSI-SDR {dr[k]:+.2f}, ΔA {da[k]:+.3f}")

        lists = {
            "PESQ up, SI-SDR down": [k for k in np.argsort(-(dp - dr / 10)) if dp[k] > 0 and dr[k] < 0],
            "SI-SDR up, PESQ down": [k for k in np.argsort(-(dr / 10 - dp)) if dr[k] > 0 and dp[k] < 0],
            "largest A gains": list(np.argsort(-da)),
            "largest A losses": list(np.argsort(da)),
        }
        out.append(f"### {name}\n")
        for title, ks in lists.items():
            count = len(ks) if "up" in title else None
            out.append(f"**{title}**" + (f" ({count} utterances)" if count is not None else "") + "\n")
            out.extend(f"- {fmt(k)}" for k in ks[: args.listen])
            out.append("")

    args.out.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
