#!/usr/bin/env python3
"""Figures for the midway report, drawn from the dev comparison Markdown.

Reads the tables that train/compare.py writes (runs/comparison.md, also
pasted into handout/midway_report.md) and writes three figures:

  fig1_method.png  DCCRN pipeline with the modified training loss marked;
  fig2_snr.png     dev A(D) on noisy input by SNR for every system;
  fig3_noise.png   paired change in A(D) against DCCRN by noise type,
                   with the 95% bootstrap intervals from the comparison.

  python report/make_figures.py                      # reads handout/midway_report.md
  python report/make_figures.py --comparison runs/comparison.md --output report/figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LABELS = {"noisy input": "Noisy input", "dccrn": "DCCRN (SI-SNR)",
          "comp_c03": "+ comp. loss, c = 0.3", "comp_c1": "+ uncomp. loss, c = 1"}
STYLE = {"noisy input": ("0.6", ":", "s"), "dccrn": ("0.2", "-", "o"),
         "comp_c03": ("tab:blue", "-", "^"), "comp_c1": ("tab:orange", "--", "v")}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def table_after(text: str, heading: str) -> list[list[str]]:
    """Rows (header first, separator dropped) of the first Markdown table under a heading."""
    start = text.index(heading)
    rows = []
    for line in text[start:].splitlines()[1:]:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
        elif rows:
            break
    if not rows:
        raise SystemExit(f"no table under {heading!r}")
    return rows


def interval(cell: str) -> tuple[float, float, float]:
    """'+0.0243 [+0.0209, +0.0279]' -> (mean, low, high)."""
    mean, low, high = (float(x) for x in re.findall(r"[-+]?\d*\.\d+", cell))
    return mean, low, high


def method_figure(path: Path) -> None:
    """Column-width diagram: the network in two rows, the training loss below."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    def box(x, y, w, h, text, face="white", edge="black", style="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=face, ec=edge,
                                    lw=0.8, ls=style))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6)

    def arrow(x0, y0, x1, y1, style="-", color="black"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.8, ls=style, color=color))

    w, h = 2.1, 1.2
    xs = (0.2, 2.75, 5.3, 7.85)
    top, mid = 5.9, 3.7
    for x, text in zip(xs, ("Noisy\nwaveform", "STFT\n512 pt", "Complex\nencoder ×6",
                            "Complex\nLSTM ×2")):
        box(x, top, w, h, text)
    for x, text in zip(xs[1:], ("iSTFT\nenhanced", "E mask\ntanh, phase",
                                "Complex\ndecoder ×6")):
        box(x, mid, w, h, text)
    box(xs[0], mid, w, h, "Clean target\n(training)", style="--")
    for x0, x1 in zip(xs, xs[1:]):
        arrow(x0 + w + 0.08, top + h / 2, x1 - 0.08, top + h / 2)
    arrow(xs[3] + w / 2, top - 0.08, xs[3] + w / 2, mid + h + 0.08)
    for x0, x1 in ((xs[3], xs[2]), (xs[2], xs[1])):
        arrow(x0 - 0.08, mid + h / 2, x1 + w + 0.08, mid + h / 2)
    arrow(xs[2] + w - 0.2, top - 0.08, xs[3] + 0.2, mid + h + 0.08, style=":", color="0.4")
    ax.text(7.25, 5.15, "skip", fontsize=5.5, color="0.4", ha="right")

    box(0.2, 0.3, 9.75, 2.0, "Training loss (our change)\n"
        r"$\mathcal{L} = \mathcal{L}_{\mathrm{SI\!-\!SNR}} + \lambda\,\mathcal{L}_{\mathrm{comp}}(c)$"
        "\n" r"$c = 0.3$, $\lambda = 500$   (ablation: $c = 1$, $\lambda = 100$)",
        face="#e8f0fb", edge="tab:blue", style="--")
    for x in xs[:2]:
        arrow(x + w / 2, mid - 0.08, x + w / 2, 2.38, style="--")
    fig.savefig(path)
    plt.close(fig)


def snr_figure(text: str, path: Path) -> None:
    rows = table_after(text, "## 3a.")
    header, body = rows[0], rows[1:]
    systems = [c for c in header[2:] if not c.startswith("Δ")]
    snrs = [float(r[0]) for r in body]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for s in systems:
        col = header.index(s)
        a = [float(r[col].split("/")[1]) for r in body]
        color, ls, marker = STYLE.get(s, ("k", "-", "o"))
        ax.plot(snrs, a, color=color, ls=ls, marker=marker, ms=4, lw=1.2, label=LABELS.get(s, s))
    ax.set_xticks(snrs)
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("Dev A(D), noisy input")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", lw=0.3, alpha=0.6)
    fig.savefig(path)
    plt.close(fig)


def noise_figure(text: str, path: Path) -> None:
    rows = table_after(text, "## 3b.")
    header, body = rows[0], rows[1:]
    deltas = [c for c in header if c.startswith("Δ")]
    noises = [r[0] for r in body]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    step = 0.36
    for k, col_name in enumerate(deltas):
        system = col_name.split()[0][1:]
        col = header.index(col_name)
        stats = [interval(r[col]) for r in body]
        ys = [i + (k - (len(deltas) - 1) / 2) * step for i in range(len(noises))]
        color, _, marker = STYLE.get(system, ("k", "-", "o"))
        ax.errorbar([m for m, _, _ in stats], ys,
                    xerr=[[m - lo for m, lo, _ in stats], [hi - m for m, _, hi in stats]],
                    fmt=marker, color=color, ms=3.5, lw=1.0, capsize=1.5,
                    label=LABELS.get(system, system))
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(range(len(noises)))
    ax.set_yticklabels(noises)
    ax.invert_yaxis()
    ax.set_xlabel("Change in A(D) vs. DCCRN (95% CI)")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(axis="x", lw=0.3, alpha=0.6)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", type=Path, default=REPO / "handout/midway_report.md",
                    help="Markdown written by train/compare.py")
    ap.add_argument("--output", type=Path, default=REPO / "report/figures")
    args = ap.parse_args()
    text = args.comparison.read_text()
    args.output.mkdir(parents=True, exist_ok=True)
    method_figure(args.output / "fig1_method.png")
    snr_figure(text, args.output / "fig2_snr.png")
    noise_figure(text, args.output / "fig3_noise.png")
    print(f"wrote 3 figures to {args.output}")


if __name__ == "__main__":
    main()
