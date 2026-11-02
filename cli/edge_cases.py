#!/usr/bin/env python3
"""Stress the submitted enhancer on the input types the evaluator checks.

The handout says the offline evaluation reruns the exact command on clean,
quiet, long, short, and amplitude-edge inputs and compares output hashes
across repeated runs. This script builds those inputs synthetically (no
corpus audio needed), runs each one through the submission twice exactly as
enhance.py does, and checks what the runner would reject:

  same 1 x T shape, all samples finite, peak <= 1.0 (the runner neither
  clips nor renormalises), and bitwise identical output on the second run.

  python edge_cases.py                       # submitted checkpoint, CPU
  python edge_cases.py --device cuda --report edge_cases.json

Exit status is 1 if any case fails, so it can gate a submission.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from common import SAMPLE_RATE
from enhance import load_submission

RNG_SEED = 523


def speechlike(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Voiced harmonics with a gliding pitch and syllable-rate envelope, peak 1."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    x = sum(np.sin(k * phase) / k for k in range(1, 20))
    x *= 0.5 * (1 + np.sin(2 * np.pi * 4 * t)) ** 2
    x += 0.02 * rng.standard_normal(t.size)
    return x / np.max(np.abs(x))


def cases(rng: np.random.Generator) -> dict[str, np.ndarray]:
    speech = speechlike(3.0, rng)
    noise = rng.standard_normal(speech.size)
    noisy = speech + 0.3 * noise / np.max(np.abs(noise))
    noisy /= np.max(np.abs(noisy))
    t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    out = {
        "clean speech-like, peak 0.5": 0.5 * speech,
        "noisy speech-like, peak 0.5": 0.5 * noisy,
        "silence, 1 s": np.zeros(SAMPLE_RATE),
        "quiet, peak 1e-3": 1e-3 * noisy,
        "very quiet, peak 1e-5": 1e-5 * noisy,
        "loud speech-like, peak 0.99": 0.99 * noisy,
        "hard clipped speech": np.clip(3.0 * noisy, -0.99, 0.99),
        "white noise, peak 0.99": 0.99 * np.clip(noise / 4, -1, 1),
        "sine 1 kHz, peak 0.99": 0.99 * np.sin(2 * np.pi * 1000 * t),
        "square 200 Hz, peak 0.99": 0.99 * np.sign(np.sin(2 * np.pi * 200 * t) + 1e-9),
        "DC offset 0.5 + speech": np.clip(0.5 + 0.3 * noisy, -1, 1),
        "long, 120 s": 0.5 * np.tile(noisy, 40),
    }
    for n in (1, 2, 50, 99, 100, 101, 399, 400, 401, 1000):
        out[f"short, {n} samples"] = 0.5 * noisy[:n]
    return {k: v.astype(np.float32) for k, v in out.items()}


def run(model, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    tensor = torch.from_numpy(x).view(1, -1).to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        y = model.enhance(tensor, SAMPLE_RATE)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    if not isinstance(y, torch.Tensor) or tuple(y.shape) != tuple(tensor.shape):
        raise ValueError(f"shape {tuple(getattr(y, 'shape', ()))} != {tuple(tensor.shape)}")
    return y.detach().to("cpu", torch.float32).numpy().reshape(-1), elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("submission/model.safetensors"))
    ap.add_argument("--source", type=Path, default=Path("submission/model.py"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--report", type=Path, help="optional JSON with every case")
    args = ap.parse_args()

    torch.manual_seed(RNG_SEED)
    module = load_submission(args.source)
    model = module.build_enhancer(SAMPLE_RATE)
    model.load_state_dict(load_file(str(args.checkpoint), device="cpu"), strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    rows = []
    for name, x in cases(np.random.default_rng(RNG_SEED)).items():
        row = {"case": name, "samples": int(x.size), "input_peak": float(np.max(np.abs(x)))}
        try:
            y1, elapsed = run(model, x, device)
            y2, _ = run(model, x, device)
            finite = bool(np.isfinite(y1).all())
            row.update(output_peak=float(np.max(np.abs(y1))) if finite else float("nan"),
                       finite=finite, deterministic=bool(np.array_equal(y1, y2)),
                       seconds=elapsed, error=None)
            row["pass"] = finite and row["output_peak"] <= 1.0 and row["deterministic"]
        except Exception as e:  # report every failure instead of stopping at the first
            row.update(output_peak=None, finite=None, deterministic=None, seconds=None,
                       error=f"{type(e).__name__}: {e}", **{"pass": False})
        rows.append(row)

    width = max(len(r["case"]) for r in rows)
    print(f"{'case':<{width}}  {'samples':>8}  {'in peak':>8}  {'out peak':>8}  det  result")
    for r in rows:
        out = "error" if r["output_peak"] is None else f"{r['output_peak']:.4f}"
        det = "-" if r["deterministic"] is None else ("yes" if r["deterministic"] else "NO")
        result = "pass" if r["pass"] else f"FAIL {r['error'] or ''}".strip()
        print(f"{r['case']:<{width}}  {r['samples']:>8}  {r['input_peak']:>8.4f}  {out:>8}  "
              f"{det:>3}  {result}")
    failed = [r["case"] for r in rows if not r["pass"]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} cases pass on {device}")

    if args.report:
        args.report.write_text(json.dumps({
            "checkpoint": str(args.checkpoint), "device": str(device),
            "torch": torch.__version__, "cases": rows}, indent=2))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
