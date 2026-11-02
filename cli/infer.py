#!/usr/bin/env python3
"""Enhance one WAV file or a whole folder of WAV files with the submitted model.

    # one file -> one file
    python infer.py --input noisy.wav --output enhanced.wav

    # a folder (searched recursively) -> a mirrored folder plus run.json
    python infer.py --input evaluation/noisy --output predictions

Defaults: --checkpoint submission/model.safetensors, --source
submission/model.py, and --device cuda when available, else cpu. Run from
this cli/ directory.

Inputs must be mono 16 kHz WAV (int16 or float32). Outputs are float32 mono
16 kHz WAV with exactly the input's sample count. The model sees only the
waveform: no filename, reference, SNR, or other metadata.

Folder mode delegates to the unmodified course runner enhance.py, so it writes
the same files and run.json (hashes, peaks, RTF) as the official command:

    python enhance.py --checkpoint submission/model.safetensors \\
        --input evaluation/noisy --output predictions --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

import enhance
from common import SAMPLE_RATE, read_wav, write_wav

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", type=Path, required=True, help="a .wav file or a folder")
    ap.add_argument("--output", type=Path, required=True,
                    help="output .wav path (file mode) or folder (folder mode)")
    ap.add_argument("--checkpoint", type=Path, default=HERE / "submission/model.safetensors")
    ap.add_argument("--source", type=Path, default=HERE / "submission/model.py")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.input.is_dir():
        sys.argv = [
            "enhance.py", "--checkpoint", str(args.checkpoint), "--source", str(args.source),
            "--input", str(args.input), "--output", str(args.output), "--device", args.device,
        ]
        enhance.main()
        return
    if not args.input.is_file():
        raise SystemExit(f"{args.input}: no such file or folder")
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("output must differ from input")

    # Same loading and inference path as enhance.py, for a single file.
    model = enhance.load_submission(args.source).build_enhancer(SAMPLE_RATE)
    model.load_state_dict(load_file(str(args.checkpoint), device="cpu"), strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    rate, x = read_wav(args.input)
    if rate != SAMPLE_RATE:
        raise SystemExit(f"{args.input}: expected {SAMPLE_RATE} Hz, got {rate}")
    t0 = time.perf_counter()
    with torch.inference_mode():
        y = model.enhance(torch.from_numpy(x).view(1, -1).to(device), SAMPLE_RATE)
    elapsed = time.perf_counter() - t0
    out = y.detach().to("cpu", torch.float32).numpy().reshape(-1)
    if out.shape != x.shape:
        raise SystemExit(f"model returned {out.shape[0]} samples for a {x.shape[0]}-sample input")
    write_wav(args.output, SAMPLE_RATE, out)  # refuses non-finite output or peak > 1.0
    print(f"{args.input} -> {args.output}  ({len(x) / SAMPLE_RATE:.2f} s audio, "
          f"{elapsed:.3f} s, peak {abs(out).max():.3f})")


if __name__ == "__main__":
    main()
