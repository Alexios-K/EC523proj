#!/usr/bin/env python3
"""Train the submission model (cli/submission/model.py) on the course split.

Objective: negative SI-SNR on aligned random crops (the DCCRN paper's loss),
optionally plus a power-law compressed spectral loss (--loss sisnr+comp).
After every epoch the whole dev split is enhanced one utterance at a time
through model.enhance (the exact submission path) and scored with the pinned
PESQ-WB / STOI / SI-SDR metrics and the course composite A(D).

Model-selection rule: keep the epoch with the highest dev A(D) on noisy
inputs. The learning rate halves when the run's own objective on dev stops
improving (for --loss sisnr, dev SI-SNR loss);
training stops early when dev A(D) has not improved for --early-stop epochs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from scipy.io import wavfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cli"))
from metrics import quality, si_sdr  # noqa: E402

DATA_ROOT = Path("/projectnb/ec523kb/projects/DeNoise/16khz")
SAMPLE_RATE = 16_000
# lambda for --loss sisnr+comp. At initialisation, on 16 real 4 s training
# crops, the SI-SNR gradient norm is ~535x that of the compressed loss with
# c=0.3 (~100x with c=1), so lambda=500 starts the two terms at about equal
# gradient strength. Use --comp-weight 100 for the c=1 ablation to match.
COMP_WEIGHT = 500.0


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_split(manifest: Path, split: str, workers: int) -> tuple[list[dict], list, list]:
    with manifest.open() as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]

    def read(row):
        _, clean = wavfile.read(DATA_ROOT / "clean_trainset_28spk_wav" / f"{row['id']}.wav")
        _, noisy = wavfile.read(DATA_ROOT / "noisy_trainset_28spk_wav" / f"{row['id']}.wav")
        return clean, noisy  # int16; converted to float per batch

    with ThreadPoolExecutor(workers) as pool:
        pairs = list(pool.map(read, rows))
    return rows, [c for c, _ in pairs], [n for _, n in pairs]


def batches(clean, noisy, batch_size, crop, rng):
    """Yield aligned crops; utterances shorter than the crop are zero-padded."""
    order = rng.permutation(len(clean))
    for start in range(0, len(order) - batch_size + 1, batch_size):
        xb = np.zeros((batch_size, crop), np.float32)
        yb = np.zeros((batch_size, crop), np.float32)
        for k, idx in enumerate(order[start:start + batch_size]):
            n = len(clean[idx])
            off = rng.integers(0, n - crop + 1) if n > crop else 0
            seg = slice(off, off + crop)
            xb[k, : min(n, crop)] = noisy[idx][seg] / 32768.0
            yb[k, : min(n, crop)] = clean[idx][seg] / 32768.0
        yield torch.from_numpy(xb), torch.from_numpy(yb)


def neg_si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate - estimate.mean(-1, keepdim=True)
    target = target - target.mean(-1, keepdim=True)
    scale = (estimate * target).sum(-1, keepdim=True) / (target.pow(2).sum(-1, keepdim=True) + eps)
    projection = scale * target
    noise = estimate - projection
    ratio = projection.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps)
    return -10 * torch.log10(ratio + eps).mean()


def compressed_spectral_loss(estimate, target, stft, power, alpha, eps=1e-8):
    """Power-law compressed magnitude + complex MSE (Braun & Tashev, 2021).

    (1-alpha) * mean| |S^|^c - |S|^c |^2 + alpha * mean| |S^|^c e^{j<S^} - |S|^c e^{j<S} |^2.
    power=1 removes the compression (the mechanism ablation).
    """
    def compress(spec):
        mag = (spec.real ** 2 + spec.imag ** 2 + eps).sqrt()
        scale = mag ** (power - 1)
        return mag ** power, spec.real * scale, spec.imag * scale

    em, er, ei = compress(stft(estimate))
    rm, rr, ri = compress(stft(target))
    magnitude = (em - rm).pow(2).mean()
    complex_ = ((er - rr).pow(2) + (ei - ri).pow(2)).mean()
    return (1 - alpha) * magnitude + alpha * complex_


def make_objective(args, model):
    """Training objective; returns (total, {term: value}) for logging."""
    def stft(x):
        return torch.stft(x, model.n_fft, model.hop, model.win_len, model.window,
                          center=True, return_complex=True)

    def objective(estimate, target):
        sisnr = neg_si_snr(estimate, target)
        if args.loss == "sisnr":
            return sisnr, {"sisnr": sisnr.item()}
        comp = compressed_spectral_loss(estimate, target, stft, args.comp_power, args.comp_alpha)
        return sisnr + args.comp_weight * comp, {"sisnr": sisnr.item(), "comp": comp.item()}

    return objective


def score_one(args):
    """PESQ-WB, STOI, SI-SDR for one utterance (runs in a worker process)."""
    clean, enhanced = args
    from pesq import pesq
    from pystoi import stoi
    try:
        p = float(pesq(SAMPLE_RATE, clean, enhanced, "wb"))
        failed = False
    except Exception:  # an official run would be held; here record the floor and count it
        p, failed = 1.0, True
    s = float(stoi(clean, enhanced, SAMPLE_RATE, extended=False))
    return p, s, si_sdr(clean, enhanced), failed


@torch.no_grad()
def evaluate(model, clean, noisy, device, pool, objective):
    model.eval()
    outputs, losses, terms = [], [], []
    for c, n in zip(clean, noisy):
        x = torch.from_numpy(n / np.float32(32768.0)).view(1, -1).to(device)
        y = torch.from_numpy(c / np.float32(32768.0)).view(1, -1).to(device)
        out = model.enhance(x, SAMPLE_RATE)
        loss, parts = objective(out, y)
        losses.append(loss.item())
        terms.append(parts)
        outputs.append(out.float().cpu().numpy().reshape(-1))
    model.train()
    refs = [c.astype(np.float32) / 32768.0 for c in clean]
    scores = list(pool.map(score_one, zip(refs, outputs), chunksize=8))
    p, s, r, failed = map(list, zip(*scores))
    peaks = np.array([np.abs(o).max() for o in outputs])
    result = quality(p, s, r)
    result.update(
        loss=float(np.mean(losses)),
        **{k: float(np.mean([t[k] for t in terms])) for k in terms[0]},
        pesq_failures=int(sum(failed)),
        max_peak=float(peaks.max()), over_unit_peak=int((peaks > 1.0).sum()),
    )
    return result


def atomic_write(path: Path, write) -> None:
    """write(tmp_path), then rename over path, so a crash (quota, timeout)
    mid-write leaves the previous file intact."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="output directory")
    ap.add_argument("--source", type=Path, default=REPO / "cli/submission/model.py")
    ap.add_argument("--manifest", type=Path, default=REPO / "data/manifest.csv")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--crop-seconds", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-patience", type=int, default=2)
    ap.add_argument("--early-stop", type=int, default=10)
    ap.add_argument("--clip-grad", type=float, default=5.0)
    ap.add_argument("--loss", choices=["sisnr", "sisnr+comp"], default="sisnr")
    ap.add_argument("--comp-power", type=float, default=0.3, help="compression exponent c")
    ap.add_argument("--comp-alpha", type=float, default=0.3, help="weight of the complex term")
    ap.add_argument("--comp-weight", type=float, default=COMP_WEIGHT, help="lambda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NSLOTS", 4)))
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N utterances per split")
    args = ap.parse_args()

    args.run.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    module = load_module(args.source)
    model = module.build_enhancer(SAMPLE_RATE).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    objective = make_objective(args, model)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=args.lr_patience)

    t0 = time.time()
    _, train_clean, train_noisy = load_split(args.manifest, "train", args.workers)
    _, dev_clean, dev_noisy = load_split(args.manifest, "dev", args.workers)
    if args.limit:
        train_clean, train_noisy = train_clean[: args.limit], train_noisy[: args.limit]
        dev_clean, dev_noisy = dev_clean[: args.limit], dev_noisy[: args.limit]
    print(f"loaded {len(train_clean)} train / {len(dev_clean)} dev in {time.time()-t0:.0f}s; "
          f"{params:,} parameters on {device}", flush=True)

    start_epoch, best, stale, history = 1, -1.0, 0, []
    resume = args.run / "last.pt"
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        start_epoch, best, stale, history = (state["epoch"] + 1, state["best"],
                                             state["stale"], state["history"])
        print(f"resumed from epoch {state['epoch']} (best dev A = {best:.4f})", flush=True)
    else:
        (args.run / "config.json").write_text(json.dumps({
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "parameters": params, "git_commit": git_commit(),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__, "train_utterances": len(train_clean),
            "dev_utterances": len(dev_clean),
        }, indent=2) + "\n")

    crop = int(args.crop_seconds * SAMPLE_RATE)
    with ProcessPoolExecutor(args.workers) as pool:
        for epoch in range(start_epoch, args.epochs + 1):
            t_epoch = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            losses, terms = [], []
            for x, y in batches(train_clean, train_noisy, args.batch_size, crop, rng):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                loss, parts = objective(model(x), y)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                if torch.isfinite(grad_norm):
                    optim.step()
                losses.append(loss.item())
                terms.append(parts)
            train_seconds = time.time() - t_epoch
            peak_gib = (torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == "cuda" else 0.0)

            dev = evaluate(model, dev_clean, dev_noisy, device, pool, objective)
            sched.step(dev["loss"])
            improved = dev["quality"] > best
            if improved:
                best, stale = dev["quality"], 0
                weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
                atomic_write(args.run / "model.safetensors", lambda p: save_file(weights, str(p)))
            else:
                stale += 1

            record = {
                "epoch": epoch, "train_loss": float(np.mean(losses)),
                **{f"train_{k}": float(np.mean([t[k] for t in terms])) for k in terms[0]},
                "lr": optim.param_groups[0]["lr"], "train_seconds": train_seconds,
                "epoch_seconds": time.time() - t_epoch, "peak_train_gib": peak_gib,
                **{f"dev_{k}": v for k, v in dev.items()}, "best": improved,
            }
            history.append(record)
            # Rewritten from the checkpointed history, so a resumed epoch is never duplicated.
            atomic_write(args.run / "history.jsonl", lambda p: p.write_text(
                "".join(json.dumps(h) + "\n" for h in history)))
            print(f"epoch {epoch:3d}  train {record['train_loss']:7.3f}  dev {dev['loss']:7.3f}  "
                  f"PESQ {dev['pesq']:.3f}  STOI {dev['stoi']:.4f}  SI-SDR {dev['si_sdr']:6.2f}  "
                  f"A {dev['quality']:.4f}{' *' if improved else '  '}  lr {record['lr']:.1e}  "
                  f"{record['epoch_seconds']:.0f}s  {peak_gib:.1f} GiB", flush=True)

            state = {
                "model": model.state_dict(), "optim": optim.state_dict(),
                "sched": sched.state_dict(), "rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(), "epoch": epoch, "best": best,
                "stale": stale, "history": history,
            }
            atomic_write(resume, lambda p: torch.save(state, p))
            if stale >= args.early_stop:
                print(f"early stop: no dev A improvement for {stale} epochs", flush=True)
                break

    gpu_hours = sum(h["epoch_seconds"] for h in history) / 3600
    print(f"best dev A = {best:.4f}; {gpu_hours:.2f} GPU-hours over {len(history)} epochs")


if __name__ == "__main__":
    main()
