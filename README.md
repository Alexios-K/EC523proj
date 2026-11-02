# EC523 Project B: DCCRN with a compressed spectral loss

Single-channel speech enhancement on VoiceBank-DEMAND at 16 kHz.

- **Seed paper (baseline):** DCCRN-E (Hu et al., Interspeech 2020,
  [arXiv:2008.00264](https://arxiv.org/abs/2008.00264)), trained with its own
  SI-SNR loss.
- **Modification:** the same network and data, trained with SI-SNR plus a
  power-law compressed spectral loss (Braun & Tashev, 2021) with exponent
  c = 0.3. An ablation with c = 1 (no compression) isolates the compression.
- **Submitted checkpoint:** `cli/submission/model.safetensors`, the c = 0.3
  model (3,671,053 parameters).

## Quick start: inference

Run from `cli/`. Inputs are mono 16 kHz WAV files (int16 or float32). Each
output is a float32 mono 16 kHz WAV with exactly the input's sample count.

```sh
cd cli

# a single file
python infer.py --input path/to/noisy.wav --output path/to/enhanced.wav

# a whole folder (searched recursively; output mirrors the relative paths
# and adds run.json with hashes, peaks, and real-time factor)
python infer.py --input path/to/noisy_dir --output path/to/enhanced_dir

# the official course command (folders only) works unchanged
python enhance.py --checkpoint submission/model.safetensors \
    --input evaluation/noisy --output predictions --device cuda
```

`infer.py` defaults to `--checkpoint submission/model.safetensors`,
`--source submission/model.py`, and `--device cuda` when a GPU is available,
else `cpu`. Folder mode calls the unmodified course runner `enhance.py`, so
both commands write identical audio.

To score enhanced files against clean references with the course metrics:

```sh
python evaluate.py --clean path/to/clean_dir --enhanced path/to/enhanced_dir \
    --report report.json
```

## Environment

**Inference** needs only the course default environment,
`cli/requirements.txt` (torch 2.6.0, numpy, scipy, safetensors). We verified
it there: the full dev set runs under torch 2.6.0, and its outputs match our
training environment's to within 5e-4 per sample (CPU vs. GPU float
differences).

```sh
pip install -r cli/requirements.txt
```

**Training** used `requirements-train.txt` (Python 3.12, torch 2.9.1). On BU
SCC that is the course conda environment plus a small venv for the two metric
packages:

```sh
module load miniconda academic-ml/spring-2026
conda activate spring-2026-pyt
python -m venv --system-site-packages .venv
.venv/bin/pip install pesq==0.0.4 pystoi==0.4.1
source .venv/bin/activate
```

## Repository layout

| Path | Contents |
|---|---|
| `cli/submission/model.py` | The model: DCCRN-E with `build_enhancer(sample_rate)` and `enhance(waveform, sample_rate)`, as the course interface requires. |
| `cli/submission/model.safetensors` | Submitted weights (c = 0.3 model). |
| `cli/infer.py` | Inference for one file or a folder (this project). |
| `cli/enhance.py`, `evaluate.py`, `score.py`, `metrics.py`, `common.py`, `tests/` | Course-provided runner, evaluator, and tests (unmodified). |
| `train/prepare_data.py` | Pair verification and the speaker-held-out train/dev split. |
| `train/train.py` | Training with per-epoch dev scoring and model selection. |
| `train/compare.py` | Dev comparison: main table, paired bootstrap intervals, SNR/noise breakdowns, listening shortlist. |
| `jobs/train.sh`, `jobs/eval_dev.sh` | SCC batch scripts for training and official-chain dev evaluation. |
| `data/manifest.csv` | Split manifest: one row per utterance with split, noise, SNR, lengths, and SHA-256 hashes. |
| `handout/` | Course handout. |

## Model

`cli/submission/model.py` follows the DCCRN-E configuration and the authors'
reference code:

- STFT: 400-sample (25 ms) sqrt-Hann window, 100-sample hop, 512-point FFT.
- Encoder: six complex conv layers, channels 32-64-128-256-256-256, kernel
  (5,2), stride (2,1), each followed by BatchNorm and PReLU.
- Bottleneck: two-layer complex LSTM (128 real + 128 imaginary units) with a
  1024-unit projection.
- Decoder: mirrored complex transposed convolutions with skip connections.
- E mask: a tanh-bounded magnitude plus a phase rotation applied to the noisy
  spectrum, then the inverse STFT to the exact input length.
- Causal in time.

The paper's text lists channels 32-64-128-128-256-256, but its reported 3.7M
parameters match the reference code's 32-64-128-256-256-256 (3.67M), which is
used here. The module docstring lists the remaining differences from the
reference code.

## Reproducing training

All commands run from the repository root.

1. **Verify the data and build the split.** This checks every pair for rate,
   format, equal length, zero cross-correlation lag, and SNR consistency with
   the corpus log. Dev holds out speakers p226 (male) and p230 (female):
   745 utterances. Train keeps the other 26 speakers: 10,827 utterances.
   ```sh
   python train/prepare_data.py   # writes data/manifest.csv and data/dev/
   ```
2. **Train** on one L40S. Each run takes about 2.5 GPU-hours and 6.6 GiB peak
   memory.
   ```sh
   qsub -N dccrn    -v RUN=dccrn_seed0,SEED=0          jobs/train.sh
   qsub -N comp_c03 -v RUN=dccrn_comp_c03_seed0,SEED=0 jobs/train.sh --loss sisnr+comp
   qsub -N comp_c1  -v RUN=dccrn_comp_c1_seed0,SEED=0  jobs/train.sh \
       --loss sisnr+comp --comp-power 1 --comp-weight 100
   ```
   Settings:
   - Adam at lr 1e-3, batch 16, aligned 4 s random crops, gradient clipping
     at 5.
   - The learning rate halves after 2 epochs without dev-objective
     improvement.
   - **Model selection:** keep the epoch with the best dev A(D).
   - Training stops early after 10 epochs without improvement.
   - λ = 500 (c = 0.3) and λ = 100 (c = 1) give the compressed term about the
     same gradient norm as SI-SNR at initialisation (see `COMP_WEIGHT` in
     `train/train.py`).
   - Resubmitting with the same `RUN` resumes from `runs/$RUN/last.pt`.
   - Outputs go to `runs/$RUN/`: `model.safetensors`, `history.jsonl`,
     `config.json` (with git commit), and `last.pt`.
3. **Evaluate** through the official enhance -> evaluate chain on noisy and
   clean dev inputs, then **compare**:
   ```sh
   qsub -v RUN=dccrn_seed0 jobs/eval_dev.sh    # likewise for each run
   python train/compare.py dccrn=runs/dccrn_seed0/dev_eval \
       comp_c03=runs/dccrn_comp_c03_seed0/dev_eval comp_c1=runs/dccrn_comp_c1_seed0/dev_eval
   ```

## Dev results (one seed per system)

| System | PESQ | STOI | SI-SDR (dB) | A(D) noisy | A(D) clean input | Dev score |
|---|---|---|---|---|---|---|
| Noisy input | 1.554 | 0.826 | 6.27 | 0.433 | — | — |
| DCCRN (seed paper, SI-SNR) | 2.414 | 0.872 | 15.59 | 0.615 | 0.920 | 66.09 |
| + compressed loss, c = 0.3 (submitted) | 2.681 | 0.884 | 15.57 | 0.649 | 0.931 | 69.12 |
| + uncompressed loss, c = 1 (ablation) | 2.423 | 0.875 | 15.56 | 0.617 | 0.927 | 66.33 |

Dev score is the leaderboard formula applied to dev:
100 (0.85 A(noisy) + 0.15 A(clean input)).

Against DCCRN, c = 0.3 changes dev score by +3.03 (95% paired bootstrap
interval [+2.88, +3.19] over utterances) and improves 92% of utterances.
Model selection also used dev, so these are not held-out numbers.

## Known limitation

The model has no output limiter, because the rules forbid clipping or
renormalising outputs. On every speech input we tested (all dev files and the
13 loudest training files, input peaks up to 0.988) output peaks stay at or
below 0.72. A synthetic full-scale square wave, however, produces an output
peak of about 1.23, which the runner rejects.
