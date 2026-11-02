Method diagram: DCCRN's pipeline (STFT → complex encoder → complex LSTM → complex decoder → E-mask → iSTFT), with your change marked as a loss on the output: SI-SNR + λ × compressed spectral loss.
Baseline and initial results: the main table and the paired intervals in runs/comparison.md. Include the noisy-input row, the DCCRN baseline, and both loss variants, as the handout requires. Label them as dev-set results from one seed each.
Metric checks:
All 11,572 pairs passed the alignment and integrity checks, with zero lag (train/prepare_data.py).
The official runner and evaluator reproduce the training-loop scores exactly.
No output peaks above 1.0 and no PESQ failures.
The clean-input preservation results.
Resource use:
3.67M parameters, under the 15M limit.
Peak training memory of 6.6 GiB on the L40S, under the 32 GiB limit.
2.5–2.8 GPU-hours per run, from history.jsonl.
RTF about 0.006 on the L40S, from run.json.
Risks:
The baseline may have trained fewer epochs after the quota crash.
One seed per system.
Dev-based model selection makes the absolute scores slightly optimistic.
The gain is smaller at 0 dB.
Planned ablations and next steps: you've already run the c = 1 ablation. Possible follow-ups are a second seed, a sweep over λ, and listening plus spectrogram analysis of the shortlisted utterances.
Contributions: who did what across the three of you, as the handout's contribution log asks.

# Dev comparison (745 utterances, speakers p226, p230)

## 1. Main table

Noisy-in columns score enhanced noisy speech; clean-in A(D) scores the model run on clean speech (clean preservation). Score = 100(0.85 A_noisy + 0.15 A_clean), the leaderboard formula applied to dev.

| System | PESQ | STOI | SI-SDR (dB) | A(D) noisy-in | A(D) clean-in | Score |
|---|---|---|---|---|---|---|
| noisy input | 1.554 | 0.8259 | 6.27 | 0.4331 | (identity) | |
| dccrn | 2.414 | 0.8720 | 15.59 | 0.6151 | 0.9201 | 66.09 |
| comp_c03 | 2.681 | 0.8836 | 15.57 | 0.6489 | 0.9309 | 69.12 |
| comp_c1 | 2.423 | 0.8748 | 15.56 | 0.6168 | 0.9270 | 66.33 |

## 2. Paired differences vs `dccrn` (95% bootstrap CI over utterance IDs, 10000 resamples)

An interval that excludes 0 means the difference is unlikely to be utterance-sampling noise. It says nothing about seed-to-seed variation.

| System | ΔPESQ | ΔSTOI | ΔSI-SDR (dB) | ΔA noisy-in | ΔA clean-in | ΔScore | better A on |
|---|---|---|---|---|---|---|---|
| comp_c03 | +0.266 [+0.252, +0.281] | +0.0116 [+0.0103, +0.0129] | -0.02 [-0.05, +0.01] | +0.0338 [+0.0320, +0.0356] | +0.0108 [+0.0095, +0.0122] | +3.03 [+2.88, +3.19] | 92% |
| comp_c1 | +0.009 [+0.000, +0.018] | +0.0028 [+0.0020, +0.0036] | -0.02 [-0.04, +0.00] | +0.0017 [+0.0006, +0.0028] | +0.0069 [+0.0059, +0.0079] | +0.25 [+0.15, +0.34] | 55% |

## 3a. By SNR (dB, as logged): mean PESQ / A(D) noisy-in

| snr_db | n | noisy input | dccrn | comp_c03 | comp_c1 | Δcomp_c03 A | Δcomp_c1 A |
|---|---|---|---|---|---|---|---|
| 0.0 | 185 | 1.21 / 0.314 | 1.84 / 0.502 | 2.01 / 0.526 | 1.83 / 0.502 | +0.0243 [+0.0209, +0.0279] | +0.0005 [-0.0016, +0.0026] |
| 5.0 | 187 | 1.37 / 0.387 | 2.19 / 0.578 | 2.50 / 0.617 | 2.21 / 0.581 | +0.0384 [+0.0347, +0.0417] | +0.0032 [+0.0009, +0.0054] |
| 10.0 | 186 | 1.64 / 0.472 | 2.62 / 0.657 | 2.92 / 0.694 | 2.63 / 0.658 | +0.0379 [+0.0343, +0.0416] | +0.0017 [-0.0007, +0.0040] |
| 15.0 | 187 | 1.99 / 0.558 | 3.00 / 0.723 | 3.28 / 0.758 | 3.01 / 0.725 | +0.0344 [+0.0309, +0.0379] | +0.0013 [-0.0010, +0.0036] |

## 3b. By noise type: mean PESQ / A(D) noisy-in

| noise | n | noisy input | dccrn | comp_c03 | comp_c1 | Δcomp_c03 A | Δcomp_c1 A |
|---|---|---|---|---|---|---|---|
| babble | 74 | 1.33 / 0.382 | 2.01 / 0.535 | 2.26 / 0.569 | 2.04 / 0.540 | +0.0338 [+0.0270, +0.0407] | +0.0047 [+0.0012, +0.0083] |
| cafeteria | 75 | 1.40 / 0.406 | 2.06 / 0.560 | 2.33 / 0.593 | 2.07 / 0.562 | +0.0330 [+0.0271, +0.0394] | +0.0014 [-0.0015, +0.0045] |
| car | 76 | 2.67 / 0.595 | 3.41 / 0.780 | 3.74 / 0.820 | 3.34 / 0.772 | +0.0396 [+0.0353, +0.0443] | -0.0077 [-0.0113, -0.0044] |
| kitchen | 74 | 1.61 / 0.468 | 2.99 / 0.722 | 3.34 / 0.765 | 3.03 / 0.727 | +0.0428 [+0.0377, +0.0480] | +0.0044 [+0.0003, +0.0088] |
| meeting | 74 | 1.48 / 0.425 | 2.19 / 0.581 | 2.44 / 0.611 | 2.19 / 0.581 | +0.0301 [+0.0239, +0.0361] | -0.0000 [-0.0038, +0.0039] |
| metro | 74 | 1.67 / 0.452 | 2.51 / 0.638 | 2.79 / 0.672 | 2.54 / 0.641 | +0.0342 [+0.0289, +0.0393] | +0.0031 [-0.0007, +0.0071] |
| restaurant | 76 | 1.25 / 0.378 | 1.95 / 0.534 | 2.15 / 0.561 | 1.97 / 0.538 | +0.0263 [+0.0208, +0.0321] | +0.0031 [+0.0004, +0.0060] |
| ssn | 75 | 1.28 / 0.383 | 2.23 / 0.572 | 2.52 / 0.609 | 2.26 / 0.576 | +0.0370 [+0.0319, +0.0424] | +0.0041 [+0.0015, +0.0067] |
| station | 74 | 1.35 / 0.408 | 2.25 / 0.587 | 2.52 / 0.621 | 2.27 / 0.590 | +0.0349 [+0.0295, +0.0404] | +0.0035 [-0.0002, +0.0069] |
| traffic | 73 | 1.47 / 0.431 | 2.53 / 0.641 | 2.74 / 0.667 | 2.53 / 0.641 | +0.0260 [+0.0209, +0.0313] | +0.0003 [-0.0032, +0.0039] |

## 4. Listening shortlist vs `dccrn`

Listen to these in data/dev/{clean,noisy} and each system's dev_eval/enhanced_noisy, and look at spectrograms. Disagreements show where the metrics trade off; extremes show the clearest wins and failures.

### comp_c03

**PESQ up, SI-SDR down** (357 utterances)

- p226_211 (babble, 15.0 dB): ΔPESQ +1.07, ΔSI-SDR -0.04, ΔA +0.136
- p226_132 (cafeteria, 5.0 dB): ΔPESQ +0.37, ΔSI-SDR -5.92, ΔA +0.001
- p226_076 (station, 10.0 dB): ΔPESQ +0.79, ΔSI-SDR -0.38, ΔA +0.109
- p226_131 (cafeteria, 10.0 dB): ΔPESQ +0.82, ΔSI-SDR -0.02, ΔA +0.094
- p230_056 (car, 5.0 dB): ΔPESQ +0.74, ΔSI-SDR -0.13, ΔA +0.083

**SI-SDR up, PESQ down** (21 utterances)

- p226_346 (metro, 15.0 dB): ΔPESQ -0.27, ΔSI-SDR +0.05, ΔA -0.030
- p226_326 (ssn, 15.0 dB): ΔPESQ -0.24, ΔSI-SDR +0.14, ΔA -0.025
- p230_368 (restaurant, 10.0 dB): ΔPESQ -0.14, ΔSI-SDR +0.08, ΔA -0.017
- p226_040 (traffic, 10.0 dB): ΔPESQ -0.11, ΔSI-SDR +0.19, ΔA -0.015
- p230_305 (cafeteria, 10.0 dB): ΔPESQ -0.12, ΔSI-SDR +0.11, ΔA -0.015

**largest A gains**

- p226_211 (babble, 15.0 dB): ΔPESQ +1.07, ΔSI-SDR -0.04, ΔA +0.136
- p226_076 (station, 10.0 dB): ΔPESQ +0.79, ΔSI-SDR -0.38, ΔA +0.109
- p230_106 (meeting, 15.0 dB): ΔPESQ +0.60, ΔSI-SDR +3.31, ΔA +0.104
- p230_216 (babble, 10.0 dB): ΔPESQ +0.78, ΔSI-SDR +0.08, ΔA +0.101
- p226_192 (metro, 5.0 dB): ΔPESQ +0.76, ΔSI-SDR +0.15, ΔA +0.100

**largest A losses**

- p226_039 (traffic, 15.0 dB): ΔPESQ -0.12, ΔSI-SDR -0.14, ΔA -0.039
- p226_346 (metro, 15.0 dB): ΔPESQ -0.27, ΔSI-SDR +0.05, ΔA -0.030
- p230_005 (babble, 0.0 dB): ΔPESQ -0.07, ΔSI-SDR -0.61, ΔA -0.029
- p230_260 (babble, 0.0 dB): ΔPESQ -0.18, ΔSI-SDR -0.97, ΔA -0.025
- p226_326 (ssn, 15.0 dB): ΔPESQ -0.24, ΔSI-SDR +0.14, ΔA -0.025

### comp_c1

**PESQ up, SI-SDR down** (202 utterances)

- p226_139 (kitchen, 10.0 dB): ΔPESQ +0.47, ΔSI-SDR -0.02, ΔA +0.058
- p226_211 (babble, 15.0 dB): ΔPESQ +0.46, ΔSI-SDR -0.18, ΔA +0.051
- p230_103 (kitchen, 10.0 dB): ΔPESQ +0.35, ΔSI-SDR -0.01, ΔA +0.044
- p226_099 (kitchen, 5.0 dB): ΔPESQ +0.33, ΔSI-SDR -0.01, ΔA +0.038
- p226_332 (station, 5.0 dB): ΔPESQ +0.31, ΔSI-SDR -0.11, ΔA +0.034

**SI-SDR up, PESQ down** (122 utterances)

- p230_355 (kitchen, 10.0 dB): ΔPESQ -0.36, ΔSI-SDR +0.69, ΔA -0.036
- p230_351 (car, 10.0 dB): ΔPESQ -0.34, ΔSI-SDR +0.24, ΔA -0.038
- p226_358 (meeting, 10.0 dB): ΔPESQ -0.31, ΔSI-SDR +0.25, ΔA -0.032
- p226_181 (kitchen, 10.0 dB): ΔPESQ -0.29, ΔSI-SDR +0.03, ΔA -0.032
- p226_001 (babble, 15.0 dB): ΔPESQ -0.27, ΔSI-SDR +0.01, ΔA -0.032

**largest A gains**

- p226_139 (kitchen, 10.0 dB): ΔPESQ +0.47, ΔSI-SDR -0.02, ΔA +0.058
- p226_097 (kitchen, 15.0 dB): ΔPESQ +0.31, ΔSI-SDR +2.06, ΔA +0.053
- p230_321 (metro, 15.0 dB): ΔPESQ +0.35, ΔSI-SDR +0.96, ΔA +0.052
- p226_211 (babble, 15.0 dB): ΔPESQ +0.46, ΔSI-SDR -0.18, ΔA +0.051
- p230_043 (traffic, 10.0 dB): ΔPESQ +0.40, ΔSI-SDR +0.21, ΔA +0.047

**largest A losses**

- p226_096 (car, 0.0 dB): ΔPESQ -0.69, ΔSI-SDR -0.52, ΔA -0.090
- p226_066 (metro, 0.0 dB): ΔPESQ -0.36, ΔSI-SDR -0.67, ΔA -0.051
- p226_331 (station, 10.0 dB): ΔPESQ -0.40, ΔSI-SDR -0.03, ΔA -0.046
- p230_261 (cafeteria, 15.0 dB): ΔPESQ -0.35, ΔSI-SDR -0.20, ΔA -0.044
- p230_183 (car, 0.0 dB): ΔPESQ -0.39, ΔSI-SDR -0.05, ΔA -0.042

