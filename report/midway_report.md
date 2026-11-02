---
header: Deep Learning Fall 2026, Midway Report
title: Does a Compressed Spectral Loss Help DCCRN Enhance Speech?
authors: Bharath Srividhya, Jason Redick, Ioannis Kontothanassis
emails: "{TODO_EMAILS}@bu.edu"
---

<!-- figure:fig1_method.png
Figure 1. DCCRN pipeline. The network and data stay the same as the baseline. Our one change is the training loss on the output (blue box).
-->

## 1. Task and Hypothesis

Our task is single channel speech enhancement. The input is a noisy 16 kHz recording and the output is an estimate of the clean speech with the same length. The baseline is DCCRN-E [1], a complex convolutional recurrent network trained with the SI-SNR loss. SI-SNR rewards waveform accuracy. It does not directly reward the spectral detail that perceptual metrics like PESQ care about.

Our single change is the loss. We add a power law compressed spectral loss [2] to SI-SNR. Compressing magnitudes with an exponent c < 1 gives quiet time frequency bins more weight than plain MSE does. Our hypothesis is that this improves PESQ and STOI without hurting SI-SDR. We test c = 0.3 and an ablation with c = 1. The c = 1 run adds the same kind of spectral term with no compression, so it isolates the compression itself.

## 2. Method

Figure 1 shows the pipeline. We follow the DCCRN-E configuration and the authors' reference code. The STFT uses a 400 sample sqrt Hann window, a 100 sample hop and a 512 point FFT. Six complex conv layers (32, 64, 128, 256, 256, 256 channels) feed a two layer complex LSTM and a mirrored complex decoder with skip connections. The E mask applies a tanh bounded magnitude and a phase rotation to the noisy spectrum. The model is causal and has 3.67M parameters.

The proposed loss is L = L<sub>SI-SNR</sub> + λ L<sub>comp</sub>(c). L<sub>comp</sub> mixes a compressed magnitude MSE (weight 0.7) and a compressed complex MSE (weight 0.3) [2]. We set λ = 500 for c = 0.3 and λ = 100 for c = 1. Each value gives the spectral term about the same gradient norm as SI-SNR at initialization.

All three systems use the same training recipe. We use Adam at learning rate 1e-3, batch 16, aligned 4 s random crops and gradient clipping at 5. The learning rate halves after 2 epochs with no dev improvement. We keep the epoch with the best dev A(D) and stop after 10 epochs with no improvement.

## 3. Data and Metrics

We use VoiceBank-DEMAND [3] at 16 kHz, 11,572 noisy and clean pairs. We hold out two speakers for dev, p226 (male) and p230 (female), which gives 745 dev and 10,827 train utterances. Dev has 10 noise types at 0, 5, 10 and 15 dB SNR. The noise types and SNRs also appear in training, so dev measures unseen speakers only.

We report PESQ, STOI and SI-SDR, and the course composite A(D), which averages normalized PESQ, STOI and SI-SDR with weights 0.4, 0.3 and 0.3. The dev score is 100 (0.85 A(noisy) + 0.15 A(clean input)), the leaderboard formula applied to dev. All scores come from the official enhance.py and evaluate.py chain.

## 4. Baseline and Initial Results

Table 1 gives dev results. Each system is one training seed.

<!-- table1 -->
| System | PESQ | STOI | SI-SDR | A noisy | A clean | Score |
|---|---|---|---|---|---|---|
| Noisy input | 1.554 | 0.826 | 6.27 | 0.433 | | |
| DCCRN (SI-SNR) | 2.414 | 0.872 | 15.59 | 0.615 | 0.920 | 66.09 |
| + comp, c = 0.3 | **2.681** | **0.884** | 15.57 | **0.649** | **0.931** | **69.12** |
| + comp, c = 1 | 2.423 | 0.875 | 15.56 | 0.617 | 0.927 | 66.33 |

*Table 1. Dev results, 745 utterances, one seed each. SI-SDR in dB. "A clean" scores the model run on clean input.*

We compare each system to DCCRN on the same utterances with a paired bootstrap over utterance IDs (10,000 resamples, 95% intervals). The c = 0.3 model raises the dev score by +3.03 [+2.88, +3.19] and improves A(D) on 92% of utterances. PESQ rises by +0.266 [+0.252, +0.281] and STOI by +0.0116 [+0.0103, +0.0129]. SI-SDR does not change: −0.02 [−0.05, +0.01] dB. The c = 1 ablation gains only +0.25 [+0.15, +0.34] in score and improves 55% of utterances. So most of the gain comes with compression, not from adding a spectral term alone. These intervals cover utterance sampling only. They say nothing about seed to seed variation.

<!-- figure:fig2_snr.png
Figure 2. Dev A(D) on noisy input by input SNR.
-->

Figure 2 shows that c = 0.3 beats DCCRN at every SNR. The gain is smallest at 0 dB: +0.024 [+0.021, +0.028] in A(D), against +0.034 to +0.038 at 5 to 15 dB. Figure 3 shows that c = 0.3 helps on all 10 noise types. The c = 1 ablation is mixed. It is worse than DCCRN on car noise (−0.0077 [−0.0113, −0.0044]).

<!-- figure:fig3_noise.png
Figure 3. Paired change in dev A(D) against DCCRN by noise type, with 95% bootstrap intervals.
-->

## 5. Metric Checks

- **Data.** All 11,572 pairs pass our checks on sample rate, format, equal length and SNR against the corpus log. Cross correlation finds zero lag for every pair (train/prepare_data.py, data/manifest.csv).
- **Scoring.** The official runner and evaluator reproduce the scores from our training loop exactly.
- **Outputs.** On dev, no output peak goes above 1.0 and no PESQ call fails.
- **Clean input.** Running the model on clean dev speech keeps A(D) at 0.920 (DCCRN) and 0.931 (c = 0.3).
- **Edge cases.** We ran the submitted checkpoint on 22 synthetic inputs: silence, very quiet audio, a 120 s file, files from 1 to 1,000 samples, and full scale signals (cli/edge_cases.py). Every output keeps the input length, is finite, and is identical across two runs. Three full scale inputs fail the peak check: a hard clipped speech like signal (output peak 1.17), a 1 kHz sine (1.03) and a 200 Hz square wave (1.46). The runner would reject these outputs.

## 6. Resource Use

| Item | Value |
|---|---|
| Parameters | 3,671,053 (limit 15M) |
| Checkpoint size | 14.7 MB |
| Peak training memory | 6.6 GiB on one L40S (limit 32 GiB) |
| Training time | 2.5 to 2.8 GPU hours per run |
| Real time factor | about 0.006 on L40S, 0.10 on an Apple M2 CPU |

## 7. Risks

- The DCCRN baseline may have trained for fewer epochs than the other runs, because a disk quota crash interrupted it.
- Each system has one seed, so we cannot yet separate the method effect from seed variation.
- We select the best epoch on dev, so dev scores are slightly optimistic.
- The gain is smallest at 0 dB, the hardest condition.
- The model has no output limiter, and the rules do not let the runner clip. Loud tonal or clipped inputs can push the output above 1.0 (Section 5).

## 8. Planned Ablations and Next Steps

1. Train a second seed of DCCRN and c = 0.3 to measure seed variation.
2. Sweep λ around 500 to check the gain is not tied to one weight.
3. Listen to the shortlisted utterances where PESQ and SI-SDR disagree and plot their spectrograms. This also covers the largest wins and losses.
4. Find the input level where full scale inputs start to exceed 1.0, and decide how to handle it within the rules.
5. If time allows, try c = 0.5 to see how the gain depends on the exponent.

## 9. Contributions

- **Ioannis Kontothanassis:** DCCRN model, training script and the compressed loss, most training runs, the dev comparison script and the submitted checkpoint.
- **Jason Redick:** SCC batch job scripts, checkpoint resume fix, and dev evaluation through the official chain.
- **Bharath Srividhya:** helped design and debug the model and training code in shared sessions, wrote the figure script (report/make_figures.py) and the edge case test (cli/edge_cases.py), ran the edge case checks, and wrote this report.

**AI use.** We used Claude to help write code for the figure and edge case scripts and to check grammar. We reviewed and ran all of this code ourselves.

## References

1. Y. Hu, Y. Liu, S. Lv, M. Xing, S. Zhang, Y. Fu, J. Wu, B. Zhang, L. Xie. DCCRN: Deep Complex Convolution Recurrent Network for Phase Aware Speech Enhancement. Interspeech, 2020.
2. S. Braun, I. Tashev. A Consolidated View of Loss Functions for Supervised Deep Learning Based Speech Enhancement. TSP, 2021.
3. C. Valentini-Botinhao, X. Wang, S. Takaki, J. Yamagishi. Investigating RNN Based Speech Enhancement Methods for Noise Robust Text to Speech. SSW, 2016.
