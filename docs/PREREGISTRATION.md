# Pre-registration: round 3, the decisive M2 vs M2-CL test

Written 7 Nov 2026, before any round-3 run started. Everything below (hypotheses, settings, primary analysis, decision rules) is fixed now. The analysis code is fixed too: `scripts/analyze_round3.py`. Anything decided after seeing round-3 results will be reported as exploratory.

**Proof of timing.** Commit this file, the three `configs/round3_*.yaml` files, `scripts/analyze_round3.py`, `notebooks/Round3_Decisive_Colab.ipynb`, `notebooks/Round3_Kaggle.ipynb` and `scripts/run_part_c.py`, and push them to a public repository (or upload them to OSF) before starting the first run. The commit or upload timestamp is the evidence that the plan came first.

## 1. Question
Rounds 1 and 2 left one question open. Does the M2-CL contrastive term change target-domain accuracy when it is applied at the authors' strength and training is not under-powered?

- Round 1 (`docs/AUDIT.md` F9): at alpha 0.003 and lr 5e-4, the term was inert (gradient share 0.4 %).
- Round 2 (F12, F13): the authors' code applies the term at 11-13 % of the cross-entropy gradient. Our lr of 5e-4 under-trains M2 by about 2.6 pp. The dose-response was a single seed on a single fold.

## 2. Hypotheses
- **H1 (primary, superiority).** At lr 1e-3 and alpha 0.07, M2-CL and M2 differ in target accuracy. Two-sided test.
- **H2 (primary, equivalence).** The difference is smaller than the effect the original paper reports. The margin is **±0.71 pp**, the ResNet-18 PACS macro-average gain of M2-CL over M2 in Ballas & Diou (2024), Table I (83.54 - 82.83). It is fixed in advance from the paper itself, not from our data.

## 3. Fixed settings (all decisions justified)

| Setting | Value | Justification |
|---|---|---|
| Architecture | 6-tap M2 / M2-CL, ResNet-18, ImageNet start | Same as rounds 1-2, so results connect; ResNet-18 matches the paper's Table I block |
| Learning rate | 1e-3, cosine, SGD momentum 0.9, wd 5e-4 | The paper states 1e-3. Round 2 showed 1e-3 beats 5e-4 on test and on source validation on all 4 folds (F13) |
| alpha | **0.07** | Chosen to match the authors' gradient share, not tuned on accuracy. At ImageNet start, on the same batches, the authors' code gives a share of 0.109-0.127 and ours 0.0149-0.0183 at alpha 0.01. The share is linear in alpha, so alpha = 0.01 x 0.109/0.0149 = 0.073 or 0.01 x 0.127/0.0183 = 0.069. Rounded to 0.07 (F12) |
| tau | 1.0 | The paper's default. In the authors' code tau has no effect on gradients |
| Batch, epochs, augmentation | 128, 30, baseline | Unchanged from rounds 1-2 |
| Split and model selection | Corrected 80/20 source split; best mean source validation accuracy | Test images are touched once, after selection |
| Seeds | 0, 1, 2 for the primary analysis (12 fold-seed pairs) | Same seeds as rounds 1-2; a fixed n avoids optional stopping |
| Pairing | Same fold, seed, split and initialization for M2 and M2-CL | Pairing removes seed variance from the comparison (F9: the paired SD was 0.5 pp vs a seed SD of up to 2.8 pp) |

## 4. Runs
| Part | Runs | Status |
|---|---|---|
| A. Primary | M2, lr 1e-3, seeds 1-2 x 4 folds (seed 0 exists in `lr/lr_0.001`); M2-CL, alpha 0.07, lr 1e-3, seeds 0-2 x 4 folds | 20 new runs |
| B. Secondary | ERM, lr 1e-3, seeds 0-2 x 4 folds | 12 runs |
| C. Replication | VLCS: ERM, M2, M2-CL (same settings), seeds 0-2 x 4 folds | 36 runs |
| D. Robustness | Part A with seeds 3-4 | 16 runs, reported separately, never pooled into the primary test |

## 5. Primary analysis (Part A, PACS)
The unit of analysis is the 12 paired differences d = acc(M2-CL) - acc(M2) in target test accuracy, in percentage points.

1. Mean d, its 95 % t-interval, a two-sided paired t-test, and the exact sign-flip permutation test (4,096 sign patterns). H1 uses the sign-flip p-value, because it needs no normality assumption; the t-test is reported alongside.
2. TOST for H2 with margin ±0.71 pp (two one-sided paired t-tests, alpha 0.05 each, equivalent to the 90 % CI lying inside ±0.71).

**Decision rule, fixed now:**
- **Effect:** sign-flip p < 0.05. Report the direction and size, and whether the 95 % CI excludes the paper's 0.71.
- **Equivalence:** TOST p < 0.05. Report "no effect as large as the paper's claimed gain", whatever the sign of the mean.
- **Inconclusive:** neither of the above. Report it as inconclusive. Part D is then reported as a separate robustness check, not used to rescue the primary test.

Both an effect and equivalence can hold if the effect is real but smaller than 0.71 pp; both are then reported.

## 6. Manipulation checks (to confirm the term is active, not to select runs)
- The gradient share at the selected checkpoint of M2-CL Art seed 0 is at least 0.05 (`--grad`).
- The backbone L2 distance between paired M2 and M2-CL selected checkpoints is above the inert level of round 1 (0.016-0.033) in at least 10 of the 12 pairs.

If a check fails, the primary result is still reported, together with the failed check.

## 7. Secondary analyses (reported whatever the outcome)
- **S1.** Per-fold mean differences (descriptive; 3 pairs per fold cannot reach p < 0.05).
- **S2.** The learning-rate effect for M2, 1e-3 vs 5e-4, over 12 seed-matched pairs (sign-flip p). This confirms or refutes the single-seed round-2 finding.
- **S3.** M2 - ERM and M2-CL - ERM at lr 1e-3 over 12 pairs each (sign-flip p, 95 % CI).
- **S4.** VLCS: the same primary analysis and S3 on VLCS. Its margin is also ±0.71 pp, because the paper's VLCS gain is not used to set it.

**Multiple comparisons.** H1 and H2 are the confirmatory tests. The secondary tests are reported with Holm correction across S2-S4. Exploratory findings are labelled as exploratory.

## 8. Things that will not change after results are seen
alpha, lr, seeds, the margin, the selection rule, the exclusion rule (none: every finished run is used, and resumed runs are reported) and the analysis code. If a run crashes, it is resumed or restarted with the same seed. It is never replaced by a different seed.

## 9. Known limits, stated in advance
- One architecture (6 taps, per-block loss), not the authors' 21-pipeline design. alpha matches their gradient share, not their loss geometry.
- With n = 12 pairs and a paired SD of about 1 pp, power for H2 is about 50 %; with SD 0.7 pp it is about 90 %. The paired SD at an active alpha is not known in advance.
- The gradient-share match was measured at ImageNet start, on 2 batches.

## Amendment 1 (7 Nov 2026, before any run on Kaggle): compute platform
Runs execute on an NVIDIA T4, on Colab or on Kaggle; Kaggle's P100 is not used. Each part runs entirely on one platform: **A, B and D on Colab** (`notebooks/Round3_Decisive_Colab.ipynb`; all three were already running on Colab when this amendment was written), **C on Kaggle** (`notebooks/Round3_Kaggle.ipynb`). So no paired comparison mixes platforms, and the primary test (Part A) is entirely on Colab. If a part has to move, the move and its reason are reported.

The Kaggle notebook uses the same configs, run names, data checks and analysis script. It differs only in:
- file locations (Kaggle inputs are read-only and there is no Google Drive);
- run order (fold and seed are the outer loop, so whole pairs finish first; each run is a separate process that sets its own seeds, so order cannot change a result);
- a clock check that starts a run only if it can finish within Kaggle's 12-hour session.

Each Kaggle session writes its GPU, torch, torchvision, CUDA and Python versions to `m2cl_v2/env/`, and each run's `result.json` records its torch version. The paper reports the platform of every part.

## Amendment 2 (7 Nov 2026, before any Part C run on the institute machine): Part C moves to the institute GPU
All 36 Part C runs (VLCS) run on one Linux machine at IIIT Sri City (an NVIDIA RTX Blackwell-generation GPU and an Intel Core Ultra 9 CPU), using `scripts/run_part_c.py`. Its GPU, CUDA, cuDNN, torch and CPU are recorded in `m2cl_v2/env/`. Parts A, B and D stay on Colab, as in Amendment 1.

**Reason: compute only.** On Kaggle a VLCS epoch took about 125 s, because decoding VLCS's large JPEGs (358 KB on average, against 19 KB for PACS) on the two loader workers limits the speed. That is about 65 minutes per run and about 40 GPU hours in total, more than Kaggle's weekly quota.

**Timing.** The move was decided while Kaggle's first VLCS session was still in its first run (M2, Caltech101, seed 0). No M2-CL or ERM VLCS run had started, so no Part C comparison existed. If that first run finishes on Kaggle, its accuracy is used only as an exploratory hardware check: it is compared with the identical run on the institute machine and never enters a pre-registered test.

**What is unchanged:** configs, run names, data checks (DomainBed's exact image counts), truncated-image handling (`docs/AUDIT.md` F17) and the analysis script. The script can train several runs at once on the GPU (`--parallel`); each run is an independent process with its own seeds.

**Correction (7 Nov 2026, after Part C started):** the institute GPU is an NVIDIA RTX Ada-generation card (compute capability 8.9), not Blackwell as written above. Only the description was wrong; the machine, the plan and the runs are the same. The exact model is recorded in `m2cl_v2/env/institute_*.json`.
