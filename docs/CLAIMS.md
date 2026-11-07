# Claims and evidence: `paper/M2CL_v3.tex`

Every quantitative claim in the draft, with where it comes from and how to reproduce it. "Zips" are in `C:\Users\Madhusudan\Downloads`. Audit sections refer to `docs/AUDIT.md`.

## Loss-scaling error
| Claim | Evidence | Reproduce |
|---|---|---|
| Mean reduction shrank the term about 28-fold (7 classes x 4 layers) | Unit test comparing the corrected loss with a verbatim copy of the notebook loss | `pytest tests/test_torch_parts.py` |
| 4-tap M2 and M2-CL backbones lie 0.034-0.041 apart vs 1.15-1.28 moved from ImageNet | E2 | `scripts/compare_checkpoint_weights.py` |

## Inert term at alpha 0.003 (Section "Does the Corrected Contrastive Term Change Training?")
| Claim | Evidence | Reproduce |
|---|---|---|
| Gradient share 0.004 (backbone), table of tau x alpha | F1 | `scripts/audit_grad_ratio.py <PACS> <checkpoint>` |
| Loss is about 170 with no class structure, falls only to about 133 at tau 1 (108 at 0.5, 89 at 0.07) | F10 | `scripts/audit_loss_bounds.py <repo>` |
| Logged loss stayed between 168.4 and 171.4 in all 12 runs | F10 | result.json histories in `m2cl_v2.zip` |
| 12 pairs: mean +0.04 pp, sign-flip p 0.88, 95 % CI -0.29 to +0.37 | F9, F11 | `scripts/audit_pair_runs.py`, `scripts/make_fig_pairs.py m2_v2.zip m2cl_v2.zip out.pdf` |
| TOST within +/-0.71 pp, p = 0.0005 (margin = the paper's own claimed gain, 83.54 - 82.83) | This session; the margin's source is the paper's Table I | `make_fig_pairs.py` prints the t-statistics; p from scipy |
| Paired backbones 0.016-0.033 apart (0.074 for the resumed pair) vs 0.73-0.95 between seeds | F9 | `scripts/audit_pair_runs.py` |
| Max per-epoch val gap 0.48 pp; final CE within 0.0003 in 11 pairs (0.0046 in one) | F10 | result.json histories |
| Fig. 1 | F11 | `scripts/make_fig_pairs.py` |

## Authors' implementation
| Claim | Evidence | Reproduce |
|---|---|---|
| Public code: share 0.109-0.127 (backbone), 0.059-0.071 (blocks); ours 0.0045-0.0055 at alpha 0.003, 0.015-0.018 at alpha 0.01; about 24 times | F12 | `scripts/audit_authors_grad_share.py <repo>/domainbed <ours> <PACS> 32 2` |
| 21 pipelines, per-pipeline loss on fc1 pre-activation, tau outside the exponential, per-domain batches, defaults | F4, F12 | Read `domainbed/mymodels/m2cl_18.py`, `lib/myloss.py`, `algorithms.py` (commit 0b1bce3) |

## Dose-response, learning rate, ERM (round 2)
| Claim | Evidence | Reproduce |
|---|---|---|
| Dose table: shares 0.003 / 0.01 / 0.19-0.20 / 0.86-0.89 / 4.1-10.8; distances 0.032-2.170; val 94.64 to 95.35; test 73.05-74.22 | F13 | `scripts/audit_round2.py <Downloads> <PACS> --grad` |
| lr 1e-3: +2.49 / +2.52 / +0.42 / +4.99, macro +2.60; val higher on all folds | F13 | same |
| Corrected-split ERM: macro 75.08; M2 - ERM +0.71 (CI -1.12 to +2.53, p 0.43); M2-CL - ERM +0.75 (p 0.43) | F13 | same |
| Split correction moved M2-CL macro 76.19 -> 75.83 | F9, Table V of the old draft | `results/legacy_v1/paper_accuracy_runs.csv` vs `m2cl_v2.zip` |

## Leakage (Sections "Domain-Leakage Investigation", "Robustness")
| Claim | Evidence | Reproduce |
|---|---|---|
| Random-init 71.0-92.1 %, ImageNet 96.1-99.6 %, majority 38.6-51.4 % | A0 re-run | `results/tables/raw_backbone/leak_arch_bias.csv` |
| 22 trained-vs-ImageNet deltas negative, 2 BH-significant | A0 re-run | `leak_v1_vs_imagenet.csv`, `leak_m2cl6_v1_vs_imagenet.csv` |
| DCT copy vs RGB copy | A0 | `scripts/audit_dct_data.py`, `scripts/verify_checkpoints.py` |

## Values that cannot be verified (stated as such in the draft)
- Table I, Sketch, M2 61.06 and M2-CL 61.03: no checkpoint found (F3).
- The earlier contaminated-split ERM seeds, apart from Sketch seeds 1-2 (`60f0ba3dff02.zip`, `2671d69ff6bb.zip`).
- The paper's own numbers: taken from arXiv 2308.14418, Table I (ResNet-18 block).

## Decisions without supporting evidence (marked "not justified" in the appendix)
- alpha 0.003 (chosen on a 2-image validation lead), lr 5e-4 (carried over), cosine schedule (our choice, untested).

## Pre-registered test
- Plan, hypotheses, margin and decision rules: `docs/PREREGISTRATION.md`.
- Analysis code, fixed in advance: `scripts/analyze_round3.py`.
- Settings: `configs/round3_m2.yaml`, `round3_m2cl.yaml`, `round3_erm.yaml`.
