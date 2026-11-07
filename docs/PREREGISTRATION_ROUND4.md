# Pre-registration, round 4: the contrastive term in the authors' own code

Written 8 Nov 2026 and pushed to the public repository before any round-4 run. The commit time on GitHub is the timestamp. Nothing below changes after the first run starts. If something must change, it is added as a dated amendment and reported in the paper as a deviation.

## 1. Question
Rounds 1-3 tested the contrastive term in **our reimplementation** of M2-CL. Our model has 6 extraction blocks, a per-block loss and batches pooled across domains. The released code (github.com/aristotelisballas/m2cl) has 21 pipelines, a per-pipeline loss and batches balanced by domain. Round 4 asks the round-3 question of the authors' code itself: **does the contrastive term change target accuracy on PACS?** We train the authors' `M2CL` algorithm with its loss weight `lparam` at 0.01 and at 0, everything else equal and paired by seed.

## 2. Hypotheses (same rules as round 3)
- **H1 (effect):** the paired difference (lparam 0.01 minus lparam 0) is not zero. Test: exact sign-flip test on the 12 paired differences, two-sided, alpha 0.05.
- **H2 (equivalence):** the difference lies within +/-0.71 pp, the M2-CL gain over M2 that the paper reports (Table I, ResNet-18 PACS: 83.54 - 82.83). Test: TOST on the paired differences, alpha 0.05.
- **Decision**, applied by `scripts/analyze_round4.py`, which reuses round 3's `paired()` and `decide()`:
  - EFFECT if H1 rejects;
  - EQUIVALENCE if H2 rejects;
  - both, if both reject;
  - otherwise INCONCLUSIVE.
- The decision is reported whatever it is.

## 3. Fixed settings
| Setting | Value | Why |
|---|---|---|
| Code | `aristotelisballas/m2cl`, commit `0b1bce3026d9d56369037129e36f81e7645b80e6`, unmodified | The authors' release; the same commit as audit F12 |
| Entry point | `python -m domainbed.scripts.train --dataset PACS --algorithm M2CL` | The README's command for M2-CL |
| Backbone | `--hparams '{"resnet18": true}'` | The paper's ResNet-18 results; the code defaults to ResNet-50 |
| Loss weight | `--lparam 0.01` (M2-CL) and `--lparam 0.0` (term off) | 0.01 is the paper's value and the code's fallback; 0 removes the term without changing the network, the initialization or the data order |
| Temperature | `--temp 1.0` | Paper and code default. It must be passed explicitly: `temp` and `lparam` are not in the code's hparams registry, so `M2CL` raises a KeyError without them |
| Learning rate | `--lr 0.001` (SGD, momentum 0.9, weight decay 5e-4 as coded) | The paper's stated value. The registry default, 5e-5, is DomainBed's default for Adam; `M2CL` uses SGD |
| Batch | Code default: 32 per source domain, 96 in total | Unchanged from the code. The paper says 128; not changed, to keep the code as released |
| Steps, evaluation | Code default for PACS: 5,001 steps, evaluation every 300 steps | Unchanged |
| Hyperparameter seed | `--hparams_seed 0` (default hparams) | No random search |
| Seeds | `--trial_seed s --seed s` for s in {0, 1, 2} | trial_seed sets the 80/20 split; seed sets the initialization and data order. Pairs share both |
| Test domains | `--test_envs` 0, 1, 2, 3 (art_painting, cartoon, photo, sketch; sorted folder order) | Leave-one-domain-out |
| Data | DomainBed's PACS: 9,991 images; art_painting 2,048, cartoon 2,344, photo 1,670, sketch 3,929; exactly these four folders | Downloaded from the Hugging Face mirror (`PTeterwak/DomainBed_OOP`, `all/PACS.tar.gz`). The runner refuses any other counts |
| Model selection | Training-domain validation, as in DomainBed's `IIDAccuracySelectionMethod`: the checkpoint with the highest mean `env{j}_out_acc` over the three source domains (the first one on a tie). Its `env{test}_in_acc` is the test accuracy | DomainBed's standard; target data are never used to select |
| Hardware | One machine: the institute's RTX 4000 Ada (20 GB) with an Intel Core Ultra 9. All 24 runs run there | Every pair runs on the same hardware and software. Versions are recorded in `env/` |

Runs: 2 settings x 4 test domains x 3 seeds = **24**. The runner is `scripts/run_round4_authors.py`. Fold and seed are the outer loops, so each pair finishes together. Runs are independent processes, and `--parallel` only sets how many run at once.

## 4. Manipulation check (reported, never used to select or exclude runs)
1. **Gradient share at the ImageNet start:** 0.109-0.127 (audit F12; threshold 0.05). Already measured, so it passes before any run.
2. **Paired weight distance.** The L2 distance between the final models (`model.pkl`) of each pair, over the network's learned non-extraction parameters. Excluded: parameter names containing `ext`, and BatchNorm running statistics.
   - Reference: the median distance between seeds within the same fold in the lparam 0 arm.
   - **The term counts as active if the paired distance exceeds 10% of that reference in at least 10 of 12 pairs.** For comparison: in our implementation, the inert term gave 3-4% and the active term about 40%.

## 5. Secondary results (descriptive, no tests)
- Mean +/- SD per domain and the macro average for each arm.
- The lparam 0.01 arm against the paper's ResNet-18 M2-CL, 83.54 macro. This shows whether the released code reproduces the published number at these settings.
- The lparam 0 arm next to the paper's M2 (82.83). It is labelled as the same network without the term, not the authors' separate `M2` model.

## 6. Failures and exclusions
- **Interrupted runs:** the code cannot resume. An interrupted run is deleted and restarted from scratch, and is still deterministic in its seeds.
- **No exclusions:** no completed run is dropped.
- **If the code cannot run in a fresh environment**, the only allowed fixes are at the environment level: package versions, installing missing packages such as `wilds` and `timm`, which the code imports. Each is recorded in `env/`. Any change to the authors' files is a deviation: it is written as an amendment before any run uses it, and reported.

## 7. Known limits, stated in advance
- With 12 pairs, the round-3 spread (SD about 1.2 pp) gives a minimum detectable effect of about 1 pp at 80% power. An effect the size of the claimed 0.71 pp may not be detected.
- One batch size and one learning rate; no hyperparameter search.
- This round uses a different GPU and PyTorch build from round 3. Its numbers are compared with round 3 only descriptively, never pooled.
