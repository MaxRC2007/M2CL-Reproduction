# Pre-registration, round 5: the contrastive term in the authors' own code, on VLCS

Written 11 Nov 2026 and pushed to the public repository before any round-5 run. The commit time on GitHub is the timestamp. Nothing below changes after the first run starts. Any necessary change is added as a dated amendment and reported in the paper as a deviation.

## 1. Question
- **Round 3:** on VLCS, our reimplementation gave M2-CL minus M2 = +0.08 pp (95% CI -0.37 to +0.53). That is equivalent to zero within +/-0.71 pp and excludes the paper's VLCS gain of +1.67 pp (Table I: 77.78 - 76.11).
- **Round 4:** on PACS, the authors' code gave +0.81 pp (-0.28 to +1.91), inconclusive.
- **Round 5 asks:** does the contrastive term change target accuracy on VLCS when it runs in the authors' own code?

It is round 4 with the dataset changed and nothing else. The authors' `M2CL` algorithm is trained with `lparam` at 0.01 and at 0, everything else equal and paired by seed.

## 2. Hypotheses (same rules as rounds 3 and 4)
- **H1 (effect):** the paired difference (lparam 0.01 minus lparam 0) is not zero. Test: exact sign-flip test on the 12 paired differences, two-sided, alpha 0.05.
- **H2 (equivalence):** the difference lies within +/-0.71 pp. Test: TOST on the paired differences, alpha 0.05.
  - This is the same margin as round-3 S4 on VLCS, so the two VLCS tests are judged by one rule.
- **Decision**, applied by `scripts/analyze_round5.py`, which reuses round 3's `paired()` and `decide()`:
  - EFFECT if H1 rejects;
  - EQUIVALENCE if H2 rejects;
  - both, if both reject;
  - otherwise INCONCLUSIVE.
- The decision is reported whatever it is.
- **Also reported, without a test:** whether the 95% CI excludes the paper's VLCS gain, +1.67 pp.

## 3. Fixed settings
Identical to `docs/PREREGISTRATION_ROUND4.md` section 3 except the dataset:

| Setting | Value |
|---|---|
| Code | `aristotelisballas/m2cl`, commit `0b1bce3026d9d56369037129e36f81e7645b80e6`, unmodified |
| Entry point | `python -m domainbed.scripts.train --dataset VLCS --algorithm M2CL` |
| Backbone | `--hparams '{"resnet18": true}'` |
| Loss weight | `--lparam 0.01` (M2-CL) and `--lparam 0.0` (term off) |
| Temperature | `--temp 1.0` |
| Learning rate | `--lr 0.001` (SGD, momentum 0.9, weight decay 5e-4, as coded) |
| Batch | code default: 32 per source domain, 96 in total |
| Steps, evaluation | code default for VLCS: 5,001 steps, evaluation every 300 steps |
| Hyperparameter seed | `--hparams_seed 0` (default hparams) |
| Seeds | `--trial_seed s --seed s` for s in {0, 1, 2}; pairs share both |
| Test domains | `--test_envs` 0, 1, 2, 3 = Caltech101, LabelMe, SUN09, VOC2007 (DomainBed sorts the folder names) |
| Data | DomainBed's VLCS: 10,729 images; Caltech101 1,415, LabelMe 2,656, SUN09 3,282, VOC2007 3,376; a folder named `VLCS` with exactly these four domain folders |
| Model selection | DomainBed's training-domain validation (`IIDAccuracySelectionMethod`), as in round 4 |
| Hardware | the institute's RTX 4000 Ada with the Intel Core Ultra 9, the same machine and environment as rounds 3 (VLCS) and 4 |

- **Data source:** the same verified copy that round 3's VLCS runs used (audit F15). The runner refuses any other counts.
- **Truncated JPEGs:** the authors' `datasets.py` loads truncated JPEGs (`ImageFile.LOAD_TRUNCATED_IMAGES = True`), and the runner lists them in `env/`.
- **Runs:** 2 settings x 4 test domains x 3 seeds = **24**. The runner is `scripts/run_round5_authors.py`. Fold and seed are the outer loops, so each pair finishes together.

## 4. Manipulation check (reported, never used to select or exclude runs)
- **Paired weight distance:** the same rule as round 4.
  - L2 distance between the final models of each pair, over the learned non-extraction parameters.
  - Reference: the median distance between seeds within the same fold in the lparam 0 arm.
  - **Active if the paired distance exceeds 10% of that reference in at least 10 of 12 pairs.**
- **Gradient share:** measured for the authors' code on PACS batches (0.109-0.127, audit F12). It is not re-measured on VLCS. The network, the loss and lparam are the same, and only the images differ.

## 5. Secondary results (descriptive, no tests)
- **Accuracy:** mean +/- SD per domain and the macro average for each arm.
- **Against the paper** (Table I, ResNet-18 VLCS):
  - the lparam 0.01 arm against the paper's M2-CL, 77.78;
  - the lparam 0 arm next to the paper's M2, 76.11, labelled as the same network without the term.
- **Against round 3:** round-3 S4 (our code, VLCS) and round 4 (authors' code, PACS), descriptively only, never pooled.

## 6. Failures and exclusions
As in round 4:
- **Interrupted runs:** an interrupted run is deleted and restarted from scratch with the same seeds.
- **No exclusions:** no completed run is dropped.
- **Allowed fixes:** environment-level fixes only, each recorded in `env/`. Any change to the authors' files is a deviation: it is written as an amendment before any run uses it, and reported.

## 7. Known limits, stated in advance
- **Detectable effect:** round 4's paired SD on PACS (1.72 pp) would give a minimum detectable effect of about 1.5 pp at 80% power. Round 3's VLCS paired SD (0.71 pp) would give about 0.63 pp. Which applies is not known before the runs.
- **Settings:** one batch size and one learning rate, with no hyperparameter search.
- **Timing:** this pre-registration is written after round 4's PACS result and round 3's VLCS result were known. Its design, rules and code copy round 4 exactly with the dataset changed, so those results could not influence any choice except the decision to run it. The reason for running it is the main remaining objection to the paper: the VLCS result rested on our reimplementation only.
