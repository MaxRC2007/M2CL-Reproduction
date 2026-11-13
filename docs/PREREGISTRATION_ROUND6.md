# Pre-registration, round 6: a confirmatory PACS test of the contrastive term at a fixed checkpoint, on new seeds

Written 13 Nov 2026 and pushed to the public repository before any round-6 run. The commit time on GitHub is the timestamp. Nothing below changes after the first run starts. Any necessary change is added as a dated amendment and reported in the paper as a deviation.

## 1. Why this round exists
- **Round 4** (pre-registered, `docs/PREREGISTRATION_ROUND4.md`): the authors' code on PACS, `lparam` 0.01 vs 0, seeds 0-2, with DomainBed's validation-selected checkpoint. Result: +0.81 pp (95% CI -0.28 to +1.91), **inconclusive**.
- **Exploratory, after rounds 4 and 5** (audit F34, `scripts/analyze_exploratory_r6basis.py`):
  - Validation selection picks a different step for the two runs of a pair, which adds variance.
  - At the final step (5000) the paired SD falls from 1.72 to 0.92 pp, and the effect is +0.74 pp (+0.15 to +1.32; sign-flip p 0.019).
  - That analysis was chosen after the data were seen, and two variants were computed, so it is not confirmatory.
- **Round 6 asks:** at the final checkpoint, does the contrastive term change PACS target accuracy in the authors' code?
  - It uses **new seeds** (3-7), never run with the authors' code, and a primary endpoint fixed now.
  - Rounds 4 and 5 stay reported exactly as they are. No round-4 or round-5 run is reused or re-analysed as confirmatory.

## 2. Primary endpoint
For each run: the test-domain accuracy (DomainBed's `env{t}_in_acc`) of the **last evaluation record, at step 5000**. This is the model the code saves as `model.pkl`. There is no model selection.

## 3. Hypotheses and decision (same rules as rounds 3-5)
- **H1 (effect):** the paired difference (lparam 0.01 minus lparam 0) is not zero. Test: exact sign-flip test on the 20 paired differences, two-sided, alpha 0.05.
- **H2 (equivalence):** the difference lies within +/-0.71 pp, the paper's own PACS gain. Test: TOST, alpha 0.05.
- **Decision**, applied by `scripts/analyze_round6.py` with round 3's `paired()` and `decide()`:
  - EFFECT if H1 rejects;
  - EQUIVALENCE if H2 rejects;
  - both, if both reject;
  - otherwise INCONCLUSIVE.
- **Also reported, without a test:** whether the 95% CI contains the paper's PACS gain (+0.71 pp) and whether it contains zero.
- **Reading, fixed in advance:**
  - EFFECT: the round-4 exploratory signal is confirmed. The paper reports a PACS gain at a fixed checkpoint, with its size and CI.
  - INCONCLUSIVE: the signal is not confirmed, and the paper says so.
  - EQUIVALENCE without an effect: the signal does not replicate. Any effect is within +/-0.71 pp.
  - The decision is reported whatever it is.

## 4. Fixed settings
Identical to `docs/PREREGISTRATION_ROUND4.md` section 3, except the seeds:

| Setting | Value |
|---|---|
| Code | `aristotelisballas/m2cl`, commit `0b1bce3026d9d56369037129e36f81e7645b80e6`, unmodified |
| Entry point | `python -m domainbed.scripts.train --dataset PACS --algorithm M2CL` |
| Backbone | `--hparams '{"resnet18": true}'` |
| Loss weight | `--lparam 0.01` (M2-CL) and `--lparam 0.0` (term off) |
| Temperature, learning rate | `--temp 1.0`, `--lr 0.001` (SGD, momentum 0.9, weight decay 5e-4, as coded) |
| Batch, steps, evaluation | code defaults: 32 per source domain; 5,001 steps; evaluation every 300 steps and at step 5000 |
| Hyperparameter seed | `--hparams_seed 0` |
| **Seeds** | `--trial_seed s --seed s` for **s in {3, 4, 5, 6, 7}**; pairs share both |
| Test domains | `--test_envs` 0-3 = art_painting, cartoon, photo, sketch |
| Data | DomainBed's PACS, 9,991 images (2,048 / 2,344 / 1,670 / 3,929), the copy round 4 verified |
| Hardware | the institute's RTX 4000 Ada, the same machine and environment as rounds 4 and 5 |

**Runs:** 2 settings x 4 test domains x 5 seeds = **40**. The runner is `scripts/run_round6_authors.py`. Fold and seed are the outer loops, so each pair finishes together.

## 5. Manipulation check (reported, never used to select or exclude runs)
- **Distance rule:** as in round 4, the L2 distance between the final models of each pair, over the learned non-extraction parameters. The reference is the median distance between seeds within the same fold in the lparam 0 arm.
- **Active** if the paired distance exceeds 10% of that reference in **at least 17 of 20 pairs**. This keeps round 4's proportion (10 of 12).

## 6. Secondary results (descriptive, no tests)
- **Validation-selected accuracy:** round 4's rule (`analyze_round4.select`), as a paired estimate.
- **Late checkpoints:** the mean over the last five checkpoints (steps 3900-5000), as a paired estimate.
- **Round 4 next to round 6:** round 4's final-step result beside round 6's, never pooled into a test.
- **Accuracy:** mean +/- SD per domain and the macro average for each arm.

## 7. Failures, stopping and exclusions
- **Interrupted runs:** an interrupted run is deleted and restarted from scratch with the same seeds, as in rounds 4-5.
- **No early look:** all 40 runs are analysed together, once. The analysis script refuses to decide with fewer than 20 complete pairs.
- **No exclusions:** no completed run is dropped.
- **No further seeds:** none are added after the result, whatever it is.
- **Allowed fixes:** environment-level fixes only, recorded in `env/`. Any change to the authors' files is a deviation, written as an amendment before any run uses it.

## 8. Power and known limits, stated in advance
- **Power,** by simulation with round 4's final-step paired SD (0.92 pp):
  - about 93% for a true effect of 0.74 pp;
  - about 64% for 0.5 pp;
  - minimum detectable effect at 80% power: 0.61 pp.
- **If the spread is larger:** with SD 1.2 pp, power at 0.74 pp is about 75%.
- **Winner's curse:** the 0.74 pp estimate was selected from exploratory analyses, so the true effect may be smaller.
- **Scope:** one batch size and one learning rate, PACS only, ResNet-18 only.
- **Timing:** written after rounds 3-5 and the exploratory analysis were known. The design copies round 4 except the seeds and the pre-declared endpoint, and both of those are motivated in section 1.
