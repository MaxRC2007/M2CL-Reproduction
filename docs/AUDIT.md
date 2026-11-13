# Code & results audit (3 Nov 2026)

Audit of the original training notebook (`Jupyter Files/M2CL_PACS_Final_Master_with_visualizations_run.ipynb`), the probe scripts (`DG Research/scripts/`) and their JSON outputs, against the draft *"Multiscale Contrastive Learning for Domain Generalization: A Reproduction Study…"*.
Every item lists the evidence, its effect on the draft, and how this repository fixes it.

**Verified as correct.** The corrected loss is the paper's sum-reduced Eq. (1)–(2) (`tests/test_torch_parts.py` checks the new loss against a verbatim copy of the notebook's loss, and the sum/mean ratio against C·M = 28). The 6-tap architecture matches the probe scripts key-for-key. The draft's Tables III, V, VI, VII, VIII and IX match the raw outputs to the printed precision; `scripts/make_tables.py` regenerates Tables V–VIII from `results/legacy_v1/paper_accuracy_runs.csv`.

| ID | Severity | Issue | Draft affected | Status |
|---|---|---|---|---|
| A0 | **Critical** | Leakage probes read the DCT copy of PACS, not the RGB images | Tables II, III, IX; abstract; Sec. V, IX, X, XII | guard added; re-run done 5 Nov 2026 (see "Re-run results") |
| A1 | **High** | Source-validation split leaks training images | all trained checkpoints; Sec. VI search | fixed; needs re-training |
| B1 | **High** | Probe point estimate and CI come from different probes | Table II (CIs in JSON) | fixed |
| B2 | **High** | Probe-init noise on random-init features is ≫ 0.8 pp | Table II, abstract, Sec. V-A | fixed; re-run done (5 probe seeds) |
| B3 | **High** | The "0.8 pp noise floor (5 probe seeds)" has no matching script/output | abstract, Sec. V-A | fixed (`leak_noise_floor` table) |
| B4 | Medium | The same ImageNet control has different values in different tables | Tables II vs IX | fixed (one control per run) |
| B5 | Medium | Linear class probes on ImageNet features ≈ 19–27 % (chance 14 %) | unreported supplementary breakdown | root cause found: A0 |
| C1 | Medium | Leakage measured only at GAP(layer4), not where the contrastive loss acts | Sec. IX–X conclusion | 4-tap re-run done but uninformative (E2); 6-tap needs re-training |
| C2 | Medium | No tap-matched, loss-off ablation | Sec. VI, VIII | config added (`configs/m2_6tap.yaml`) |
| D1 | Low | "Sign-permutation" p-values are the binomial sign test | Table VIII (Cartoon: 0.75, not 1.00) | both reported |
| D2 | Low | Table II SDs are population SDs (ddof=0) | Table II | sample SD (ddof=1) used; stated in caption |
| D3 | Low | Optimiser momentum 0.9, weight decay 5e-4, cosine schedule not reported | Table IV | add to Table IV |
| D4 | Low | `os.listdir` order is OS-dependent (Windows vs Colab) | split reproducibility across machines | listings sorted |
| E1 | **High** | Domain probes saturate on real images, so a backbone-level null is weak evidence | Sec. IX–XII conclusion | stated in v2 draft; needs a harder test |
| E2 | **High** | 4-tap M2 and M2-CL are almost the same network | Table III; C1 | measured; needs 6-tap M2 re-training |
| E3 | Medium | Table II rows are held-out folds, but the draft reads them as the domain's own statistics | Sec. V-B | corrected in v2 draft |
| E4 | Medium | Checkpoint accuracies differ slightly from Tables I and V | Tables I, V | measured; cause unknown |

---

### A0 — Leakage probes read DCT images instead of RGB PACS
`Downloads/PACS/<domain>/<class>/` holds `dct_*` files: frequency-domain images that look like noise. The RGB images are in `Downloads/PACS/pacs_data/pacs_data/`. Every JSON in `results/legacy_v1/` records `data_root = C:\Users\Madhusudan\Downloads\PACS`, and the probe scripts read `<root>/<domain>/<class>`, so Tables II, III and IX were measured on the DCT copy.

`scripts/audit_dct_data.py` probes the ImageNet-only ResNet-18 on both copies (backbone feature, 2 probe seeds, full dataset; `results/audit/dct_vs_rgb.json`):

| Held-out | Domain: draft JSON | Domain: DCT | Domain: RGB | Class: draft JSON | Class: DCT | Class: RGB |
|---|---|---|---|---|---|---|
| Art Painting | 90.56 | 91.15 | 99.60 | 22.85 | 22.39 | 68.12 |
| Cartoon | 88.63 | 88.87 | 98.65 | 24.91 | 22.80 | 51.07 |
| Photo | 88.43 | 87.55 | 99.12 | 25.39 | 24.52 | 95.69 |
| Sketch | 67.18 | 66.77 | 96.15 | 26.06 | 27.92 | 41.63 |

*Effect.* The DCT column reproduces the draft to within about 1 pp on domain accuracy and 2 pp on class accuracy; RGB does not. The draft's Tables II, III and IX therefore say nothing about real images; "Re-run results" below replaces them.

*Accuracy tables.* Target accuracy is a separate pipeline. The master notebook's output shows `PACS_ROOT: /content/pacs_data/pacs_data` (RGB), and `scripts/verify_checkpoints.py` gives `erm_art_painting.pt` 73.14 % on RGB (= Table I) vs 12.0 % on a 500-image DCT sample (`results/audit/checkpoint_accuracy.json`). All 22 checkpoints on disk are now checked: see E4.

*Fix.* `index_pacs` now rejects `dct_*` files unless the audit explicitly allows them, and the Colab notebook selects the RGB copy.

### Table I vs Section V-C
Table I reports 4-tap M2 and M2-CL accuracy on Sketch, but Section V-C says no M2-CL checkpoint had been trained with Sketch held out, and `m2_sketch.pt` / `m2cl_sketch.pt` (4-tap, this architecture) are not on disk. The only `m2cl_sketch.pt` found, in `Madhu/project/checkpoints`, has an older architecture. The author must establish where these two Table I values came from.

### Re-run results (5 Nov 2026, laptop CPU, RGB PACS, 5 probe seeds)
Source: `scripts/run_overnight.py`; tables in `results/tables/raw_backbone/` and `results/tables/raw_taps_v1/`.

| Held-out fold | Majority | Random init (5 seeds) | ImageNet | Draft Table II ImageNet (DCT) |
|---|---|---|---|---|
| Art Painting | 49.48 | 92.14 ± 0.43 | 99.61 | 91.2 |
| Cartoon | 51.37 | 86.07 ± 0.38 | 98.62 | 88.7 |
| Photo | 47.22 | 89.75 ± 0.49 | 99.13 | 88.3 |
| Sketch | 38.65 | 71.03 ± 0.31 | 96.11 | 66.5 |

- All 22 trained-vs-ImageNet deltas are negative (−0.09 to −0.71 pp). Two survive BH: ERM on the Photo fold (−0.71, p = 0.002; the draft had +1.40) and 6-tap M2-CL on the Photo fold, seed 2 (−0.65, p = 0.002). The deltas share one ImageNet control and test split, so the matching signs are not independent evidence.
- Probe noise (B2, B3): for ImageNet, the range across 5 probe seeds is 0.04 / 0.22 / 0.16 / 0.82 pp (Art / Cartoon / Photo / Sketch folds). For random init it reaches 10.50 pp, on the Sketch fold. So B2's concern was real, and averaging over probe seeds is needed.
- Class probes (B5) are healthy: ImageNet source-test accuracy is 82–89 %.

### E1 — Domain probes saturate on real images
Trained and ImageNet features reach 95.5–99.6 % on the 3-way domain task, so only 0.4–4.5 % of the test set can show a difference. "No detectable change" at this feature cannot be read as "training leaves domain information unchanged". Candidates for a harder test: tap-level probes on the 6-tap models, probes trained on far fewer images, or a measure that does not saturate.

### E2 — 4-tap M2 and M2-CL are almost the same network
Tap-level probes found 1 BH-significant result in 18 (Photo fold, eb2, +0.15 pp), with every |Δ| ≤ 0.15 pp. `scripts/compare_checkpoint_weights.py` (`results/audit/checkpoint_weight_distance.json`) shows why this test cannot say much:

| Fold | M2 from ImageNet | M2-CL from ImageNet | M2 vs M2-CL | 6-tap M2-CL seed 0 vs 1 |
|---|---|---|---|---|
| Art Painting | 1.154 | 1.163 | 0.034 | 0.905 |
| Cartoon | 1.170 | 1.175 | 0.035 | 0.882 |
| Photo | 1.273 | 1.275 | 0.041 | 0.961 |

(L2 distance over the 60 learned backbone parameter tensors, excluding BatchNorm running statistics.) M2 and M2-CL differ by about 3 % of the distance either moved in training. This fits the mean-reduction bug making the contrastive term negligible, probably with a shared starting seed; neither cause is proven. It is the first direct evidence for the draft's "negligible gradient signal" claim, but it also means the 4-tap comparison says nothing about what a working contrastive loss does. That needs 6-tap M2-CL against 6-tap M2 (C2).

### E3 — Table II rows are held-out folds
A probe's "Sketch" row is trained on Art, Cartoon and Photo; it contains no sketches. The draft explains the low Sketch value by how random filters respond to line drawings, which cannot be right. A likelier reading is that this fold excludes the most visually distinct domain, so its three domains are harder to tell apart: every fold containing sketches scores higher. The same care applies to "extending coverage to Sketch": what is new is the Sketch-held-out fold.

### E4 — Checkpoint accuracy vs the draft
All 22 checkpoints give the reported RGB accuracy to within 0.13 pp (largest: `m2cl_cartoon.pt`, 68.17 vs 68.30 in Table I) and only 3.8–24.2 % on a 500-image DCT sample. So those values were measured on real images. The cause of the small differences is unknown. Not checkable: ERM Sketch seed 0 in Table VI, the other ERM folds in that table, and the 4-tap M2 / M2-CL Sketch values, which have no checkpoints found so far. ERM Sketch seeds 1 and 2 were later found in `Downloads/60f0ba3dff02.zip` and `Downloads/2671d69ff6bb.zip`: their stored test accuracies (61.49 and 65.77) match Table VI, but these two have not been re-evaluated on the images. The 4-tap Sketch values trace to the saved output of `Downloads/M2CL_PACS_Corrected_Final.ipynb` (cells 14 and 16: M2 61.06, M2-CL 61.03, in the same table as the other Table I values). That output prints stored results ("already done"), so it records the values but is not a training log, and the matching `m2_sketch.pt` and `m2cl_sketch.pt` have not been found. `Madhu/project/checkpoints/m2cl_sketch.pt` shares the name but is an earlier prototype from March with a different architecture (768-dimensional classifier input against 1792 in `m2cl_photo.pt`), so it cannot verify 61.03. The Sketch alpha sweep in `m2cl_improvement_checkpoints-20260822T214728Z-1-001.zip` cannot stand in for them: its runs stop at epoch 15 and store no test accuracy.

### A1 — Source-validation split leaks training images
`split_domainwise()` permutes the three source domains with **one** `RandomState(seed)` stream, but `make_loaders()` rebuilds each domain's validation set with a **fresh** `RandomState(seed)`. Only the first source domain gets a matching split. For the other two, the validation images are drawn from a different permutation, and about 80 % of them are training images.

Reproduced for all 12 final runs (`tests/test_stats_and_splits.py::test_legacy_split_leaks_val_into_train`):

```
target=sketch seed=0  -> art_painting 0%  cartoon 80%  photo 80% of "val" images are train images
target=photo  seed=1  -> art_painting 0%  cartoon 82%  sketch 80%
```

*Effect.* Target-domain test accuracy is still measured on unseen images; there is **no target leakage**. But best-checkpoint selection and every "source-domain validation accuracy" in the Sec. VI search mostly measure training fit, so the α/τ/tap choices were made on a contaminated signal. Its likely effect is a bias toward later epochs; its size on target accuracy is unknown until re-run.

*Fix.* `src/m2cl/splits.py::split_train_val` computes the split once, and the validation loaders reuse it. Re-running Tables V/VI with the fixed split is part of the Colab notebook.

### B1 — Point estimate and CI from different probes
In `probe_random_init_multiseed.py`, `linear_acc` comes from an **unseeded** probe, while `bootstrap_ci()` trains a **second**, seeded probe. In `random_init_multiseed.json`, 14 of the 20 random-init `backbone_final` accuracies fall outside their own 95 % CI (e.g. Sketch seed 3: 37.6 % vs CI [46.0, 50.6]).

*Fix.* `src/m2cl/probe.py` is fully seeded. The paired bootstrap resamples the per-sample correctness of the very probes that produce the reported accuracy, so Δ always equals acc_A − acc_B.

### B2 — Probe noise on random-init features
`layer4` and `backbone_final` in those scripts are the **same 512-d vector** (fc = Identity; `layer4` is GAP-ed by the hook). The only difference between the two probes is their unseeded initialisation, yet their accuracies differ by up to:

```
photo 5.6 pp | art 6.0 pp | cartoon 6.9 pp | sketch 13.4 pp   (random-init, raw features)
```

So the random-init column of Table II (SD 1.1–3.8 pp) mixes weight-seed variance with probe-init variance, and the Sketch "architectural bias" (+6.0 pp) is within probe noise.

*Fix.* Each condition is probed with 5 probe seeds and averaged before any comparison. `--standardize` (z-scoring fitted on the probe-train split) is a robustness check; random-init raw features have very uneven scales.

### B3 / B4 — Noise floor and the ImageNet control
No script repeats the ImageNet probe over five probe seeds. The "deterministic" ImageNet control has different values in different outputs:

| Held-out | random_init_multiseed | imagenet_control | table_viii (Table IX) |
|---|---|---|---|
| Art Painting | 91.19 | 90.56 | 91.19 |
| Cartoon | 88.67 | 88.63 | 88.41 |
| Sketch | 66.47 | 67.18 | 67.29 |

Table II uses the first column and Table IX the third. The largest spread (Sketch, 0.82 pp) is probably where "0.8 pp" came from.

*Fix.* The `leak_noise_floor` table reports the probe-seed SD and range for every condition and layer. All comparisons in one run use one ImageNet control.

### B5 — Class probes near chance
`imagenet_control_results.json` → `class_probes.*.linear_acc` = 0.19–0.27 for ImageNet features. That cannot be right: a linear probe on ImageNet ResNet-18 features classifies PACS objects far above 70 %. The draft omits these numbers (good), but the same pipeline produced the domain probes.

Root cause: the DCT images (A0). On RGB the same probe reaches 45–96 % on the held-out domain (5-seed re-run). The class probe stays in the `leak_class_sanity` table as a standing check.

### C1 — Probing where the loss acts
The contrastive term acts on the **extraction-block outputs**, but every leakage test used GAP(layer4). A null result there does not show that the loss leaves domain information unchanged, because it might act only on the taps.

The new `cl_effect_at_taps` families compare M2-CL with M2 (same taps, loss on vs off) on each `eb*` and on `eb_concat`. The 4-tap version has now been run, but it cannot answer the question (E2).

### C2 — Tap-matched ablation
`configs/m2_6tap.yaml` is identical to the locked M2-CL config except α = 0. Together with ERM it separates "more taps" from "loss fixed" (Sec. XI, first limitation), and it is the comparison E2 now requires.

**6-tap M2 run log (5 Nov 2026, Colab T4).** All 12 runs finished on torch 2.11.0+cu130 with the fixed split: Art 75.52 ± 1.85, Cartoon 69.40 ± 2.38, Photo 95.51 ± 0.36, Sketch 62.71 ± 0.82, macro-average 75.79. `m2_6tap_sketch_seed1` (62.66, best epoch 23) is a resumed run: it was interrupted after epoch 15 and continued from its checkpoint in a different Colab session with the hardened engine. The older checkpoint did not store the data-loader shuffle state, so its shuffle order restarted at epoch 16; report it as resumed. `m2_6tap_sketch_seed2` ran entirely in the second session. These values cannot yet be compared directly with the draft's 6-tap M2-CL and ERM runs, which selected checkpoints using the leaky validation split (A1). The fair comparison needs 6-tap M2-CL, and ideally ERM, re-run with the fixed split.

### D1–D4
- **D1:** for n = 3 seeds both tests are now reported. For Cartoon (−2.35, +5.50, +1.96) the sign-flip test gives 0.75 and the sign test gives 1.00.
- **D2:** sample SDs for draft Table II would be 1.3 / 2.4 / 1.8 / 4.2.
- **D3:** report the full optimiser settings.
- **D4:** directory listings are now sorted, so splits are identical on Windows and Colab.

## F. Review against the original paper and public code (5 Nov 2026)

Sources: arXiv 2308.14418 and the authors' public repository `github.com/aristotelisballas/m2cl` (commit 0b1bce3). The code was read, not run. The exact command behind the published tables is not public, and the repository's defaults differ from the paper (ResNet-50, batch 32, lr 5e-5, 5,001 steps), so the code shows one implementation, not necessarily the published runs.

### F1 — The contrastive term is nearly inert at the locked settings
Measured with `scripts/audit_grad_ratio.py` (two batches of 128 from the Sketch-held-out fold, train mode, CPU), the gradient norm of alpha x contrastive over cross-entropy:

| Network | tau | Backbone, alpha 0.003 | Backbone, alpha 0.01 | Backbone, alpha 0.03 | Extraction blocks, alpha 0.003 |
|---|---|---|---|---|---|
| ImageNet start | 1.0 | 0.004 | 0.013 | 0.04 | 0.002–0.003 |
| Trained 6-tap M2-CL | 1.0 | 0.004 | 0.015 | 0.04 | 0.016–0.017 |
| Trained 6-tap M2-CL | 0.5 | 0.009 | 0.03 | 0.09 | 0.03 |
| Trained 6-tap M2-CL | 0.07 | 0.066 | 0.22 | 0.66 | 0.24 |

The summed loss is about 171 at ImageNet weights and about 171 after training, so training barely lowers it. `scripts/audit_alpha_sweep.py` agrees: in the 15-epoch, same-seed Sketch screening runs, all sum-reduced, backbone distance from alpha 0 is 0.011 for alpha up to 0.003, 0.013 at 0.01 and 0.022 at 0.03, against about 1.2 moved from ImageNet in full training. The 6-tap tau runs (0.5, 1.0, 1.5) are 0.011 apart and tie exactly. So fixing the reduction was correct but did not by itself make the loss matter. Limits: few batches, a norm ratio ignores direction and consistency over steps, and the sweep distances use early best checkpoints. The authors' model has not been measured (F7).

### F2 — Learning rate history
- Mar 19–29: lr 1e-3 runs read `dct2_images`; ERM reached 29.1 % on Art at epoch 30.
- Mar 30: `M2CL_PACS_Fixed_v2` moved to RGB with lr 1e-4 / 1e-3 per parameter group. `M2CL_PACS_Final_Corrected_(2)` then used uniform 5e-4 on RGB, labelled "DomainBed default for PACS". The registry gives 5e-5 for PACS, and stock DomainBed uses Adam, so that label is wrong.
- Apr 7–8: later notebooks show lr 1e-3 (and in one, tau 0.07) in code, but every run was skipped as already done. These values produced no results, though they sit beside Table I.
- No uniform 1e-3 run on RGB has surviving output. Direct logs at 5e-4 exist for only three Table I values (M2 Art, M2-CL Art, ERM Cartoon), so "all experiments use 5e-4" is inferred for the others.
- Cosine annealing is our choice: the paper states no schedule and the authors' M2-CL optimiser has none. Mean lr is 2.5e-4, so cumulative lr is a quarter of constant 1e-3, not a half.

### F3 — Provenance of the two 4-tap Sketch values
April sessions restored only `m2_sketch.pt` and `m2cl_sketch.pt` from Drive, with `PACS_ROOT` at `dct2_images` (all runs skipped). The May session restored the other 10 with `PACS_ROOT` at `pacs_data`. This may reflect different Drive contents or accounts. So the training session of 61.06 and 61.03, including its data folder and lr, is undocumented. DCT is unlikely, because DCT runs reached only about 29 % test accuracy and 47–48 % train accuracy on this fold at epoch 15, but it is not excluded. "Training used the RGB images throughout" should be limited to the verified checkpoints. Look for `m2cl_checkpoints/` and `m2cl_history.json` on each account's Drive.

### F4 — Undeclared differences from the authors' code

| Item | Authors' code | This project |
|---|---|---|
| Taps | 13 locations, 21 pipelines; some taps are raw conv outputs before BN | 6 stage or sub-block outputs, 16 pipelines |
| Contrastive terms | One per pipeline (21), on the first MLP layer before activation | One per block (6), on concatenated MLP outputs |
| Total weight, alpha x terms | 0.01 x 21 = 0.21 | 0.003 x 6 = 0.018 |
| Classifier input | 149,504; backbone GAP not included | 2,560; backbone GAP included |
| Pipeline | conv, dropout 0.3, max pool, ReLU; MLP hidden 1024, GELU | conv, BN, ReLU, dropout 0.5, adaptive max pool; 128 with ReLU |
| Batch | `batch_size` per source domain, sampled with replacement, domain-balanced | 128 pooled, shuffled, not domain-balanced |
| Length | Steps (default 5,001) | 30 epochs, about 1,140–1,560 steps |
| Schedule | None for M2-CL | Cosine |
| Test preprocessing | Resize to 224 x 224 | Resize 256, centre crop 224 |
| Colour jitter hue | 0.3 | 0.1 |
| tau | Divides outside the exponential, so it changes only a constant in the log | Inside the exponential, as in the paper |

Momentum 0.9, weight decay 5e-4 and the 80/20 holdout match.

### F5 — Claims in `paper/M2CL_v2_rgb.tex` to correct
- l.103–107: says the authors report the contrastive objective as beneficial on every domain and calls this a discrepancy in their result. Their ablation has M2 77.90 above M2-CL 77.07 on Sketch, and their text excepts Sketch.
- l.237: says the three listed deviations were imposed by compute. The learning rate was not, and the list is incomplete (F4).
- l.192–200: attributes GAP concatenation to the original design and calls the pipeline count unrecoverable. The public code gives 21 and excludes GAP from the classifier input. The repository is never cited.
- l.236: "same batch size" and epoch budget hold against the paper's text, not necessarily the code.
- l.328: RGB overclaim (F3).
- l.479–484: the bug "plausibly explains" near-identical M2 and M2-CL, but F1 shows the sum-reduced term is also nearly inert.
- l.559–564: tau ties break toward the published default, but alpha, also within noise (93.20 vs 93.05), does not.
- l.682: says Sketch has the largest seed variability in both tables, while Cartoon is largest for M2-CL (2.78, l.645).
- l.732: Cartoon's 1.00 is the sign test; the sign-flip test gives 0.75. The column is labelled as one test.
- l.781 and l.282: "below the paper on every domain" holds for M2-CL only. ERM Photo is 5.46 pp above the paper.
- l.785: "reduced training-image counts" is unsupported; the 80/20 split matches.
- l.954: partly out of date (two ERM Sketch checkpoints found).
- l.167–171: check SelfReg directly; it is described as using only positive pairs, which does not fit the cross-domain-negative argument attributed to it.

### F6 — Learning-rate and pilot checks still to run
Use a separate `--out_dir`, because run names do not include the learning rate and a changed config retrains a finished run. Do not add `RunConfig` fields while runs are in progress: the saved config comparison would stop matching.

### F7 — Open
The authors' model and loss have not been measured; running downloaded code was blocked by the session's permission rules. Either approve that run or rebuild their 21-pipeline design in this code base.

### F8 — Paired 6-tap M2 and M2-CL, first 8 of 12 runs (5 Nov 2026)
From the Colab training log of `configs/m2cl_6tap.yaml` (fixed split) against the finished `m2_6tap_*` results, same fold and seed. M2-CL Photo seed 2 and Sketch were not finished, and the M2-CL checkpoints have not yet been inspected, so this compares logged values only.

| Fold | Seed | M2 test | M2-CL test | Difference | Best epoch M2 / M2-CL | Largest per-epoch val gap | Final CE M2 / M2-CL |
|---|---|---|---|---|---|---|---|
| Art | 0 | 73.39 | 73.24 | -0.15 | 27 / 29 | 0.22 | 0.1916 / 0.1914 |
| Art | 1 | 76.61 | 76.46 | -0.15 | 28 / 28 | 0.14 | 0.1924 / 0.1922 |
| Art | 2 | 76.56 | 75.44 | -1.12 | 26 / 30 | 0.18 | 0.1822 / 0.1820 |
| Cartoon | 0 | 68.47 | 68.81 | +0.34 | 28 / 27 | 0.12 | 0.1783 / 0.1782 |
| Cartoon | 1 | 72.10 | 72.99 | +0.89 | 30 / 28 | 0.16 | 0.1739 / 0.1737 |
| Cartoon | 2 | 67.62 | 67.62 | 0.00 | 25 / 25 | 0.21 | 0.1792 / 0.1790 |
| Photo | 0 | 95.33 | 95.15 | -0.18 | 30 / 27 | 0.25 | 0.2213 / 0.2210 |
| Photo | 1 | 95.93 | 95.93 | 0.00 | 27 / 27 | 0.24 | 0.2150 / 0.2148 |

Mean difference -0.05 pp: 2 higher, 4 lower, 2 ties. M2-CL CE is computed as logged loss minus 0.003 x logged cl. The three pairs that selected the same epoch differ by at most 0.15 pp (3 images of 2,048); the two largest differences (-1.12 and +0.89) both selected different epochs, which points to checkpoint selection, though per-epoch test accuracy was not logged. M2 test accuracy varies by 3.22 (Art) and 4.48 (Cartoon) across seeds. The contrastive loss stays at about 170 from first epoch to last.

So at alpha 0.003 and tau 1.0, M2-CL trains as M2 on these folds. This confirms F1 in training behaviour, but it is not a test of the contrastive method: the term is too weak to change the run, and the authors' design differs (F4). The legacy lead of M2-CL over M2 (76.19 vs 75.79) should not be reported as a contrastive effect.

### F9 — All 12 paired 6-tap M2 and M2-CL runs, with weight distances (6 Nov 2026)
Source: `Downloads/m2_v2.zip` (previous Colab account: 12 M2 runs, the first 9 M2-CL runs, and the resume checkpoints of M2-CL Sketch seed 0) and `Downloads/m2cl_v2.zip` (new account: all 12 M2-CL runs). The 9 M2-CL `best.pt` files present in both zips are byte-identical. All runs: fixed split, lr 5e-4, torch 2.11.0+cu130, CUDA. M2-CL Sketch seed 0 resumed at epoch 13 on the new account; its checkpoint stored the data-loader state, so the shuffle order continued. M2 Sketch seed 1 is the resumed run whose shuffle order restarted at epoch 16 (see the M2 run log above). Measured with best.pt loaded as weights only; backbone = the 60 learned ResNet-18 tensors, excluding BatchNorm running statistics; "rest" = extraction blocks and classifier.

| Fold | Seed | M2 test | M2-CL test | Difference | Best epoch | Max val gap | Backbone distance, pair | Rest, pair | M2 backbone from ImageNet |
|---|---|---|---|---|---|---|---|---|---|
| Art | 0 | 73.39 | 73.24 | -0.15 | 27 / 29 | 0.21 | 0.032 | 0.020 | 1.167 |
| Art | 1 | 76.61 | 76.46 | -0.15 | 28 / 28 | 0.14 | 0.031 | 0.020 | 1.153 |
| Art | 2 | 76.56 | 75.44 | -1.12 | 26 / 30 | 0.18 | 0.033 | 0.021 | 1.150 |
| Cartoon | 0 | 68.47 | 68.81 | +0.34 | 28 / 27 | 0.12 | 0.033 | 0.019 | 1.156 |
| Cartoon | 1 | 72.10 | 72.99 | +0.90 | 30 / 28 | 0.17 | 0.030 | 0.019 | 1.145 |
| Cartoon | 2 | 67.62 | 67.62 | 0.00 | 25 / 25 | 0.21 | 0.028 | 0.018 | 1.135 |
| Photo | 0 | 95.33 | 95.15 | -0.18 | 30 / 27 | 0.24 | 0.032 | 0.019 | 1.254 |
| Photo | 1 | 95.93 | 95.93 | 0.00 | 27 / 27 | 0.23 | 0.033 | 0.020 | 1.249 |
| Photo | 2 | 95.27 | 95.27 | 0.00 | 28 / 28 | 0.24 | 0.030 | 0.020 | 1.238 |
| Sketch | 0 | 61.92 | 61.92 | 0.00 | 30 / 30 | 0.16 | 0.018 | 0.014 | 1.006 |
| Sketch | 1 | 62.66 | 63.55 | +0.89 | 23 / 29 | 0.48 | 0.074 | 0.056 | 0.995 |
| Sketch | 2 | 63.55 | 63.50 | -0.05 | 23 / 23 | 0.23 | 0.016 | 0.013 | 0.989 |

Fold means, M2 vs M2-CL: Art 75.52 ± 1.85 vs 75.05 ± 1.65; Cartoon 69.40 ± 2.38 vs 69.81 ± 2.82; Photo 95.51 ± 0.36 vs 95.45 ± 0.42; Sketch 62.71 ± 0.82 vs 62.99 ± 0.93. Macro-average 75.79 vs 75.83. Mean paired difference +0.04 pp (3 higher, 5 lower, 4 ties); exact sign-flip p over 12 pairs = 0.88.

Same-seed backbone distance is 0.016–0.033 (0.074 for Sketch seed 1), against 0.73–0.95 between seeds of the same method and 0.99–1.25 moved from ImageNet: about 3 % of the training movement, as for the 4-tap pair (E2). Between-seed distances are nearly identical for M2 and M2-CL (for example Art 0.906 vs 0.908). The six pairs that selected the same epoch differ by at most 0.15 pp. The largest pair distance belongs to Sketch seed 1, where the M2 run's shuffle order restarted on resume, so that one interruption moved the network more than the contrastive term did elsewhere.

Conclusion: at alpha 0.003 and tau 1.0, sum-reduced, M2-CL and M2 are the same network to within about 3 % of training movement on all four folds. These runs cannot show whether the contrastive objective helps or harms domain generalisation; they show only that this configuration does not apply it with any measurable force (F1). The tap-level leakage probes would therefore compare two near-copies of one network.

### F10 — Why the loss barely moves at tau 1.0, and the v3 draft (6 Nov 2026)
`scripts/audit_loss_bounds.py` evaluates the project's loss on batches of 128 drawn with PACS class proportions (dog 1,729, elephant 1,654, giraffe 1,566, guitar 1,113, horse 1,540, house 943, person 1,446) and 6 representation sets. With no class structure (random unit vectors) the summed loss is about 169.6 at every tau. With every class collapsed to one point it falls only to 132.9 at tau 1.0 (simplex classes; 137.3 with orthogonal classes), 107.9 at tau 0.5 and 89.1 at tau 0.07. Two properties of Eq. (1) cause this: the denominator includes every class's positive pairs, so p(c) cannot exceed class c's share of same-class pairs, and at tau 1 all pair weights lie within a factor e^2 of each other. Across all epochs of the 12 corrected M2-CL runs the logged loss stayed between 168.39 and 171.37, at the no-structure level. Final training cross-entropy (loss minus alpha x cl) matches the paired M2 run to within 0.0003 in 11 pairs and 0.0046 in Sketch seed 1, the pair whose M2 run resumed with a restarted data order.

Sources checked for the draft (arXiv 2308.14418 HTML): the paper's Table I has ResNet-18 and ResNet-50 blocks, and the draft's paper values are the ResNet-18 block. The paper states SGD, lr 0.001, batch 128, 30 epochs, alpha 0.01, tau 1.0 and 3 runs, but not momentum, weight decay, schedule or model selection. It says the loss helps on every domain except Sketch with ResNet-18, and that accuracy falls back to M2 as alpha approaches zero (its Table IV writes 0.01 as "10e-3"). SelfReg uses only positive pairs; the InfoNCE argument about domain-irrelevant common features is DomCLP's (arXiv 2412.09074, unsupervised DG). Not verified online: the TAI volume, issue and pages, the DomCLP AAAI pages, and the SelfReg ICCV pages.

`paper/M2CL_v3.tex` rewrites the draft around these results; `paper/M2CL_v2_rgb.tex` is unchanged and `paper/M2CL_v2_rgb_backup_2026-11-06.tex` is a copy. v3 fixes every F5 item, replaces the outdated comparison figure with tables, and leaves three `\pending` experiments (dose-response, learning rate, corrected-split ERM) and two `\authorcheck`s (61.06/61.03 provenance, repository URL). It has passed a structural check only: no LaTeX compiler is installed on the laptop.

### F11 — Equivalence test, figure, and the round-2 notebook (7 Nov 2026)
`scripts/make_fig_pairs.py` (inputs: `Downloads/m2_v2.zip`, `Downloads/m2cl_v2.zip`) gives, over the 12 seed-matched pairs: mean difference +0.040 pp, SD 0.525; 90 % CI -0.23 to +0.31, 95 % CI -0.29 to +0.37; TOST equivalence within +/-0.4 pp p = 0.018, within +/-0.5 pp p = 0.006 (paired t, 11 df). The largest deviation of one M2 seed from its fold mean is 2.13 (Art), 2.70 (Cartoon), 0.42 (Photo) and 0.84 (Sketch) pp. It writes `paper/fig_pairs.pdf`, now Fig. 1 of `paper/M2CL_v3.tex`. The palette validator could not run (no Node.js on the laptop); the two series differ by marker shape as well as colour.

`notebooks/Round2_Experiments_Colab.ipynb` runs the three `\pending` experiments into separate Drive folders: `m2cl_v2/dose/alpha_<a>/` (M2-CL, Art, seed 0, alpha 0.01, 0.1, 0.3, 1.0), `m2cl_v2/lr/lr_0.001/` (M2, seed 0, four folds) and `m2cl_v2/erm/` (12 runs). Tested locally: every cell compiles; the exact shell commands, parsed by the real `train_lodo.py`, build the intended 20 runs, the dose runs differ from `m2cl_6tap.yaml` only in alpha and the lr runs from `m2_6tap.yaml` only in lr; the restore cell copies only these three folders, skips `.tmp` files and keeps newer progress; `scripts/smoke_test.py` passed (ERM and M2-CL one-epoch training, resume skip, 17 unit tests). Not run on Colab yet.

Added to the v3 draft: GradNorm, SupCon and Pineau et al. citations (from memory, not checked online).

### F12 — Gradient share in the authors' own code (7 Nov 2026, run with the user's approval)
`scripts/audit_authors_grad_share.py` imports the public repository's `domainbed/mymodels/m2cl_18.py` and `domainbed/lib/myloss.py` by path (commit 0b1bce3; both import only torch). Their ImageNet download (`resnet18-5c106cde.pth`) is redirected to the local torchvision cache file `resnet18-f37072fd.pth`, assumed to hold the same ImageNet weights. It replicates `M2CL.update` exactly, including its same-class pair construction, with the defaults hard-coded in `algorithms.py` (alpha `lparam` 0.01, tau `temp` 1.0). It uses two batches of 96 images: 32 from each source domain, with Sketch held out, sampled with replacement, with DomainBed's PACS training augmentation, in train mode, at ImageNet start. Our 6-tap model is measured on the same batches.

| Same batches | Backbone | Extraction blocks |
|---|---|---|
| Authors, alpha 0.01, tau 1 (21 pipelines) | 0.1089, 0.1272 | 0.0710, 0.0585 |
| Ours, alpha 0.003, tau 1 (6 blocks) | 0.0045, 0.0055 | 0.0025, 0.0028 |
| Ours, alpha 0.01 | 0.0149, 0.0183 | -- |

The authors' contrastive term is 6.01 and 6.18, that is alpha x 21 pipelines x about 29 per pipeline, against cross-entropy 1.95. Their contrastive gradient is about 24 times our locked share and 7 times ours at the same alpha. Matching it in our model needs alpha of about 0.07, so the round-2 dose point alpha 0.1 slightly exceeds it. Code details found while reading: the loss uses `fc1` pre-activation outputs of each pipeline's MLP, which is applied per channel over spatial positions (for example 16 channels x 1,024 units in the first blocks). The denominator includes self-pairs and both pair orders. The same-class pair list skips any class whose label is at least the number of classes present in the batch. `ext_add_block0` and `ext_add_block4` are computed but used neither by the classifier nor by the loss. Limits: ImageNet start only (no trained checkpoint of theirs), 2 batches, the repo default of 32 per domain (not 128), and norms only. The laptop had 0.6 GB of free RAM at the time.

Draft `paper/M2CL_v3.tex` updated: new Table "tab:authors" and a subsection, plus the abstract, introduction, discussion, limitations and conclusion.

### F13 — Round-2 results: dose-response, learning rate, corrected-split ERM (7 Nov 2026)
Sources: `Downloads/dose.zip`, `lr.zip` and `erm.zip` (from `notebooks/Round2_Experiments_Colab.ipynb`). The zips are intact, all runs are complete, and the configs are as intended: the dose runs differ from `m2cl_6tap.yaml` only in alpha, the lr runs from `m2_6tap.yaml` only in lr, and ERM uses `erm.yaml`. All ran on torch 2.11.0+cu130. ERM Photo seed 1 resumed at epoch 5, using the engine that stores the data order. Analysis: `scripts/audit_round2.py <Downloads> <PACS root> --grad`.

**Dose-response** (Art held out, seed 0, same seed, split and initialization as M2 seed 0). Share is alpha x |grad CL| / |grad CE| on the backbone at the selected checkpoint, over 2 batches of 128. Dist. is the backbone L2 distance from M2 seed 0; M2 Art seeds lie 0.88-0.91 apart.

| alpha | share | cl first -> last | dist. | best val | test | train acc last | CE last |
|---|---|---|---|---|---|---|---|
| 0 (M2) | 0 | - | 0 | 94.64 | 73.39 | 94.2 | 0.1916 |
| 0.003 | 0.003 | 170.4 -> 170.2 | 0.032 | 94.68 | 73.24 | 94.1 | 0.1914 |
| 0.01 | 0.010-0.012 | 170.4 -> 170.1 | 0.038 | 94.75 | 73.24 | 94.1 | 0.1908 |
| 0.1 | 0.19-0.20 | 170.4 -> 167.9 | 0.314 | 94.81 | 74.22 | 94.0 | 0.1856 |
| 0.3 | 0.86-0.89 | 170.3 -> 165.4 | 0.748 | 95.06 | 74.12 | 94.5 | 0.1671 |
| 1.0 | 4.1-10.8 | 170.3 -> 160.8 | 2.170 | 95.35 | 73.05 | 96.1 | 0.1208 |

From alpha 0.1, above the authors' share of 11-13 %, the term is active: the network moves up to 2.4 times the seed-to-seed distance, source validation rises monotonically, and training CE falls. Target accuracy stays within 0.9 pp of M2, while M2's three Art seeds span 3.2 pp. Single fold and seed: effects below about 2 pp cannot be detected.

**Learning rate 1e-3** (M2, seed 0, cosine kept): Art 73.39 -> 75.88 (+2.49), Cartoon 68.47 -> 70.99 (+2.52), Photo 95.33 -> 95.75 (+0.42), Sketch 61.92 -> 66.91 (+4.99), macro 74.78 -> 77.38 (+2.60). Best validation is higher on all four folds (for example Art 94.64 -> 96.23), so validation-based selection would pick 1e-3. Final training CE roughly halves (Art 0.1916 -> 0.0950), and the backbone moves further from ImageNet (Art 1.167 -> 1.602). The 5e-4 runs are under-trained.

**ERM, corrected split** (12 runs): Art 73.60 +/- 1.62 (71.88, 75.10, 73.83), Cartoon 67.43 +/- 3.08 (70.99, 65.53, 65.78), Photo 95.01 +/- 0.38 (94.79, 95.45, 94.79), Sketch 64.27 +/- 3.35 (66.51, 60.42, 65.89), macro 75.08. Paired by seed: M2 - ERM +0.71 (95 % CI -1.12 to +2.53, sign-flip p 0.426, 9 of 12 higher), M2-CL - ERM +0.75 (-1.18 to +2.67, p 0.431). Per fold, M2 - ERM is Art +1.92, Cartoon +1.96, Photo +0.50, Sketch -1.56. Coincidence noted, not interpreted: ERM Sketch seed 0 = 66.51, the same value as the initial single-seed ERM of Table I.

`paper/M2CL_v3.tex` updated: the three `\pending` markers are replaced by Table "tab:dose" (Section "Dose-Response"), Table "tab:lr" (Section "Learning Rate") and an ERM column in Table "tab:accuracy". The abstract (221 words), introduction, discussion, limitations, run log and conclusion are revised. Remaining author checks: 61.06/61.03 provenance and the repository URL. Next decisive experiment: M2 vs M2-CL at lr 1e-3 with alpha about 0.07-0.1, 3+ seeds, 4 folds, ideally also on VLCS.

### F14 — Logical audit of the draft, VLCS support, and the pre-registered round 3 (7 Nov 2026)
**Fallacies found in `paper/M2CL_v3.tex` and fixed:**
- *Post-hoc equivalence margin* (+/-0.4 was chosen after seeing the data). The paper now uses +/-0.71 pp, the original paper's own M2-CL gain (83.54 - 82.83), as the margin: TOST p = 0.0005; +/-0.4 is mentioned but not relied on.
- *Absence of evidence read as evidence of absence* (M2 vs ERM). It is now stated explicitly as inconclusive: neither interval excludes a 2 pp advantage.
- *Mixing like with unlike* (dose share at a trained checkpoint compared with the authors' share at ImageNet start). The draft now compares both at ImageNet start: ours at alpha 0.1 is 0.15-0.18 vs theirs 0.11-0.13.
- *Causal overreach from n = 1* (dose: "did not carry over"). Now "not accompanied by a detectable change".
- *Incommensurable comparison* (the lr effect "costs 3x the paper's gain"). Now "larger than, though measured in different setups".
- *Equivocation* ("M2-CL" meaning ours or theirs). The terms are defined, and every claim is scoped to our implementation.
- *Appeal to authority / reference values as ground truth.* There is a new discussion paragraph: the paper's ResNet-18 ERM Photo of 88.91 vs our 95.01 and its own ResNet-50 97.30.
- *Use of test information in design* (Sketch chosen as the development fold from test results). This is acknowledged.
- *Regression to the mean* (the initial reversal against a favourable ERM seed), *confounding* (split vs code version), *multiple comparisons* (Bonferroni across the 4 accuracy tests changes nothing; equivalence p = 0.002 after correction), *survivorship / selective reporting* ("every run is reported; none excluded").
- "Limitations" is replaced by "Threats to Validity" (construct, internal, statistical-conclusion and external validity). A new Appendix A table justifies every design decision and marks the unjustified ones (alpha 0.003, lr 5e-4, cosine) as such.

**VLCS support** (`src/m2cl/splits.py`, `data.py`, `scripts/train_lodo.py`): the dataset is detected from its domain folders, and PACS is checked first. PACS is proved unchanged: old (from the Colab zip) and new code give identical indexes, all 20 fold x seed splits (seeds 0-4) and domain seeds on the real 9,991-image PACS. `tests/test_vlcs.py` adds 3 tests (20 of 20 pass). A real one-epoch VLCS M2-CL run completes through the CLI. `colab_upload/M2CL_code.zip` was rebuilt by the new `scripts/build_colab_zip.py`; the previous zip is kept as `M2CL_code_2026-11-05.zip`.

**Round 3, pre-registered** (`docs/PREREGISTRATION.md`): M2 vs M2-CL at lr 1e-3, alpha 0.07, which matches the authors' gradient share (0.01 x 0.109/0.0149 = 0.073; 0.01 x 0.127/0.0183 = 0.069). It uses seeds 0-2 on 4 PACS folds, H1 sign-flip and H2 TOST at +/-0.71, a fixed decision rule, and manipulation checks (share >= 0.05; distance > 0.033 in at least 10 of 12 pairs). The secondary analyses are ERM at 1e-3, the lr effect over 12 pairs, and VLCS, all Holm-corrected; seeds 3-4 are reported separately. Configs: `configs/round3_{m2,m2cl,erm}.yaml`. Notebook: `notebooks/Round3_Decisive_Colab.ipynb` (parts A-D, 84 runs). Analysis, fixed in advance: `scripts/analyze_round3.py`. Tested: every cell compiles; the commands build exactly 20/12/36/16 runs; the new M2 config equals the round-2 seed-0 run's config; M2-CL differs from the locked config only in alpha and lr; the status cell tracks exactly the 84 runs; the analysis reaches the right decision on synthetic equivalence, effect, inconclusive and incomplete cases; a run with non-pre-registered settings stops the analysis; on the real `lr.zip` + `m2_v2.zip`, S2 reproduces +2.60 and the primary test reports INCOMPLETE. Not run on Colab yet.

New docs: `docs/CLAIMS.md` (every claim -> evidence -> how to reproduce), `docs/RUN_GUIDE.md` (steps, timestamping, citation checks, a draft email to the authors).

### F15 - VLCS source (6 Nov 2026)
DomainBed's Google Drive link for VLCS (id 1skwblH1_okBwxWxmRsp9_qi15hyPpxg8) failed on Colab with gdown `FileURLRetrievalError: Cannot retrieve the public link of the file`, which means Google's download limit for that public file was reached. The Part C cell of `notebooks/Round3_Decisive_Colab.ipynb` now tries three sources in order: a `VLCS.tar.gz` file or shortcut at the top of My Drive; DomainBed's link (with gdown upgraded); the Hugging Face mirror `PTeterwak/DomainBed_OOP`, `all/VLCS.tar.gz` (3,838,010,386 bytes; release of "Do Domain Generalization methods Generalize Beyond their Pre-training?"). Whatever the source, the cell stops unless the per-domain image counts equal DomainBed's exactly.

The mirror was checked by streaming the full archive on the laptop. Its layout is `VLCS/<domain>/<class>/<hash>.jpg`, and it holds 10,729 images: Caltech101 1,415 (bird 237, car 123, chair 118, dog 67, person 870), LabelMe 2,656 (80/1209/88/42/1237), SUN09 3,282 (20/932/1036/30/1264), VOC2007 3,376 (330/699/428/420/1499). This equals DomainBed's reported total and domain sizes. The pre-registration does not name a data source, so this is not a deviation, but the paper should state the source used. The cell was tested on synthetic archives (correct copy -> used; one image short -> rejected and falls through; no source -> stops). It has not run on Colab yet.

### F16 - Kaggle as a second platform (7 Nov 2026)
`notebooks/Round3_Kaggle.ipynb` runs part C on Kaggle (T4, 30 GPU hours a week, 12-hour sessions), while A, B and D stay on Colab (D had already started there; the first draft of the amendment put D on Kaggle and was corrected the same day, before any Kaggle run): the allocation is fixed in `docs/PREREGISTRATION.md`, Amendment 1, so no paired comparison crosses platforms. It differs from the Colab notebook only in plumbing:
- inputs found anywhere under /kaggle/input, whether Kaggle unpacked the zips or not;
- progress carried between sessions by adding the previous version's output as an input (or a Colab `m2cl_v2` zip);
- one subprocess per (fold, seed), fold and seed outer;
- a clock check before each run;
- VLCS from an input folder with DomainBed's exact counts, otherwise streamed from the Hugging Face mirror (F15);
- an environment record in `m2cl_v2/env/`.

It refuses P100 and requires internet (ImageNet weights).

Tested on the laptop with a simulated Kaggle: the real code zip as Kaggle unpacks it, the real PACS (9,991 images), a fake VLCS input with DomainBed's counts, a fake T4, a fake clock, and training replaced by writing a complete `result.json` from a real template (30 minutes per run). Results:
- Session 1 (C, D) ran 22 runs and stopped at 11.0 h.
- Session 2, with session 1's output as input, copied 23 files (22 results and the env record), continued at the 23rd run, finished C (36 of 36) and stopped at 11.0 h with D at 8 of 16.
- Run names match `train_lodo.py`; the plan has 20/12/36/16 runs; the analysis script read the outputs (S4 n=7 after session 1, n=12 after session 2) and wrote its JSON.

It has not run on Kaggle yet: real training, Kaggle's unzip layout and the version-output mechanics are untested.

### F17 - Truncated VLCS images (7 Nov 2026)
The first Kaggle session failed in the first VLCS run (M2, Caltech101, seed 0, epoch 1): PIL `OSError: image file is truncated (4038 bytes not processed)`. DomainBed's VLCS contains truncated JPEGs, and DomainBed and the authors' code load them by setting `ImageFile.LOAD_TRUNCATED_IMAGES = True` (`domainbed/datasets.py`, line 15, commit 0b1bce3). Our loader (`src/m2cl/data.py`) now sets the same flag.

This cannot change any PACS result. All 9,991 RGB PACS images decode without error with the flag off, and every one decodes to byte-identical pixels with the flag on and off (checked on the laptop). Runs from the Colab zips without the flag (all of round 3's A, B and D) are therefore equivalent.

- New test `tests/test_vlcs.py::test_truncated_jpeg_loads`: a cut JPEG fails without the flag with the same error and loads with it. 21 of 21 tests pass.
- `colab_upload/M2CL_code.zip` was rebuilt; the previous zip is kept as `M2CL_code_2026-11-06.zip`.
- The Kaggle notebook adds the same line to an older uploaded zip if it lacks it, recorded as `code_zip_patched_for_truncated_images` in `env/`. It also lists every truncated VLCS image in `env/vlcs_truncated_images.json`, so the paper can report how many there are.
- Tested on the 6 Nov and the new zip: the flag is on after import in both. The simulated Kaggle sessions still run end to end.

### F18 - Part C moves to the institute GPU (7 Nov 2026)
VLCS on Kaggle ran at about 125 s per epoch. The cause is CPU decoding, not the GPU: VLCS JPEGs average 358 KB (archive size / 10,729), against 19 KB and about 52k pixels for PACS. The pre-registered config fixes two loader workers, and changing that (or pre-shrinking the images) would change the augmentation streams or the input pixels, so neither was done. The user's institute provides Linux machines with an RTX Blackwell GPU and an Intel Core Ultra 9. Part C moves there (`docs/PREREGISTRATION.md`, Amendment 2) with the new `scripts/run_part_c.py`.

The script:
- checks the GPU and that the PyTorch build has kernels for it (Blackwell needs PyTorch 2.7 or newer with CUDA 12.8);
- streams VLCS from the verified mirror (F15) with tarfile's safe "data" filter, or uses an existing copy, and requires DomainBed's exact counts;
- refuses code without the truncated-image line (F17) and records the truncated images;
- downloads the ImageNet weights once before parallel runs;
- runs the 36 runs with fold and seed outer, `--parallel N` at a time, one log per run;
- stops starting runs after a failure, skips finished runs on restart (unfinished ones resume through the engine), and refuses a second copy on the same folder through a lock file;
- writes an environment record and runs `analyze_round3.py` at the end;
- has a `--smoke` mode: one 1-epoch M2-CL run in a separate folder.

Tested on the laptop with a fake 10,729-image archive served as file:// (one truncated JPEG) and training replaced by a fake trainer that writes the real config:
- the smoke test passes;
- a 3-way parallel run with one forced failure stopped new starts, let the running ones finish (14 finished) and left no lock;
- a restart skipped those 14 and finished all 36;
- the truncated image was recorded, 36 logs were written, and the analysis read the results (S4 n=12) and wrote its JSON;
- a stale lock is ignored.

Not yet run on the real machine. Setup steps: `docs/RUN_INSTITUTE.md`.

### F19 - Round-3 Colab results merged (7 Nov 2026)
The three Drive downloads were merged into `Downloads/m2cl_v2_round3_colab.zip` (2.20 GB, 69 files under `m2cl_v2/`, every CRC checked, a manifest included):
- `round3-20261107T105938Z-1-001.zip`: Part B;
- `drive-download-20261107T105918Z-1-001.zip`: account A;
- `drive-download-20261107T105956Z-1-001.zip`: account D.

No run appeared in more than one download. Contents:
- Part B: 12 of 12 ERM runs.
- Part A: M2 seeds 1-2, 8 of 8 (seed 0 is in `lr.zip`); M2-CL 3 of 12 (art seeds 0-2), plus cartoon seed 0 unfinished with `latest.pt` at epoch index 12 (its `latest_prev.pt` is at epoch index 1, an older Drive sync; both have the round-3 config).
- Part D: M2 8 of 8; M2-CL art seeds 3-4 finished, plus cartoon seed 3 unfinished (`latest.pt` at epoch index 1).

Interim output of `analyze_round3.py`. This is a progress report, not the pre-registered decision, which needs all 12 pairs:
- the primary result is INCOMPLETE (3 of 12 pairs);
- manipulation check: the M2-CL vs M2 backbone distance is 0.418-0.435 in 3 of 3 pairs, against the inert level of 0.033, so at alpha 0.07 the term changes the network;
- S2: n=12, +1.87 pp, Holm p 0.0015;
- S3 M2 - ERM: n=12, +0.33 pp, p 0.62.

**Update (7 Nov 2026, 21:44).** `Downloads/m2cl_round3.zip` (the Part A account's `round3/` folder) was merged in. Every file it shared with the earlier zip had the same CRC and size. Its new runs are M2-CL cartoon seeds 0-2, photo seeds 0-2 and sketch seed 0, all complete, with lr 1e-3, alpha 0.07 and 30 epochs in their configs. `latest.pt` and `latest_prev.pt` of finished runs (art seed 0, cartoon seed 0) were left out; the manifest lists them. The previous zip is kept as `m2cl_v2_round3_colab_2026-11-07_1635.zip`. The new zip passed a CRC check and the notebook's restore and status cells: A M2-CL is 10 of 12 (sketch seeds 1-2 left) and D M2-CL is 2 of 8 (cartoon seed 3 still at the same `latest.pt`). Interim, not the decision: n=10, M2-CL - M2 +0.18 pp (95% CI -0.51 to +0.87), sign-flip p 0.61. Backbone distance is 0.306-0.444 in 10 of 10 pairs.

### F21 - Parts A, B and D complete; pre-registered PACS decision (7 Nov 2026, 23:50)
Three more downloads were merged into `Downloads/m2cl_v2_round3_colab.zip` (2.54 GB, 96 files plus the manifest; every CRC checked):
- `round3-20261107T180623Z-1-001.zip` (Part A account): M2-CL sketch seeds 1-2;
- `round3.zip` (Part D account): M2-CL cartoon, photo and sketch seeds 3-4;
- `round3 part2.zip`: only stale checkpoints.

Of the files that appeared in more than one source, 116 were byte-identical and none conflicted. All 52 result files are `complete`, with lr 1e-3, 30 epochs, the seed matching the folder name, and alpha 0.07 (M2-CL) or 0 (M2). The `latest*.pt` files of finished runs were left out (listed in the manifest). The previous zip is kept as `m2cl_v2_round3_colab_2026-11-07_2144.zip`.

Output of `scripts/analyze_round3.py` on that zip plus `lr.zip` and `m2_v2.zip`, with `--grad` and `--data_root` (written to `results/round3/round3_analysis_pacs.json`):
- **Primary, H1/H2 (12 pairs): M2-CL - M2 = +0.44 pp, 95% CI -0.30 to +1.18; sign-flip p 0.24; TOST(+/-0.71) p 0.22. Decision: INCONCLUSIVE** (neither an effect nor equivalence). Per fold: art +0.07, cartoon +0.54, photo -0.16, sketch +1.33.
- Manipulation checks pass. Backbone distance is 0.268-0.444 in 12 of 12 pairs (inert level 0.033; the threshold was 10 of 12). The gradient share at M2-CL Art seed 0 is 0.148 and 0.22 (the threshold was 0.05). The null therefore can't be blamed on an inert term.
- S2, M2 at lr 1e-3 vs lr 5e-4: +1.87 pp, Holm p 0.0015.
- S3, M2 - ERM: +0.33 pp, Holm p 0.62. S3, M2-CL - ERM: +0.77 pp, Holm p 0.32.
- Macro averages: ERM 77.33, M2 77.66, M2-CL 78.10.
- D, seeds 3-4 (8 pairs; reported separately): +0.36 pp, 95% CI -0.21 to +0.93, sign-flip p 0.22.

Exploratory, not pre-registered:
- Pooling all 20 pairs (seeds 0-4) gives +0.41 pp, 95% CI -0.05 to +0.87, sign-flip p 0.075.
- The minimum detectable effect at 80% power is about 1.0 pp with 12 pairs and 0.65 pp with 20.
- Sketch is positive in 5 of 5 seeds (mean +1.09); photo is negative in 5 of 5 (mean -0.22). This is one of four folds checked, so it is a hypothesis, not a finding.
- The mean per-fold seed SD (seeds 0-2) is ERM 1.92, M2 1.24, M2-CL 0.87.
- In-domain validation accuracy is unchanged (M2-CL - M2 +0.06 pp).

**Deviation: the PyTorch version changed mid-experiment.** Colab upgraded its PyTorch during Part A:
- every M2 and ERM run, and M2-CL art seeds 0-4, used torch 2.11.0+cu130;
- M2-CL cartoon, photo and sketch (15 runs) used 2.14.1+cu130. Cartoon seeds 0 and 3 were resumed across the upgrade.

So 15 of the 20 M2-CL vs M2 pairs differ in torch version as well as method. Configs, data and seeds are the same, and cuDNN is deterministic. Different kernel versions can still change numerics, which adds noise and could add a small bias. The size of this effect is unknown until it is measured: retrain some M2 runs on 2.14.1 and compare them with their 2.11 results. The paper must report this.

### F20 - Part C running on the institute machine (7 Nov 2026)
The owner reports that Part C is running on the institute machine. Its GPU is an RTX Ada-generation card, not Blackwell; Amendment 2 now carries a dated correction. The owner reports about 95% GPU use, an estimate of about 7 hours for all 36 runs, and per-epoch numbers close to the Kaggle run.

The owner also reports making edits during setup. **Not yet recorded:** which files changed and how. Before the results are analysed, the edits must be listed here and the files compared with the package (`M2CL_PartC_institute.zip`). An edit that touches `configs/`, `src/` or the training path of `scripts/` is a deviation from the pre-registration and must be reported in the paper. **Listed 10 Nov 2026 in F26:** only the GPU check in `scripts/run_part_c.py`, plus a local note in the institute copy of this file.

### F22 - Checkpoint archive (8 Nov 2026)
Every checkpoint on the laptop was collected into `Honors/checkpoints_archive/` (zips 01-07, ordered by date, 10.2 GB; `INDEX.md` there describes each). Every file in the 17 original zips and the loose `.pt` files in `Downloads` was matched by CRC and size to a copy in the archive before the originals were moved to the Recycle Bin, with the owner's approval. Archive paths for files cited elsewhere in this log:
- `Downloads/lr.zip`, `dose.zip` and `erm.zip` are now in zip 06;
- `m2_v2.zip` and `m2cl_v2.zip` are in zip 05;
- `m2cl_v2_round3_colab.zip` and the round-3 downloads are in zip 07;
- the June 4-tap checkpoints are in zip 02;
- the August contaminated-split runs are in zip 04.

Running `analyze_round3.py` on zips 05-07 reproduces F21 exactly. Not found anywhere: the 4-tap Sketch M2/M2-CL checkpoints (61.06/61.03) and the contaminated-split ERM Sketch seed 0.

### F23 - Public repository (8 Nov 2026)
The repository was pushed to https://github.com/MaxRC2007/M2CL-Reproduction. GitHub records its creation at 2026-11-07T19:01:48Z and its first push at 19:01:56Z (00:31 IST on 8 Nov); the first commit is `65c8a5b`. At that time Part C (VLCS) was still running on the institute machine, and no Part C comparison had been computed or viewed. **Corrected 10 Nov 2026 (F26):** this sentence is wrong. The Part C files show that Part C finished at 23:15 IST on 7 Nov, and the institute script wrote its VLCS comparison at 23:15:30 IST, 1 h 16 min before the push. The push therefore does not show that the VLCS analysis was public before the results existed. The protection for Part C is that the analysis code was frozen on 6 Nov (F26). The paper draft and the internal review are excluded (`.gitignore`) until submission. A Software Heritage archive snapshot is still to be requested by the owner (`docs/TIMESTAMP_EVIDENCE.md`). The earlier public stages are linked from the README.

**F21 addendum (8 Nov 2026).** The owner decided not to retrain the M2 runs on torch 2.14.1. The PyTorch version change is reported in the paper as a limitation of the PACS primary test, not measured.

### F24 - Exploratory leakage re-probe of the round-3 checkpoints (8 Nov 2026, finished 05:02 IST)
This is not pre-registered. The leakage sections of the draft probed networks whose contrastive term was inert. To probe networks where it is active, the 36 round-3 best checkpoints (M2 and M2-CL at alpha 0.07, ERM; lr 1e-3; seeds 0-2) were extracted from archive zips 06 and 07 into `Honors/probe_work/round3_ckpts`. They are probed with the same pipeline and settings as `results/leakage/raw_*`:
- driver: `scripts/run_probes_round3.py`;
- inputs: `configs/leakage_manifest_round3.yaml` and `configs/comparisons_round3.yaml`;
- features in `features_r3/`.

Families:
- M2-CL vs M2 at every extraction block and the backbone;
- each method vs the ImageNet-only control.

It runs on the laptop CPU. The log is `results/leakage/round3.log`.

**Result (finished 8 Nov 05:02 IST; about 4 h with feature extraction).** All 36 checkpoints were probed: 4 targets x 3 seeds per method, 5 probe seeds each. Outputs are in `results/leakage/round3_taps` and `round3_erm`, with tables in `results/tables/round3_taps/tables.md` and `round3_erm/tables.md`. The probe noise floor is SD 0.10-0.31 pp. The class-probe sanity check passes (source-test class accuracy 96.9-97.7% for trained networks, 82-89% for ImageNet).

Domain-probe accuracy, M2-CL minus M2 (12 seed-matched pairs; mean, with range):

| tap | M2 | M2-CL | difference | pairs lower |
|---|---|---|---|---|
| eb1a, eb1, eb2a, eb2, eb3 | 90.8-92.6 | the same | +0.04 to +0.12 (each pair within -0.58 to +0.82) | 1-7 of 12 |
| **eb4** | **90.0** | **76.8** | **-13.2 (-17.2 to -7.2)** | **12 of 12** |
| eb_concat | 96.9 | 96.6 | -0.30 (-0.89 to -0.02) | 12 of 12 |
| backbone (GAP of layer4) | 98.0 | 97.9 | -0.12 (-0.23 to -0.03) | 12 of 12 |

- **The active term removes domain information at one place only, eb4 (the last extraction block).** It does so in every pair and every target domain, Benjamini-Hochberg significant in all 12. Class information at eb4 is kept: source class accuracy +0.09 pp, target class accuracy +2.44 pp. So this is not a collapse of the representation.
- The loss is applied with equal weight to all six taps, yet the five earlier taps show no change beyond the probe noise.
- **The classifier's input still carries the domain.** The classifier sees all taps plus the backbone GAP. eb_concat and the backbone change by -0.3 and -0.1 pp. These shifts are consistent in sign but 40-100x smaller than at eb4, and domain identity stays 96-98% decodable there.
- **No method reduces backbone domain information meaningfully.** Against the ImageNet-only backbone, ERM, M2 and M2-CL are all 0.1-0.8 pp lower. Their 12-pair ranges overlap: ERM -0.11 to -0.76; M2 -0.16 to -0.74; M2-CL -0.29 to -0.84.
- Reading: at an active strength (alpha 0.07, gradient share in the range of the authors' code), the contrastive term reshapes the block it acts on most strongly. Domain information is still present in the other taps and the backbone, which reach the classifier. This fits the inconclusive accuracy result (+0.44 pp, F21). It replaces the draft's leakage claims, which were measured on networks with an inert term (F8-F9).
- The 6-tap pattern (eb4 only) is an observation, not a tested hypothesis. Why the earlier taps are unaffected (for example, where the loss gradient concentrates) is not established.
- The `[warn] family r3_cl_effect_at_taps: no matching conditions` line in the log comes from the ERM-only table pass. That pass has no M2/M2-CL groups, so the warning is expected; the family is reported in `round3_taps`.
- **Status:** exploratory, not pre-registered. Round 4 (F25) does not probe representations.

### F25 - Round 4 planned: the authors' own code (8 Nov 2026)
`docs/PREREGISTRATION_ROUND4.md` was written and pushed before any run. It covers the authors' `M2CL` at commit 0b1bce3, unmodified, ResNet-18, `lparam` 0.01 vs 0, 12 seed-matched pairs on PACS, on the institute machine. It uses the same decision rule and margin as round 3.
- Runner: `scripts/run_round4_authors.py`. It pins and verifies the commit, checks DomainBed's PACS counts, has a smoke mode, runs jobs in parallel, and restarts any unfinished run from scratch.
- Analysis: `scripts/analyze_round4.py`.
- Tests: `tests/test_round4.py`, which runs the full 24-run flow against a stand-in for DomainBed's `train.py`. The authors' code itself could not be executed on the laptop (the assistant's sandbox blocks running downloaded code), so its first real execution is the institute smoke run.

### F26 - Part C (VLCS) results (10 Nov 2026)
The owner brought `m2cl_v2_C.zip` (1.73 GB) from the institute machine on 10 Nov.
- **Contents:** 36 complete runs (ERM, M2, and M2-CL at alpha 0.07; lr 1e-3; seeds 0-2 x 4 VLCS folds), with logs, `env/`, the list of truncated VLCS images and the institute's own `round3_analysis_institute.json`.
- **Archive:** archived unchanged as `checkpoints_archive/08_2026-11_round3_vlcs_C.zip`; all 112 CRCs were checked.
- **In the repo:** the logs, `env/`, the 36 `result.json` files and the institute analysis are in `results/round3/vlcs_institute/`.
- **Settings:** every run is `complete`, with the pre-registered settings (the analysis refuses others).

**Hardware:** NVIDIA RTX 4000 Ada (sm_89, 20 GB), Intel Core Ultra 9 285 (24 threads), Linux 6.17, Python 3.12.3, torch 2.11.0+cu128, cuDNN 9.19, 2 runs in parallel. This is the same torch version as all PACS M2 runs. Runs began 7 Nov 17:20 IST (11:50 UTC) and the last finished at 23:15 IST. Zip entry times are institute local time; the env file names give UTC.

**Interruption and resume:**
- At about 17:38 IST the two first runs, `m2_6tap_Caltech101_seed0` and `m2cl_6tap_Caltech101_seed0`, both crashed during the epoch-22 validation pass. The error was `PIL.UnidentifiedImageError: cannot identify image file .../VLCS/LabelMe/person/713ac5c609bc4de4b63211a493b0cc6b.jpg`.
- That image had loaded in each of the previous 21 epochs.
- The runner was restarted at 17:44 IST (second `env/` record). Both runs resumed from their epoch-22 checkpoint (`latest.pt`, written at the end of epoch 21). After the restart, the image counts were unchanged (train 7,449; val 532/657/676; test 1,415).
- **Both runs selected epoch 19**, before the crash. Their reported test accuracies therefore come from the uninterrupted part of training.

**Cause (owner's account, 10 Nov 2026, from the Antigravity session on the institute machine):**
- During epoch 22 the project folder `~/Desktop/S20240030357/M2CL_PartC_institute` (holding `m2cl_v2/`) was moved to `~/Desktop/S20240030357/M2CL`. The Python environment `m2cl-env` was moved at the same time.
- At 17:44 symlinks from the old paths were created and the same command was restarted.
- This fits the evidence, with one correction to the session's explanation (open file descriptors). The data folder `~/data/VLCS_raw` was not moved.
  - The loaders do not keep workers alive between passes, so the validation pass forks fresh workers.
  - The main process never opens images, so Pillow's JPEG plugin is first imported inside those workers, from the environment's path.
  - With the environment moved, that import failed. `PIL.Image.preinit` swallows plugin `ImportError`s, and `Image.open` then raises `UnidentifiedImageError` on the first file it reads, here a LabelMe image.
  - So the image itself was never damaged, and no VLCS file changed. The image counts after the restart are identical, and both `env/` records show the same torch, CUDA and cuDNN.

**Setup edits (owner's account, 10 Nov 2026; the folder is not a git checkout, so there is no `git diff`).** Two files were changed before any run:
- `scripts/run_part_c.py`, `check_gpu()` only:
  - torch 2.11.0+cu128 lists no `sm_89` binary, so the shipped check refused the RTX 4000 Ada;
  - the edit accepts `sm_89` when `sm_86` is listed (Ada runs `sm_86` kernels and PTX JIT);
  - the live mixed-precision test kernel still runs.

  The edit is in the GPU gate only, not in training, data or evaluation. The traceback line numbers in `engine.py`, `data.py` and `train_lodo.py` match the shipped package.
- The institute's copy of `docs/AUDIT.md` gained a local note numbered "F20" describing that edit. It differs from F20 in this log, which was written separately on the laptop.

**Not a deviation from the pre-registration:** neither change touches the configs, the model, the data or the analysis. The exact diff is recovered by the round-4 add-on check (F27). It is reported in the paper's compute and software table.

**Pre-registered analysis.** `scripts/analyze_round3.py` was run on zips 05-07 plus `m2cl_v2_C.zip`, with `--grad` and `--data_root`, and written to `results/round3/round3_analysis_all.json`.
- PACS is unchanged from F21. Primary +0.44 pp, CI -0.30 to +1.18, sign-flip p 0.24, TOST p 0.22: **INCONCLUSIVE**. Manipulation checks pass.
- **S4 VLCS, M2-CL - M2: +0.08 pp, 95% CI -0.37 to +0.53, sign-flip p 0.72, TOST(+/-0.71) p 0.0050.** Per fold: Caltech101 +0.09, LabelMe +0.77, SUN09 +0.15, VOC2007 -0.70.
  - The 90% CI implied by TOST lies inside +/-0.71, so on VLCS the contrastive term is **equivalent to no effect** at the pre-registered margin.
  - Holm is pre-registered for the S2-S4 sign-flip tests, not for TOST. Even a Bonferroni factor of 6 on the TOST p gives 0.030 < 0.05.
- S4 VLCS, M2 - ERM: +0.37 pp (CI -0.69 to +1.43, p 0.47). M2-CL - ERM: +0.44 pp (CI -0.40 to +1.29, p 0.26).
- **Holm across S2-S4, now complete:** S2 p 0.0029 (was 0.0015 in F21, computed before S4 existed; still significant). Every S3 and S4 test has Holm p >= 0.80.
- Macro averages on VLCS: ERM 76.34, M2 76.70, M2-CL 76.78. **Checked 10 Nov against the paper's Table I** (ResNet-18; `Honors/references/M2CL.pdf`): ERM 69.81, M2 76.11, M2-CL 77.78. So the paper's VLCS contrastive gain is **+1.67 pp**, outside our 95% CI (-0.37 to +0.53).
  - Our M2 is 0.6 pp above the paper's; our M2-CL is 1.0 pp below.
  - Our ERM is 6.5 pp above the paper's ERM.
  - The paper prints its PACS ERM average as 80.53 (recomputed from its per-domain values: 80.52).
- The institute's own analysis gives identical S4 numbers.

**Exploratory manipulation check on VLCS (not pre-registered).** The same-seed backbone distance between M2-CL and M2 is above the inert level 0.033 in 12 of 12 pairs (0.262-0.403, median 0.355). This is the same range as PACS (0.268-0.444), so the term was active on VLCS too. For scale, M2 seed 0 vs seed 1 differ by 0.89-1.10.

**Reading:**
- The contrastive term, at a strength that changes the networks, produced no detectable gain: inconclusive on PACS, equivalent to zero on VLCS.
- The lr change (S2) remains the only significant effect.
- The paper's VLCS number is not reproduced as a contrastive effect.

### F27 - Round-4 package as an add-on to the Part C folder (8-10 Nov 2026)
`M2CL_Round4_addon.zip` replaces the full `M2CL_Round4_institute.zip`.
- **What it adds:** only round 4's own files, unzipped with `unzip -n` into the Part C folder (now `~/Desktop/S20240030357/M2CL`). It overwrites nothing.
- **Code:** byte-identical to the pre-registration commit 715a2d1.
- **Checks it adds** (brief: `docs/START_HERE_ROUND4.md`):
  - `ROUND4_SHA256SUMS.txt` covers the add-on and `scripts/analyze_round3.py`;
  - `ROUND4_reference/PARTC_SHA256SUMS.txt` covers the 52 other shipped Part C files, which verifies independently that nothing else was edited;
  - `ROUND4_reference/run_part_c.py.orig`, diffed against the institute copy, recovers the F20 edit as `~/edits.diff`, which comes back inside `m2cl_round4.zip`.
- **Round 4 imports `check_gpu` from the institute's edited `run_part_c.py`.** The unedited check would refuse the RTX 4000 Ada. The edit concerns only that gate, so it is not a deviation from the round-4 plan.
- **Tested** on a copy of the Part C package with the owner's edit applied: no file was overwritten, all checksums matched, the diff showed exactly the four edited lines, and the runner's tests passed.

**Laptop pre-check of the authors' code (10 Nov 2026; run by the owner, not by the assistant; not part of the experiment).**
- Setup:
  - `scripts/check_authors_cpu.py` ran in a separate venv (`Downloads/r4check`): torch 2.5.1 CPU, wilds 2.0.0, timm 0.9.16.
  - It used the authors' repository at 0b1bce3, checked unmodified.
- It passed:
  - the authors' `algorithms`, `datasets` (which imports wilds) and `hparams_registry` import;
  - `M2CL` builds with ResNet-18 (18.0 M parameters, ImageNet weights `resnet18-5c106cde.pth` downloaded);
  - two SGD updates on 96 random images run at both settings. With lparam 0.01 the loss is 7.905 -> 7.895; with lparam 0.0 it is 1.936 -> 1.926, about ln 7, as expected for 7 classes.
  - `predict` returns the logits plus 3 activation tensors of width 16,384.
- Harmless warnings: a `SyntaxWarning` from a docstring in the authors' `lib/misc.py`, and wilds reporting that 2.0.1 exists.
- Not covered: data loaders, PACS and the GPU. Those are first exercised by the institute smoke run.

### F28 - Torch-version check planned (10 Nov 2026, before any run)
The owner decided to measure the F21 deviation after all (reversing the 8 Nov addendum).
- **Why:** in the round-3 PACS primary test, every M2 run used torch 2.11.0, and 15 of the 20 M2-CL runs used 2.14.1.
- **Runs:** the four seed-0 M2 runs of the primary test (`configs/round3_m2.yaml`, lr 1e-3, one per fold), retrained on a Colab T4 twice:
  - on torch 2.14.1;
  - on torch 2.11.0, as a same-version control for GPU nondeterminism (cuDNN runs deterministic, but pooling backward on CUDA is not guaranteed bit-exact).
  - Notebook: `notebooks/TorchCheck_Colab.ipynb`. It pins torch from the cu130 index, requires a T4, writes to `m2cl_v2/torchcheck/torch_<version>/`, and records `env_*.json`.
- **Analysis**, fixed in `scripts/analyze_torchcheck.py` before the runs:
  - version shift = 2.14.1 rerun minus 2.11.0 rerun;
  - rerun noise = 2.11.0 rerun minus the original (archive zip 06);
  - backbone distances to the original for both versions.
- **Reading** (n = 4, descriptive only): NEGLIGIBLE if the mean |shift| <= max(2 x mean |noise|, 0.25 pp) and the |mean shift| <= 0.22 pp (half the primary +0.44); otherwise the shift is REPORTED beside the primary result as a possible bias of that size.
- **Tested** on fake reruns built from the real originals: both verdicts, and the refusal of a run on the wrong torch version.
- The primary decision (F21) is not recomputed or changed by this check.
- **Result (10 Nov 2026, 21:21 IST):**
  - **Inputs:** the owner downloaded two zips from Drive, `torchcheck-…154219Z` (torch_2.11.0) and `torch_2.14.1-…154301Z`. Both passed `unzip -t`.
  - **Environments:** the 2.11.0 rerun ran on torch 2.11.0+cu130 with cuDNN 9.19 on a Tesla T4. The 2.14.1 rerun ran on 2.14.1+cu130 with cuDNN 9.28, also a T4. The original runs were also 2.11.0+cu130. So "version" here means torch together with its bundled cuDNN.
  - **Run:** `python -I scripts/analyze_torchcheck.py probe_work/torchcheck_in checkpoints_archive/06_….zip --json results/torchcheck/torchcheck_analysis.json`. All 12 runs were complete and matched the round-3 M2 config.

  | fold | original | 2.11.0 rerun | 2.14.1 | shift | noise | best epochs (orig/2.14.1/2.11) | dist 2.14.1 / 2.11 vs orig |
  |---|---|---|---|---|---|---|---|
  | art_painting | 75.88 | 76.61 | 75.10 | -1.51 | +0.73 | 23 / 28 / 24 | 0.096 / 0.099 |
  | cartoon | 70.99 | 70.82 | 71.12 | +0.30 | -0.17 | 20 / 24 / 20 | 0.115 / 0.092 |
  | photo | 95.75 | 95.69 | 95.75 | +0.06 | -0.06 | 16 / 16 / 29 | 0.098 / 0.228 |
  | sketch | 66.91 | 65.67 | 65.72 | +0.05 | -1.25 | 24 / 19 / 19 | 0.085 / 0.085 |

  - **Summary:** mean version shift -0.28 pp (mean |shift| 0.48). Mean rerun noise -0.19 pp (mean |noise| 0.55).
  - **Verdict: REPORTED**, as fixed in advance:
    - the first condition passes (0.48 <= 1.11);
    - the second fails (|-0.28| > 0.22);
    - art_painting alone drives the signed mean.
  - **Reading:**
    - The mean |shift| between versions is smaller than the mean |difference| between two runs on the *same* version. The check therefore cannot tell a version effect apart from ordinary rerun variation.
    - As pre-specified, the -0.28 pp is reported beside the primary result as a possible bias of that size.
    - Its sign means 2.14.1 scored M2 lower. Most M2-CL runs used 2.14.1, so a real effect of this sign would make the primary estimate *understate* M2-CL's gain.
  - **Primary decision:** unchanged (inconclusive). Even moving the whole interval by +0.28 would leave it crossing zero (about -0.02 to +1.46). That shift is an illustration only, not a recomputed test.

### F29 - Exploratory VLCS leakage probe (10 Nov 2026, started 18:31 IST)
This is the VLCS replication of F24 and is not pre-registered. It asks whether, on a second dataset, the active term again lowers domain decodability only at eb4.
- **Data:** VLCS from the verified Hugging Face mirror (`VLCS.tar.gz`, 3,838,010,386 bytes; F15). It was extracted with tarfile's `data` filter into `Honors/data_vlcs/extracted/VLCS` and holds exactly DomainBed's counts (1,415 / 2,656 / 3,282 / 3,376; 5 classes in every domain).
- **Networks:** the 36 VLCS best checkpoints, copied from archive zip 08 into `Honors/probe_work/round3_vlcs_ckpts`.
- **Settings:** the same pipeline and settings as F24:
  - driver: `scripts/run_probes_round3.py --dataset vlcs`;
  - configs: `configs/leakage_manifest_round3_vlcs.yaml` and `comparisons_round3_vlcs.yaml`;
  - features in `features_r3_vlcs/`; log in `results/leakage/round3_vlcs.log`;
  - tables in `results/tables/round3_vlcs_{taps,erm}`.
- **Code changes, made so the pipeline accepts VLCS:**
  - `features.py`: the class count and the domain order come from the dataset;
  - `extract_features.py`: passes `num_classes` from the manifest;
  - `run_leakage.py`: looks up the domain index for VLCS targets;
  - `make_tables.py`: lists the VLCS domains.

  For PACS the sample order, and so every earlier result, is unchanged. The unit tests pass. The full chain was tested on a tiny fake VLCS with the real checkpoints.

**Result (finished 11 Nov 2026, 02:04 IST).**
- **Timing:** features took 19:05-00:18 (37 files, slowed twice when the laptop ran on battery). M2 / M2-CL probes 00:18-01:55, ERM probes 01:55-02:04, then tables.
- **Coverage:** all 36 checkpoints and the ImageNet control were probed, 5 probe seeds each, with no errors.
- **Outputs:** `results/leakage/round3_vlcs_{taps,erm}`, tables in `results/tables/round3_vlcs_{taps,erm}/tables.md`.
  - The ERM table step prints `family r3v_cl_effect_at_taps: no matching conditions`. This is expected: the ERM run has no extraction blocks, and PACS printed the same warning.
- **Sanity checks:**
  - Probe-seed noise floor: mean SD 0.15-0.49 pp per layer.
  - Class probe on source-test data: 84-88% for trained networks against 78-83% for ImageNet. VLCS classes are harder than PACS classes.

M2-CL minus M2 domain-probe accuracy, 12 network pairs per layer (4 targets x 3 seeds), BH q = 0.05 within the table:

| layer | VLCS mean (range) | VLCS sig. lower | PACS mean (F24) | PACS sig. lower |
|---|---|---|---|---|
| eb1a | +0.05 (-0.38, +0.42) | 0/12 | +0.12 | 0/12 |
| eb1 | +0.01 (-0.29, +0.38) | 0/12 | +0.04 | 0/12 |
| eb2a | -0.07 (-0.57, +0.30) | 0/12 | +0.09 | 0/12 |
| eb2 | -0.01 (-0.58, +0.83) | 0/12 (1 sig. higher) | +0.07 | 0/12 |
| eb3 | +0.45 (-0.39, +1.12) | 0/12 | +0.10 | 0/12 (1 sig. higher) |
| **eb4** | **-4.42 (-7.67, -0.92)** | **9/12** | **-13.23** | **12/12** |
| eb_concat | -0.88 (-2.25, +0.29) | 6/12 | -0.30 | 1/12 |
| backbone | +0.03 (-0.31, +0.27) | 0/12 | -0.12 | 0/12 |

**Reading (exploratory):**
- **The PACS pattern replicates on VLCS.** The active contrastive term lowers linear domain decodability at the last extraction block, eb4, and nowhere earlier. It leaves the backbone feature that the classifier reads unchanged.
- **eb4 details:**
  - All 12 eb4 differences are negative.
  - The 3 that are not significant are LabelMe seed 1 (-1.40), SUN09 seed 2 (-1.02) and VOC2007 seed 1 (-0.92).
  - The effect is about a third of the PACS size.
- **eb_concat:** the drop reaches the joined blocks on VLCS more often than on PACS (6/12 against 1/12). It is still small (mean -0.88).
- **Target accuracy:** on both datasets it is unchanged at the round-3 level (VLCS S4 is equivalent to zero, F26). The term reshapes the representation it is attached to without moving the representation used for prediction, or the accuracy.

### F30 - Round 4 result: the authors' M2CL code, lparam 0.01 vs 0 (11 Nov 2026)
This is the pre-registered test in `docs/PREREGISTRATION_ROUND4.md` (pushed in 715a2d1 before any run).
- **Inputs:**
  - The owner brought back `m2cl_round4.zip` (1,597,848,885 bytes, SHA-256 prefix `20e5d68f76358294`; `unzip -t` OK).
  - It was extracted to `Honors/probe_work/round4_in`.
  - 24 of 24 runs are complete: `done`, `results.jsonl` and `model.pkl` for every run, plus 24 logs, `env/`, `edits.diff` and the institute's own analysis.
- **Environment** (`env/institute_round4_20261110T112021Z.json`):
  - NVIDIA RTX 4000 Ada (sm_89) with an Intel Core Ultra 9 285;
  - torch 2.11.0+cu128, cuDNN 9.19, Python 3.12.3;
  - the authors' commit `0b1bce3`, `--parallel 2`.
  - The runs started 10 Nov 16:50 IST, and the last `done` file is dated 11 Nov 10:15 (zip entry time).
- **Edits:**
  - `edits.diff` contains only the `check_gpu` change already recorded in F26 (sm_89 is accepted when sm_86 kernels are present). No other code differs from the package.
  - Two runs have a non-empty `err.txt`: the warning "WILDS package is out of date (2.0.0 vs 2.0.1)". It has no effect on PACS.
- **Analysis:**
  - `python -I scripts/analyze_round4.py probe_work/round4_in/m2cl_round4 --json results/round4/round4_analysis.json`
  - It agrees with the institute's `round4_analysis_institute.json` to within 1e-6 on every run, every paired difference and the manipulation checks.

**Primary (pre-registered):** lparam 0.01 minus lparam 0, PACS, 12 pairs (4 folds x 3 seeds):
- mean **+0.81 pp, 95% CI [-0.28, +1.91]**;
- t p = 0.130, sign-flip p = 0.132, TOST (+/-0.71) p = 0.580.
- **Decision: INCONCLUSIVE.** Neither an effect nor equivalence is shown.
- Per fold: art_painting +0.83, cartoon +2.01, photo -0.22, sketch +0.64.

**Manipulation checks (both pass):**
- the contrastive gradient share at the ImageNet start is 0.109-0.127 (pre-registered: at least 0.05);
- the paired weight distance over the between-seed median is 0.298-0.338, above 10% in 12 of 12 pairs (pre-registered: at least 10 of 12).

**Secondary (descriptive), test accuracy as mean +/- SD over 3 seeds:**

| arm | art_painting | cartoon | photo | sketch | macro |
|---|---|---|---|---|---|
| lparam 0.01 (M2-CL) | 81.68 +/- 2.77 | 76.90 +/- 0.43 | 95.98 +/- 0.93 | 73.77 +/- 4.66 | **82.08** |
| lparam 0 (term off) | 80.84 +/- 0.91 | 74.89 +/- 1.89 | 96.21 +/- 0.35 | 73.13 +/- 3.16 | **81.27** |
| paper (ResNet-18) | | | | | M2-CL 83.54, M2 82.83 |

**Reading:**
- **Same verdict in both code bases.** The authors' own code gives the same answer as our reimplementation on PACS (round 3: +0.44 [-0.30, +1.18], inconclusive).
  - Both point estimates are positive and near the paper's PACS gain (+0.71).
  - Neither design can confirm or exclude a gain of that size with 12 pairs.
  - The authors' code has larger seed spread: sketch SD 4.66 and art_painting 2.77, so its interval is wider.
- **Pooling:** the round-3 and round-4 PACS results are not pooled into a confirmatory test, because pooling was not pre-registered. Any combined view is descriptive only.
- **Reproduction of the paper's numbers:** at these settings the released code reproduces the paper's M2-CL macro to within 1.5 pp (82.08 against 83.54). The lparam-0 arm is 1.6 pp below the paper's separate M2 model, which is a different network.
- **The decision rule leaves no follow-up.** Pre-registration allowed one only for a clear gain, and none was found. No further run is planned.

### F31 - The Kaggle VLCS hardware-check run (11 Nov 2026)
- **What happened:** the owner confirmed that the Kaggle VLCS run (M2, Caltech101, seed 0; Amendment 2) was stopped because it ran too slowly, and never completed. No Kaggle VLCS run was run again.
- **Effect:** no Kaggle run produced a result, so nothing from Kaggle enters any analysis. The paper (v4) says so in its compute section.

### F32 - Round 5 planned: the authors' code on VLCS (11 Nov 2026, before any run)
- **Why:** an independent review of the v4 draft (Fable 5.2, 11 Nov) named the main remaining objection. The decisive VLCS result (equivalent to zero, excluding the published +1.67 pp) rested on our reimplementation alone, while the authors' code had been run only on PACS. The owner decided to run it.
- **Design:** `docs/PREREGISTRATION_ROUND5.md`, round 4 with only the dataset changed.
  - VLCS, `lparam` 0.01 vs 0, seeds 0-2, 24 runs on the institute machine.
  - The same ±0.71 pp margin and decision rule as rounds 3-4.
  - Also reported: whether the CI excludes +1.67 pp.
- **Code:**
  - `scripts/run_round5_authors.py`: reuses round 4's runner helpers, which are unchanged. It finds a folder named `VLCS` with exactly DomainBed's four domains and counts, and records truncated images.
  - `scripts/analyze_round5.py`: round 4's analysis with VLCS domains and paper values.
  - `tests/test_round5.py`: 7 tests against a stand-in trainer.
  - `docs/START_HERE_ROUND5.md`: the brief for the institute machine.
- **Timing:** the pre-registration is written after the round-3 VLCS and round-4 PACS results were known. Its rules copy round 4 exactly, and this is disclosed in its section 7.
- **Paper (v4):** the Kaggle paragraph was removed at the owner's request. The sentence now reads "every completed training run is reported". The Kaggle attempt stays documented in F16-F18, F31 and the public Amendments 1-2.

### F33 - Round 5 result: the authors' M2CL code on VLCS, lparam 0.01 vs 0 (13 Nov 2026)
- **Run:** institute machine (RTX 4000 Ada, torch 2.11.0+cu128), `--parallel 2`. It started 11 Nov 2026 at 15:44 IST (`env/institute_round5_20261111T101431Z.json`), 19 minutes after the pre-registration was pushed (16fbf12, 15:25 IST). The last run finished 12 Nov at 22:42 IST.
- **Integrity:**
  - 24 of 24 runs complete, each with all 18 planned checkpoints (steps 0-4800 every 300, and 5000) and the pre-registered settings (checked by `analyze_round5.py`);
  - no restart, error or NaN in any log;
  - `edits.diff` is byte-identical to round 4's: the 4-line GPU-check patch for `sm_89`, outside training (F20);
  - four LabelMe JPEGs are truncated and were loaded with truncated-image loading, as in DomainBed and the released code (`env/vlcs_truncated_images.json`);
  - `m2cl_round5.zip` SHA-256 `22dd75d8ddf726823c31872a573599fe82f0969d26605ae7be7ec5a251c6ff69` (1.57 GB, kept outside the repository; model files not committed).
- **Reproduced:** `python -I scripts/analyze_round5.py m2cl_round5.zip --json results/round5/round5_analysis.json` agrees with the institute's `round5_analysis_institute.json` on every value (largest difference 5e-12, floating-point order).

**Primary (pre-registered):** lparam 0.01 − 0, 12 seed-matched pairs.
- Mean 0.00 pp (−0.0037), 95% CI [−1.24, +1.23]; t p 0.995, sign-flip p 0.992, TOST (±0.71) p 0.118.
- **Decision: INCONCLUSIVE.**
- The 95% CI upper limit (+1.23) **excludes the paper's VLCS gain of +1.67 pp** (descriptive, as pre-registered).
- 6 pairs positive, 5 negative, 1 tie. Per fold: Caltech101 −1.83, LabelMe +0.14, SUN09 +1.02, VOC2007 +0.65.
- MDE at 80% power (0.888 × SD 1.95): 1.73 pp.

**Manipulation check (passes):** paired weight distance over the between-seed median is 0.309-0.351, above 10% in 12 of 12 pairs (pre-registered: at least 10 of 12). The gradient share was not re-measured, as pre-registered.

**Secondary (descriptive), test accuracy as mean +/- SD over 3 seeds:**

| arm | Caltech101 | LabelMe | SUN09 | VOC2007 | macro |
|---|---|---|---|---|---|
| lparam 0.01 (M2-CL) | 95.76 +/- 3.98 | 61.63 +/- 0.31 | 70.58 +/- 0.87 | 75.18 +/- 1.26 | **75.79** |
| lparam 0 (term off) | 97.59 +/- 0.89 | 61.49 +/- 0.32 | 69.56 +/- 1.47 | 74.53 +/- 0.45 | **75.79** |
| paper (ResNet-18) | | | | | M2-CL 77.78, M2 76.11 |

**Reading:**
- **One pair dominates the spread.** Caltech101 seed 2 is −5.39 pp: validation picked step 2100 for the run with the term (test 91.17) and step 900 without it (96.55). It stays in, as pre-registered. Without it (exploratory): +0.49 [−0.19, +1.16], which also excludes +1.67.
- **Pooling (exploratory, not pre-registered):** rounds 4 and 5 together, 24 pairs: +0.40 [−0.37, +1.18].
- **Agreement with round 3 VLCS:** our reimplementation gave +0.08 [−0.37, +0.53], equivalent to zero. The authors' code has wider seed spread, so it cannot show equivalence, but both exclude the published gain.
- **Reproduction of the paper's numbers:** both arms average 75.79, 2.0 pp below the paper's M2-CL (77.78) and 0.3 pp below its M2 (76.11).
- **No further runs.** The pre-registration allows no follow-up. Showing equivalence within ±0.71 pp at this spread would need about 67 pairs (52 on PACS) for 80% power; adding seeds after seeing an inconclusive result would be optional stopping.
- **Paper (v4):** round 5 is added to the abstract, contributions, Section VII (text and Table `tab:round4`), Table `tab:primary`, Fig. 1 (fourth row; the −5.39 pair is drawn at the axis edge), the compute table, Discussion, Threats and Conclusion.
