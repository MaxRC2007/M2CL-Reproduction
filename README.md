# M2-CL on PACS: Reproduction, Corrected Objective and Domain-Leakage Analysis

Code for the honours project *"Multiscale Contrastive Learning for Domain Generalization: A Reproduction Study with a Corrected Objective, Multi-Seed Evaluation, and Representation-Level Leakage Analysis"*.

Madhusudan Reddy C., IIIT Sri City. Supervisor: Dr. Rajendra Prasath.

Original method: A. Ballas and C. Diou, *Multiscale and Multilayer Contrastive Learning for Domain Generalization*, IEEE TAI 5(12), 2024.

The repository answers two questions:

1. **Does M2-CL reproduce?** It is a ResNet-18 that taps intermediate layers with *extraction blocks* and adds a supervised contrastive loss on those taps. We reproduce it on PACS with leave-one-domain-out (LODO) evaluation, find and fix a loss-scaling bug (mean instead of sum, about 28× too weak), and compare against matched-seed ERM and a tap-matched M2 ablation.
2. **Does contrastive DG training change how much *domain identity* is linearly decodable from the representation?** We use frozen-feature linear probes, a random-init and ImageNet-only control, a paired bootstrap on a shared test set, and Benjamini-Hochberg correction.

> **Status (10 Nov 2026).**
> - The code base was rebuilt from the original notebook and probe scripts after an audit (see [docs/AUDIT.md](docs/AUDIT.md)). The audit found three issues that required re-runs:
>   - leakage probes that read a DCT copy of PACS instead of the RGB images;
>   - a train/validation split leak;
>   - unseeded probes.
> - The draft's earlier numbers are kept in `results/legacy_v1/`.
> - Round 3, the pre-specified decisive test of M2 vs M2-CL, is described in [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md). Its timing evidence is in [docs/TIMESTAMP_EVIDENCE.md](docs/TIMESTAMP_EVIDENCE.md).
> - Round 3 is finished:
>   - PACS (`results/round3/`): primary M2-CL − M2 = +0.44 pp [−0.30, +1.18], **inconclusive**;
>   - VLCS (`results/round3/vlcs_institute/`, audit F26): +0.08 pp [−0.37, +0.53], **equivalent to zero** within ±0.71 pp;
>   - the full analysis is in `results/round3/round3_analysis_all.json`.
>
>   Part C had already finished when this repository was first pushed (corrected in the timing evidence). Its protection is the analysis code frozen on 6 Nov.
> - Exploratory probes of the round-3 checkpoints (audit F24): the active contrastive term lowers domain decodability only in the last extraction block. The same pattern replicates on VLCS (audit F29), at about a third of the PACS size.
> - Round 4, the authors' own code with the contrastive weight 0.01 vs 0 (pre-registered in [docs/PREREGISTRATION_ROUND4.md](docs/PREREGISTRATION_ROUND4.md)): +0.81 pp [-0.28, +1.91], **inconclusive**, the same verdict as our reimplementation (audit F30, `results/round4/`).
> - Torch-version check (audit F28): mean shift -0.28 pp, no larger than same-version rerun noise (0.48 vs 0.55 mean |difference|); reported beside the primary result as the pre-specified rule requires.
> - The paper draft is not included until submission.
>
> **Earlier public stages of this project:**
> - [Reproducing-M2CL-Domain-Generalization](https://github.com/MaxRC2007/Reproducing-M2CL-Domain-Generalization): the first 4-tap reproduction notebook.
> - [Domain-Leakage-Analysis-on-PACS-and-M2CL](https://github.com/MaxRC2007/Domain-Leakage-Analysis-on-PACS-and-M2CL): the first leakage-probe analysis.

---

## Repository layout

```
M2CL-Reproduction/
├── src/m2cl/
│   ├── splits.py      PACS indexing, FIXED train/val split, probe split (NumPy only)
│   ├── data.py        transforms (= notebook), datasets, seeded loaders
│   ├── models.py      ERM, M2 / M2-CL with 4 or 6 taps (checkpoint-compatible names)
│   ├── losses.py      M2-CL contrastive loss, sum (paper) or mean (original bug)
│   ├── engine.py      training loop, per-epoch checkpoint + resume, single target evaluation
│   ├── features.py    frozen-feature extraction for probes (imagenet / random / checkpoint)
│   ├── probe.py       seeded linear probes, multi-seed correctness matrices
│   └── stats.py       paired bootstrap, BH, sign and sign-flip tests
├── scripts/
│   ├── train_lodo.py        train ERM / M2 / M2-CL over targets × seeds
│   ├── extract_features.py  cache features for every condition (manifest and/or finished runs)
│   ├── run_leakage.py       domain + class probes on cached features (CPU is fine)
│   ├── make_tables.py       every table → Markdown + LaTeX + CSV (no torch needed)
│   ├── audit_dct_data.py    evidence for audit A0: ImageNet probes on DCT vs RGB copies
│   ├── verify_checkpoints.py target accuracy of saved checkpoints on both copies
│   ├── compare_checkpoint_weights.py backbone weight distances (audit E2)
│   ├── run_overnight.py     resumable CPU driver: features, probes, tables, checkpoint check
│   └── smoke_test.py        full pipeline on a fake mini-PACS (offline)
├── configs/           locked training configs, leakage manifest, comparison families
├── notebooks/M2CL_PACS_Colab.ipynb   one-click Colab driver (T4)
├── tests/             pytest: stats, splits, loss = notebook loss, model keys, probe determinism
├── results/legacy_v1/ draft-v1 raw outputs (JSON) + per-run accuracies from the draft
├── results/tables/    generated tables
├── docs/AUDIT.md      issues found, evidence, fixes
└── paper/             LaTeX source of the draft
```

## Installation

```bash
pip install -r requirements.txt     # torch>=2.0, torchvision, numpy, scipy, scikit-learn, pandas, pyyaml, tabulate
python scripts/smoke_test.py        # must end with "SMOKE TEST PASSED"
```

PACS: the RGB copy, whose root contains `art_painting/ cartoon/ photo/ sketch/`, each holding the 7 class folders (9,991 images). Folders of `dct_*` files are rejected (docs/AUDIT.md, A0).

## Reproducing everything

**On Colab (recommended for training):** upload this folder and `PACS.zip` to Google Drive, open `notebooks/M2CL_PACS_Colab.ipynb`, set the three paths in the first cell, and run all cells. Every cell can be re-run after a disconnect and resumes where it stopped.

**On a local machine (Windows example, Anaconda prompt):**

```bat
cd C:\Users\Madhusudan\Downloads\Madhu\Honors\M2CL-Reproduction
set DATA=C:\Users\Madhusudan\Downloads\PACS\pacs_data\pacs_data

:: training (GPU strongly recommended)
python scripts\train_lodo.py --config configs\m2cl_6tap.yaml --data_root %DATA% --out_dir runs
python scripts\train_lodo.py --config configs\erm.yaml       --data_root %DATA% --out_dir runs
python scripts\train_lodo.py --config configs\m2_6tap.yaml   --data_root %DATA% --out_dir runs

:: leakage: works today on the EXISTING checkpoints (edit checkpoint_root in configs\leakage_manifest.yaml)
python scripts\extract_features.py --manifest configs\leakage_manifest.yaml --from_runs runs --data_root %DATA% --out_dir features
python scripts\run_leakage.py --features_dir features --out results\leakage\raw
python scripts\run_leakage.py --features_dir features --out results\leakage\zscore --standardize

:: tables
python scripts\make_tables.py --runs runs --acc_compare m2cl_6tap:erm m2cl_6tap:m2_6tap m2_6tap:erm --leakage results\leakage\raw --comparisons configs\comparisons.yaml --out results\tables\raw
```

### Paper table → source

| Draft table | Generated table (`results/tables/*/tables.md`) | Produced by |
|---|---|---|
| I, V, VI | `acc_<group>` | `train_lodo.py` → `make_tables.py --runs` |
| VII, VIII | `acc_m2cl_6tap_vs_erm` (+ `vs_m2_6tap`, the new ablation) | `make_tables.py --acc_compare` |
| II | `leak_arch_bias` | `run_leakage.py` (groups `random`, `imagenet`) |
| III | `leak_v1_vs_imagenet` | original 4-tap checkpoints, via the manifest |
| IX | `leak_m2cl6_v1_vs_imagenet`, `leak_new_vs_imagenet` | manifest + `--from_runs` |
| noise floor (Sec. V-A) | `leak_noise_floor` | `run_leakage.py --probe_seeds ...` |
| **new**: where the loss acts | `leak_cl_effect_at_taps(_v1)` | M2-CL vs M2 on each extraction block |
| **new**: sanity check | `leak_class_sanity` | class probe on the same features |

The draft's own numbers can be regenerated right now (no GPU):

```bash
python scripts/make_tables.py --accuracy_csv results/legacy_v1/paper_accuracy_runs.csv --acc_compare m2cl_6tap:erm --out results/tables/legacy_v1
```

## Protocol summary

| | ERM | M2 (ablation) | M2-CL (corrected) |
|---|---|---|---|
| Backbone | ResNet-18, ImageNet init | same | same |
| Extraction taps | – | 6 | 6 |
| α, τ | – | 0, – | 0.003, 1.0 |
| Optimiser | SGD lr 5e-4, momentum 0.9, wd 5e-4, cosine, 30 epochs, batch 128 | same | same |
| Augmentation | RandomResizedCrop(0.7–1), flip, ColorJitter(0.3,0.3,0.3,0.1), grayscale p=0.1 | same | same |
| Model selection | best mean per-domain source-val accuracy (20 % of each source domain) | same | same |
| Target domain | evaluated once, after selection | same | same |

**Leakage probe:** linear layer, Adam 1e-3, 30 epochs, batch 64. It is trained on a stratified 70/30 split of the three source domains (split seed 42, shared by all conditions) and repeated over 5 probe seeds. Comparisons use a paired bootstrap (2,000 resamples) on per-sample correctness averaged over probe seeds, with BH correction inside each table.

## Known deviations from Ballas & Diou (2024)
- 4 or 6 tap locations instead of 13.
- lr 5e-4 instead of 1e-3.
- 3 seeds (5 recommended).

All are listed in the paper's Limitations section.

## Citation

```bibtex
@misc{reddy2026m2clrepro,
  title  = {Multiscale Contrastive Learning for Domain Generalization: A Reproduction Study with a
            Corrected Objective, Multi-Seed Evaluation, and Representation-Level Leakage Analysis},
  author = {Reddy C., Madhusudan},
  year   = {2026},
  note   = {Honours project, IIIT Sri City. Supervisor: Dr. Rajendra Prasath}
}
```

MIT licence (see `LICENSE`).
