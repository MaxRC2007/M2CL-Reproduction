# Generated tables

Do not edit by hand: regenerate with scripts/make_tables.py

## Probe-training noise floor (domain probe, same frozen features, 5 probe seeds)

| group    | layer    |   mean_sd_pp |   max_range_pp |   n |
|:---------|:---------|-------------:|---------------:|----:|
| imagenet | backbone |         0.24 |           0.98 |   4 |
| r3v_erm  | backbone |         0.22 |           1.22 |  12 |

*max_range_pp = largest max-min accuracy spread across probe seeds for any target.*

## VLCS round 3 (lr 1e-3): ERM / M2 / M2-CL vs ImageNet-only (backbone)

| Domain     | A       | B        |   seed | layer    |   acc A |   acc B |   Delta |   CI lo |   CI hi |    p |   p (BH) | sig (BH, q=.05)   |
|:-----------|:--------|:---------|-------:|:---------|--------:|--------:|--------:|--------:|--------:|-----:|---------:|:------------------|
| Caltech101 | r3v_erm | imagenet |      0 | backbone |   82.58 |   82.78 |   -0.20 |   -1.17 |    0.72 | 0.64 |     0.86 | False             |
| Caltech101 | r3v_erm | imagenet |      1 | backbone |   83.33 |   82.78 |    0.55 |   -0.44 |    1.42 | 0.28 |     0.60 | False             |
| Caltech101 | r3v_erm | imagenet |      2 | backbone |   82.68 |   82.78 |   -0.10 |   -1.03 |    0.81 | 0.83 |     0.91 | False             |
| LabelMe    | r3v_erm | imagenet |      0 | backbone |   88.60 |   88.55 |    0.04 |   -0.69 |    0.80 | 0.96 |     0.96 | False             |
| LabelMe    | r3v_erm | imagenet |      1 | backbone |   88.76 |   88.55 |    0.21 |   -0.57 |    0.93 | 0.59 |     0.86 | False             |
| LabelMe    | r3v_erm | imagenet |      2 | backbone |   88.65 |   88.55 |    0.09 |   -0.63 |    0.78 | 0.83 |     0.91 | False             |
| SUN09      | r3v_erm | imagenet |      0 | backbone |   92.36 |   92.65 |   -0.30 |   -0.98 |    0.37 | 0.37 |     0.63 | False             |
| SUN09      | r3v_erm | imagenet |      1 | backbone |   92.31 |   92.65 |   -0.34 |   -0.92 |    0.27 | 0.25 |     0.60 | False             |
| SUN09      | r3v_erm | imagenet |      2 | backbone |   92.35 |   92.65 |   -0.30 |   -0.94 |    0.32 | 0.30 |     0.60 | False             |
| VOC2007    | r3v_erm | imagenet |      0 | backbone |   92.51 |   91.70 |    0.81 |    0.02 |    1.59 | 0.05 |     0.19 | False             |
| VOC2007    | r3v_erm | imagenet |      1 | backbone |   92.77 |   91.70 |    1.06 |    0.22 |    1.89 | 0.01 |     0.12 | False             |
| VOC2007    | r3v_erm | imagenet |      2 | backbone |   92.55 |   91.70 |    0.84 |    0.06 |    1.64 | 0.03 |     0.19 | False             |

*Paired bootstrap (2000 resamples) on per-sample correctness averaged over probe seeds; Benjamini-Hochberg across the 12 comparisons in this table.*

## Sanity check: linear CLASS probe on the same backbone features (%)

| group    | target     |   class acc source-test |   class acc target |
|:---------|:-----------|------------------------:|-------------------:|
| imagenet | Caltech101 |                   78.25 |              95.83 |
| imagenet | LabelMe    |                   83.44 |              59.90 |
| imagenet | SUN09      |                   83.49 |              69.84 |
| imagenet | VOC2007    |                   81.46 |              74.56 |
| r3v_erm  | Caltech101 |                   84.46 |              94.28 |
| r3v_erm  | LabelMe    |                   88.36 |              61.65 |
| r3v_erm  | SUN09      |                   86.98 |              69.87 |
| r3v_erm  | VOC2007    |                   85.65 |              69.80 |

*ImageNet features should give high class accuracy; near-chance (14%) values mean a label bug.*
