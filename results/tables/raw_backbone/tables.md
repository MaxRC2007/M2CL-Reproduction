# Generated tables

Do not edit by hand: regenerate with scripts/make_tables.py

## Probe-training noise floor (domain probe, same frozen features, 5 probe seeds)

| group        | layer    |   mean_sd_pp |   max_range_pp |   n |
|:-------------|:---------|-------------:|---------------:|----:|
| erm_v1       | backbone |         0.15 |           0.88 |   4 |
| imagenet     | backbone |         0.12 |           0.82 |   4 |
| m2_4tap_v1   | backbone |         0.10 |           0.35 |   3 |
| m2cl_4tap_v1 | backbone |         0.08 |           0.25 |   3 |
| m2cl_6tap_v1 | backbone |         0.11 |           0.49 |  12 |
| random       | backbone |         1.60 |          10.50 |  20 |

*max_range_pp = largest max-min accuracy spread across probe seeds for any target.*

## Domain-probe accuracy (%) at the backbone feature: random init vs ImageNet

| Held-out     |   Random-init mean |   Random-init SD |   n init |   Majority |   Arch. bias |   ImageNet |   Pretrain amp. |
|:-------------|-------------------:|-----------------:|---------:|-----------:|-------------:|-----------:|----------------:|
| Art Painting |              92.14 |             0.43 |        5 |      49.48 |        42.67 |      99.61 |            7.47 |
| Cartoon      |              86.07 |             0.38 |        5 |      51.37 |        34.69 |      98.62 |           12.56 |
| Photo        |              89.75 |             0.49 |        5 |      47.22 |        42.53 |      99.13 |            9.39 |
| Sketch       |              71.03 |             0.31 |        5 |      38.65 |        32.38 |      96.11 |           25.08 |

*Each random-init value is first averaged over probe seeds; SD is across weight-initialisation seeds (ddof=1).*

## Draft Table III re-run: original 4-tap checkpoints vs ImageNet-only (backbone)

| Domain       | A            | B        |   seed | layer    |   acc A |   acc B |   Delta |   CI lo |   CI hi |    p |   p (BH) | sig (BH, q=.05)   |
|:-------------|:-------------|:---------|-------:|:---------|--------:|--------:|--------:|--------:|--------:|-----:|---------:|:------------------|
| Art Painting | erm_v1       | imagenet |      0 | backbone |   99.40 |   99.61 |   -0.21 |   -0.50 |    0.07 | 0.13 |     0.22 | False             |
| Art Painting | m2_4tap_v1   | imagenet |      0 | backbone |   99.26 |   99.61 |   -0.35 |   -0.67 |   -0.06 | 0.02 |     0.06 | False             |
| Art Painting | m2cl_4tap_v1 | imagenet |      0 | backbone |   99.31 |   99.61 |   -0.30 |   -0.61 |   -0.01 | 0.05 |     0.09 | False             |
| Cartoon      | erm_v1       | imagenet |      0 | backbone |   98.33 |   98.62 |   -0.30 |   -0.80 |    0.18 | 0.21 |     0.29 | False             |
| Cartoon      | m2_4tap_v1   | imagenet |      0 | backbone |   98.54 |   98.62 |   -0.09 |   -0.55 |    0.40 | 0.73 |     0.73 | False             |
| Cartoon      | m2cl_4tap_v1 | imagenet |      0 | backbone |   98.50 |   98.62 |   -0.12 |   -0.59 |    0.36 | 0.64 |     0.71 | False             |
| Photo        | erm_v1       | imagenet |      0 | backbone |   98.42 |   99.13 |   -0.71 |   -1.16 |   -0.26 | 0.00 |     0.02 | True              |
| Photo        | m2_4tap_v1   | imagenet |      0 | backbone |   98.62 |   99.13 |   -0.51 |   -0.93 |   -0.10 | 0.02 |     0.06 | False             |
| Photo        | m2cl_4tap_v1 | imagenet |      0 | backbone |   98.61 |   99.13 |   -0.52 |   -0.94 |   -0.10 | 0.02 |     0.06 | False             |
| Sketch       | erm_v1       | imagenet |      0 | backbone |   95.62 |   96.11 |   -0.48 |   -1.30 |    0.29 | 0.23 |     0.29 | False             |

*Paired bootstrap (2000 resamples) on per-sample correctness averaged over probe seeds; Benjamini-Hochberg across the 10 comparisons in this table.*

## Draft Table IX re-run: corrected 6-tap M2-CL (3 seeds) vs ImageNet-only (backbone)

| Domain       | A            | B        |   seed | layer    |   acc A |   acc B |   Delta |   CI lo |   CI hi |    p |   p (BH) | sig (BH, q=.05)   |
|:-------------|:-------------|:---------|-------:|:---------|--------:|--------:|--------:|--------:|--------:|-----:|---------:|:------------------|
| Art Painting | m2cl_6tap_v1 | imagenet |      0 | backbone |   99.38 |   99.61 |   -0.23 |   -0.49 |    0.01 | 0.06 |     0.14 | False             |
| Art Painting | m2cl_6tap_v1 | imagenet |      1 | backbone |   99.40 |   99.61 |   -0.22 |   -0.47 |    0.02 | 0.08 |     0.16 | False             |
| Art Painting | m2cl_6tap_v1 | imagenet |      2 | backbone |   99.29 |   99.61 |   -0.33 |   -0.61 |   -0.06 | 0.02 |     0.12 | False             |
| Cartoon      | m2cl_6tap_v1 | imagenet |      0 | backbone |   98.33 |   98.62 |   -0.30 |   -0.79 |    0.21 | 0.23 |     0.26 | False             |
| Cartoon      | m2cl_6tap_v1 | imagenet |      1 | backbone |   98.27 |   98.62 |   -0.35 |   -0.84 |    0.16 | 0.15 |     0.20 | False             |
| Cartoon      | m2cl_6tap_v1 | imagenet |      2 | backbone |   98.33 |   98.62 |   -0.30 |   -0.78 |    0.20 | 0.24 |     0.26 | False             |
| Photo        | m2cl_6tap_v1 | imagenet |      0 | backbone |   98.73 |   99.13 |   -0.40 |   -0.82 |    0.01 | 0.06 |     0.14 | False             |
| Photo        | m2cl_6tap_v1 | imagenet |      1 | backbone |   98.73 |   99.13 |   -0.41 |   -0.78 |   -0.02 | 0.03 |     0.12 | False             |
| Photo        | m2cl_6tap_v1 | imagenet |      2 | backbone |   98.49 |   99.13 |   -0.65 |   -1.07 |   -0.22 | 0.00 |     0.02 | True              |
| Sketch       | m2cl_6tap_v1 | imagenet |      0 | backbone |   95.66 |   96.11 |   -0.45 |   -1.23 |    0.34 | 0.27 |     0.27 | False             |
| Sketch       | m2cl_6tap_v1 | imagenet |      1 | backbone |   95.48 |   96.11 |   -0.63 |   -1.43 |    0.18 | 0.15 |     0.20 | False             |
| Sketch       | m2cl_6tap_v1 | imagenet |      2 | backbone |   95.51 |   96.11 |   -0.59 |   -1.33 |    0.18 | 0.14 |     0.20 | False             |

*Paired bootstrap (2000 resamples) on per-sample correctness averaged over probe seeds; Benjamini-Hochberg across the 12 comparisons in this table.*

## Where the loss acts: M2-CL vs M2 (original 4-tap checkpoints), per extraction block

| Domain       | A            | B          |   seed | layer    |   acc A |   acc B |   Delta |   CI lo |   CI hi |    p |   p (BH) | sig (BH, q=.05)   |
|:-------------|:-------------|:-----------|-------:|:---------|--------:|--------:|--------:|--------:|--------:|-----:|---------:|:------------------|
| Art Painting | m2cl_4tap_v1 | m2_4tap_v1 |      0 | backbone |   99.31 |   99.26 |    0.05 |   -0.01 |    0.13 | 0.15 |     0.44 | False             |
| Cartoon      | m2cl_4tap_v1 | m2_4tap_v1 |      0 | backbone |   98.50 |   98.54 |   -0.03 |   -0.13 |    0.05 | 0.46 |     0.69 | False             |
| Photo        | m2cl_4tap_v1 | m2_4tap_v1 |      0 | backbone |   98.61 |   98.62 |   -0.01 |   -0.07 |    0.06 | 0.80 |     0.80 | False             |

*Paired bootstrap (2000 resamples) on per-sample correctness averaged over probe seeds; Benjamini-Hochberg across the 3 comparisons in this table.*

## Sanity check: linear CLASS probe on the same backbone features (%)

| group        | target       |   class acc source-test |   class acc target |
|:-------------|:-------------|------------------------:|-------------------:|
| erm_v1       | art_painting |                   95.13 |              73.34 |
| erm_v1       | cartoon      |                   95.42 |              70.99 |
| erm_v1       | photo        |                   94.51 |              94.79 |
| erm_v1       | sketch       |                   95.11 |              64.57 |
| imagenet     | art_painting |                   85.40 |              69.09 |
| imagenet     | cartoon      |                   87.23 |              51.07 |
| imagenet     | photo        |                   82.30 |              95.75 |
| imagenet     | sketch       |                   88.95 |              45.18 |
| m2_4tap_v1   | art_painting |                   95.17 |              71.73 |
| m2_4tap_v1   | cartoon      |                   95.60 |              66.13 |
| m2_4tap_v1   | photo        |                   94.31 |              95.93 |
| m2cl_4tap_v1 | art_painting |                   94.84 |              72.41 |
| m2cl_4tap_v1 | cartoon      |                   95.56 |              66.17 |
| m2cl_4tap_v1 | photo        |                   94.11 |              96.05 |
| m2cl_6tap_v1 | art_painting |                   95.24 |              73.57 |
| m2cl_6tap_v1 | cartoon      |                   95.89 |              68.43 |
| m2cl_6tap_v1 | photo        |                   94.30 |              95.05 |
| m2cl_6tap_v1 | sketch       |                   94.72 |              62.45 |
| random       | art_painting |                   44.31 |              25.91 |
| random       | cartoon      |                   38.41 |              30.87 |
| random       | photo        |                   39.35 |              30.08 |
| random       | sketch       |                   40.11 |              13.97 |

*ImageNet features should give high class accuracy; near-chance (14%) values mean a label bug.*
