# Generated tables

Do not edit by hand: regenerate with scripts/make_tables.py

## Probe-training noise floor (domain probe, same frozen features, 5 probe seeds)

| group    | layer    |   mean_sd_pp |   max_range_pp |   n |
|:---------|:---------|-------------:|---------------:|----:|
| imagenet | backbone |         0.12 |           0.82 |   4 |
| r3_erm   | backbone |         0.12 |           0.49 |  12 |

*max_range_pp = largest max-min accuracy spread across probe seeds for any target.*

## Round 3 (lr 1e-3): ERM / M2 / M2-CL vs ImageNet-only (backbone)

| Domain       | A      | B        |   seed | layer    |   acc A |   acc B |   Delta |   CI lo |   CI hi |    p |   p (BH) | sig (BH, q=.05)   |
|:-------------|:-------|:---------|-------:|:---------|--------:|--------:|--------:|--------:|--------:|-----:|---------:|:------------------|
| Art Painting | r3_erm | imagenet |      0 | backbone |   99.45 |   99.61 |   -0.16 |   -0.45 |    0.12 | 0.26 |     0.32 | False             |
| Art Painting | r3_erm | imagenet |      1 | backbone |   99.33 |   99.61 |   -0.29 |   -0.59 |    0.00 | 0.05 |     0.16 | False             |
| Art Painting | r3_erm | imagenet |      2 | backbone |   99.42 |   99.61 |   -0.19 |   -0.46 |    0.04 | 0.12 |     0.20 | False             |
| Cartoon      | r3_erm | imagenet |      0 | backbone |   98.27 |   98.62 |   -0.36 |   -0.88 |    0.13 | 0.16 |     0.24 | False             |
| Cartoon      | r3_erm | imagenet |      1 | backbone |   98.42 |   98.62 |   -0.20 |   -0.69 |    0.30 | 0.41 |     0.44 | False             |
| Cartoon      | r3_erm | imagenet |      2 | backbone |   98.21 |   98.62 |   -0.41 |   -0.92 |    0.07 | 0.10 |     0.20 | False             |
| Photo        | r3_erm | imagenet |      0 | backbone |   98.41 |   99.13 |   -0.73 |   -1.19 |   -0.30 | 0.00 |     0.01 | True              |
| Photo        | r3_erm | imagenet |      1 | backbone |   98.53 |   99.13 |   -0.60 |   -1.01 |   -0.18 | 0.01 |     0.02 | True              |
| Photo        | r3_erm | imagenet |      2 | backbone |   98.55 |   99.13 |   -0.58 |   -0.98 |   -0.18 | 0.01 |     0.02 | True              |
| Sketch       | r3_erm | imagenet |      0 | backbone |   95.63 |   96.11 |   -0.47 |   -1.28 |    0.32 | 0.25 |     0.32 | False             |
| Sketch       | r3_erm | imagenet |      1 | backbone |   95.35 |   96.11 |   -0.76 |   -1.64 |    0.12 | 0.09 |     0.20 | False             |
| Sketch       | r3_erm | imagenet |      2 | backbone |   96.00 |   96.11 |   -0.11 |   -0.90 |    0.71 | 0.81 |     0.81 | False             |

*Paired bootstrap (2000 resamples) on per-sample correctness averaged over probe seeds; Benjamini-Hochberg across the 12 comparisons in this table.*

## Sanity check: linear CLASS probe on the same backbone features (%)

| group    | target       |   class acc source-test |   class acc target |
|:---------|:-------------|------------------------:|-------------------:|
| imagenet | art_painting |                   85.40 |              69.09 |
| imagenet | cartoon      |                   87.23 |              51.07 |
| imagenet | photo        |                   82.30 |              95.75 |
| imagenet | sketch       |                   88.95 |              45.18 |
| r3_erm   | art_painting |                   97.44 |              73.62 |
| r3_erm   | cartoon      |                   97.55 |              69.31 |
| r3_erm   | photo        |                   97.18 |              95.93 |
| r3_erm   | sketch       |                   97.14 |              62.99 |

*ImageNet features should give high class accuracy; near-chance (14%) values mean a label bug.*
