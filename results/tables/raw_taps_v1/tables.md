# Generated tables

Do not edit by hand: regenerate with scripts/make_tables.py

## Probe-training noise floor (domain probe, same frozen features, 5 probe seeds)

| group        | layer     |   mean_sd_pp |   max_range_pp |   n |
|:-------------|:----------|-------------:|---------------:|----:|
| m2_4tap_v1   | backbone  |         0.10 |           0.35 |   3 |
| m2_4tap_v1   | eb1       |         0.19 |           0.68 |   3 |
| m2_4tap_v1   | eb2       |         0.15 |           0.38 |   3 |
| m2_4tap_v1   | eb3       |         0.16 |           0.42 |   3 |
| m2_4tap_v1   | eb4       |         0.12 |           0.40 |   3 |
| m2_4tap_v1   | eb_concat |         0.09 |           0.35 |   3 |
| m2cl_4tap_v1 | backbone  |         0.08 |           0.25 |   3 |
| m2cl_4tap_v1 | eb1       |         0.22 |           0.64 |   3 |
| m2cl_4tap_v1 | eb2       |         0.13 |           0.44 |   3 |
| m2cl_4tap_v1 | eb3       |         0.17 |           0.63 |   3 |
| m2cl_4tap_v1 | eb4       |         0.17 |           0.48 |   3 |
| m2cl_4tap_v1 | eb_concat |         0.10 |           0.35 |   3 |

*max_range_pp = largest max-min accuracy spread across probe seeds for any target.*

## Where the loss acts: M2-CL vs M2 (original 4-tap checkpoints), per extraction block

| Domain       | A            | B          |   seed | layer     |   acc A |   acc B |   Delta |   CI lo |   CI hi |    p |   p (BH) | sig (BH, q=.05)   |
|:-------------|:-------------|:-----------|-------:|:----------|--------:|--------:|--------:|--------:|--------:|-----:|---------:|:------------------|
| Art Painting | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb1       |   96.46 |   96.44 |    0.02 |   -0.07 |    0.10 | 0.71 |     0.93 | False             |
| Art Painting | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb2       |   96.58 |   96.58 |    0.00 |   -0.08 |    0.08 | 1.00 |     1.00 | False             |
| Art Painting | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb3       |   96.04 |   96.08 |   -0.04 |   -0.18 |    0.10 | 0.59 |     0.93 | False             |
| Art Painting | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb4       |   94.95 |   94.91 |    0.04 |   -0.13 |    0.23 | 0.70 |     0.93 | False             |
| Art Painting | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb_concat |   99.14 |   99.14 |    0.00 |   -0.08 |    0.08 | 1.00 |     1.00 | False             |
| Art Painting | m2cl_4tap_v1 | m2_4tap_v1 |      0 | backbone  |   99.31 |   99.26 |    0.05 |   -0.01 |    0.13 | 0.15 |     0.81 | False             |
| Cartoon      | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb1       |   90.57 |   90.63 |   -0.06 |   -0.17 |    0.06 | 0.28 |     0.85 | False             |
| Cartoon      | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb2       |   90.54 |   90.50 |    0.03 |   -0.10 |    0.17 | 0.64 |     0.93 | False             |
| Cartoon      | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb3       |   91.38 |   91.50 |   -0.12 |   -0.32 |    0.07 | 0.22 |     0.81 | False             |
| Cartoon      | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb4       |   92.12 |   92.17 |   -0.04 |   -0.24 |    0.14 | 0.64 |     0.93 | False             |
| Cartoon      | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb_concat |   96.29 |   96.44 |   -0.15 |   -0.28 |   -0.02 | 0.03 |     0.25 | False             |
| Cartoon      | m2cl_4tap_v1 | m2_4tap_v1 |      0 | backbone  |   98.50 |   98.54 |   -0.03 |   -0.13 |    0.05 | 0.46 |     0.93 | False             |
| Photo        | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb1       |   94.71 |   94.73 |   -0.02 |   -0.10 |    0.06 | 0.74 |     0.93 | False             |
| Photo        | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb2       |   95.80 |   95.65 |    0.15 |    0.06 |    0.26 | 0.00 |     0.02 | True              |
| Photo        | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb3       |   94.97 |   94.99 |   -0.02 |   -0.16 |    0.14 | 0.82 |     0.93 | False             |
| Photo        | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb4       |   92.82 |   92.74 |    0.08 |   -0.12 |    0.27 | 0.43 |     0.93 | False             |
| Photo        | m2cl_4tap_v1 | m2_4tap_v1 |      0 | eb_concat |   98.31 |   98.35 |   -0.04 |   -0.10 |    0.01 | 0.21 |     0.81 | False             |
| Photo        | m2cl_4tap_v1 | m2_4tap_v1 |      0 | backbone  |   98.61 |   98.62 |   -0.01 |   -0.07 |    0.06 | 0.80 |     0.93 | False             |

*Paired bootstrap (2000 resamples) on per-sample correctness averaged over probe seeds; Benjamini-Hochberg across the 18 comparisons in this table.*

## Sanity check: linear CLASS probe on the same backbone features (%)

| group        | target       |   class acc source-test |   class acc target |
|:-------------|:-------------|------------------------:|-------------------:|
| m2_4tap_v1   | art_painting |                   95.17 |              71.73 |
| m2_4tap_v1   | cartoon      |                   95.60 |              66.13 |
| m2_4tap_v1   | photo        |                   94.31 |              95.93 |
| m2cl_4tap_v1 | art_painting |                   94.84 |              72.41 |
| m2cl_4tap_v1 | cartoon      |                   95.56 |              66.17 |
| m2cl_4tap_v1 | photo        |                   94.11 |              96.05 |

*ImageNet features should give high class accuracy; near-chance (14%) values mean a label bug.*
