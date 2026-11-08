# Generated tables

Do not edit by hand: regenerate with scripts/make_tables.py

## Probe-training noise floor (domain probe, same frozen features, 5 probe seeds)

| group    | layer     |   mean_sd_pp |   max_range_pp |   n |
|:---------|:----------|-------------:|---------------:|----:|
| imagenet | backbone  |         0.12 |           0.82 |   4 |
| r3_m2    | backbone  |         0.10 |           0.49 |  12 |
| r3_m2    | eb1       |         0.25 |           1.24 |  12 |
| r3_m2    | eb1a      |         0.30 |           1.53 |  12 |
| r3_m2    | eb2       |         0.19 |           1.10 |  12 |
| r3_m2    | eb2a      |         0.18 |           0.93 |  12 |
| r3_m2    | eb3       |         0.18 |           0.93 |  12 |
| r3_m2    | eb4       |         0.15 |           0.66 |  12 |
| r3_m2    | eb_concat |         0.11 |           0.71 |  12 |
| r3_m2cl  | backbone  |         0.11 |           0.93 |  12 |
| r3_m2cl  | eb1       |         0.22 |           1.04 |  12 |
| r3_m2cl  | eb1a      |         0.28 |           1.74 |  12 |
| r3_m2cl  | eb2       |         0.17 |           1.04 |  12 |
| r3_m2cl  | eb2a      |         0.18 |           0.99 |  12 |
| r3_m2cl  | eb3       |         0.15 |           0.66 |  12 |
| r3_m2cl  | eb4       |         0.31 |           1.51 |  12 |
| r3_m2cl  | eb_concat |         0.11 |           0.77 |  12 |

*max_range_pp = largest max-min accuracy spread across probe seeds for any target.*

## Round 3, active term: M2-CL (alpha 0.07) vs M2, lr 1e-3, matched seeds, per extraction block

| Domain       | A       | B     |   seed | layer     |   acc A |   acc B |   Delta |   CI lo |   CI hi |    p |   p (BH) | sig (BH, q=.05)   |
|:-------------|:--------|:------|-------:|:----------|--------:|--------:|--------:|--------:|--------:|-----:|---------:|:------------------|
| Art Painting | r3_m2cl | r3_m2 |      0 | eb1a      |   97.89 |   97.79 |    0.10 |   -0.05 |    0.29 | 0.23 |     0.44 | False             |
| Art Painting | r3_m2cl | r3_m2 |      0 | eb1       |   96.96 |   97.07 |   -0.11 |   -0.29 |    0.07 | 0.25 |     0.48 | False             |
| Art Painting | r3_m2cl | r3_m2 |      0 | eb2a      |   97.89 |   97.75 |    0.14 |   -0.11 |    0.41 | 0.28 |     0.50 | False             |
| Art Painting | r3_m2cl | r3_m2 |      0 | eb2       |   96.81 |   96.48 |    0.34 |    0.09 |    0.57 | 0.01 |     0.07 | False             |
| Art Painting | r3_m2cl | r3_m2 |      0 | eb3       |   95.92 |   95.35 |    0.57 |    0.00 |    1.15 | 0.05 |     0.17 | False             |
| Art Painting | r3_m2cl | r3_m2 |      0 | eb4       |   78.75 |   94.13 |  -15.38 |  -16.95 |  -13.80 | 0.00 |     0.00 | True              |
| Art Painting | r3_m2cl | r3_m2 |      0 | eb_concat |   99.38 |   99.60 |   -0.22 |   -0.42 |   -0.05 | 0.01 |     0.08 | False             |
| Art Painting | r3_m2cl | r3_m2 |      0 | backbone  |   99.26 |   99.35 |   -0.09 |   -0.24 |    0.05 | 0.20 |     0.41 | False             |
| Art Painting | r3_m2cl | r3_m2 |      1 | eb1a      |   97.44 |   97.21 |    0.23 |    0.08 |    0.40 | 0.00 |     0.01 | True              |
| Art Painting | r3_m2cl | r3_m2 |      1 | eb1       |   97.73 |   97.52 |    0.20 |    0.03 |    0.40 | 0.02 |     0.09 | False             |
| Art Painting | r3_m2cl | r3_m2 |      1 | eb2a      |   98.65 |   98.41 |    0.24 |    0.05 |    0.48 | 0.02 |     0.10 | False             |
| Art Painting | r3_m2cl | r3_m2 |      1 | eb2       |   97.49 |   97.21 |    0.28 |   -0.00 |    0.59 | 0.06 |     0.20 | False             |
| Art Painting | r3_m2cl | r3_m2 |      1 | eb3       |   95.47 |   95.54 |   -0.08 |   -0.65 |    0.52 | 0.78 |     0.89 | False             |
| Art Painting | r3_m2cl | r3_m2 |      1 | eb4       |   77.61 |   94.43 |  -16.82 |  -18.47 |  -15.20 | 0.00 |     0.00 | True              |
| Art Painting | r3_m2cl | r3_m2 |      1 | eb_concat |   99.53 |   99.56 |   -0.03 |   -0.16 |    0.09 | 0.69 |     0.86 | False             |
| Art Painting | r3_m2cl | r3_m2 |      1 | backbone  |   99.23 |   99.25 |   -0.03 |   -0.15 |    0.11 | 0.74 |     0.86 | False             |
| Art Painting | r3_m2cl | r3_m2 |      2 | eb1a      |   97.71 |   97.55 |    0.16 |   -0.07 |    0.38 | 0.15 |     0.37 | False             |
| Art Painting | r3_m2cl | r3_m2 |      2 | eb1       |   97.12 |   96.91 |    0.21 |    0.03 |    0.40 | 0.02 |     0.10 | False             |
| Art Painting | r3_m2cl | r3_m2 |      2 | eb2a      |   98.36 |   98.19 |    0.17 |   -0.03 |    0.38 | 0.10 |     0.31 | False             |
| Art Painting | r3_m2cl | r3_m2 |      2 | eb2       |   97.33 |   97.13 |    0.20 |   -0.05 |    0.46 | 0.13 |     0.34 | False             |
| Art Painting | r3_m2cl | r3_m2 |      2 | eb3       |   93.89 |   93.94 |   -0.05 |   -0.74 |    0.65 | 0.89 |     0.93 | False             |
| Art Painting | r3_m2cl | r3_m2 |      2 | eb4       |   79.01 |   93.55 |  -14.54 |  -16.22 |  -12.84 | 0.00 |     0.00 | True              |
| Art Painting | r3_m2cl | r3_m2 |      2 | eb_concat |   99.56 |   99.61 |   -0.06 |   -0.20 |    0.08 | 0.44 |     0.66 | False             |
| Art Painting | r3_m2cl | r3_m2 |      2 | backbone  |   99.19 |   99.23 |   -0.03 |   -0.21 |    0.13 | 0.74 |     0.86 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      0 | eb1a      |   91.99 |   91.95 |    0.04 |   -0.17 |    0.27 | 0.71 |     0.86 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      0 | eb1       |   90.78 |   90.73 |    0.05 |   -0.18 |    0.28 | 0.71 |     0.86 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      0 | eb2a      |   92.16 |   91.83 |    0.33 |   -0.09 |    0.78 | 0.13 |     0.34 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      0 | eb2       |   90.45 |   90.66 |   -0.21 |   -0.62 |    0.21 | 0.33 |     0.54 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      0 | eb3       |   90.61 |   91.18 |   -0.58 |   -1.30 |    0.16 | 0.11 |     0.33 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      0 | eb4       |   78.26 |   91.76 |  -13.51 |  -15.05 |  -11.87 | 0.00 |     0.00 | True              |
| Cartoon      | r3_m2cl | r3_m2 |      0 | eb_concat |   96.00 |   96.41 |   -0.41 |   -0.97 |    0.17 | 0.17 |     0.40 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      0 | backbone  |   98.34 |   98.47 |   -0.13 |   -0.37 |    0.10 | 0.25 |     0.48 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      1 | eb1a      |   91.44 |   91.44 |   -0.00 |   -0.21 |    0.21 | 0.94 |     0.96 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      1 | eb1       |   90.67 |   90.80 |   -0.13 |   -0.32 |    0.06 | 0.17 |     0.40 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      1 | eb2a      |   91.26 |   91.02 |    0.24 |   -0.08 |    0.56 | 0.14 |     0.36 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      1 | eb2       |   91.90 |   91.98 |   -0.09 |   -0.43 |    0.23 | 0.59 |     0.78 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      1 | eb3       |   92.33 |   92.66 |   -0.33 |   -0.98 |    0.31 | 0.29 |     0.50 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      1 | eb4       |   78.14 |   91.64 |  -13.51 |  -15.02 |  -11.86 | 0.00 |     0.00 | True              |
| Cartoon      | r3_m2cl | r3_m2 |      1 | eb_concat |   96.11 |   96.46 |   -0.35 |   -0.85 |    0.16 | 0.20 |     0.41 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      1 | backbone  |   98.07 |   98.30 |   -0.23 |   -0.45 |   -0.00 | 0.05 |     0.17 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      2 | eb1a      |   91.22 |   91.17 |    0.04 |   -0.25 |    0.35 | 0.74 |     0.86 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      2 | eb1       |   90.68 |   90.53 |    0.15 |   -0.13 |    0.44 | 0.32 |     0.54 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      2 | eb2a      |   90.80 |   90.77 |    0.03 |   -0.37 |    0.45 | 0.90 |     0.93 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      2 | eb2       |   90.17 |   90.34 |   -0.17 |   -0.62 |    0.27 | 0.43 |     0.66 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      2 | eb3       |   90.77 |   90.80 |   -0.03 |   -0.71 |    0.66 | 0.95 |     0.96 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      2 | eb4       |   79.31 |   91.93 |  -12.62 |  -14.15 |  -11.02 | 0.00 |     0.00 | True              |
| Cartoon      | r3_m2cl | r3_m2 |      2 | eb_concat |   96.16 |   96.21 |   -0.05 |   -0.58 |    0.50 | 0.88 |     0.93 | False             |
| Cartoon      | r3_m2cl | r3_m2 |      2 | backbone  |   98.21 |   98.36 |   -0.15 |   -0.37 |    0.06 | 0.19 |     0.41 | False             |
| Photo        | r3_m2cl | r3_m2 |      0 | eb1a      |   95.42 |   95.29 |    0.13 |   -0.08 |    0.35 | 0.27 |     0.49 | False             |
| Photo        | r3_m2cl | r3_m2 |      0 | eb1       |   95.29 |   95.19 |    0.10 |   -0.10 |    0.31 | 0.35 |     0.57 | False             |
| Photo        | r3_m2cl | r3_m2 |      0 | eb2a      |   96.90 |   96.78 |    0.12 |   -0.11 |    0.36 | 0.33 |     0.54 | False             |
| Photo        | r3_m2cl | r3_m2 |      0 | eb2       |   95.76 |   95.71 |    0.05 |   -0.21 |    0.32 | 0.75 |     0.86 | False             |
| Photo        | r3_m2cl | r3_m2 |      0 | eb3       |   94.84 |   94.02 |    0.82 |    0.24 |    1.43 | 0.01 |     0.04 | True              |
| Photo        | r3_m2cl | r3_m2 |      0 | eb4       |   80.72 |   91.63 |  -10.91 |  -12.45 |   -9.37 | 0.00 |     0.00 | True              |
| Photo        | r3_m2cl | r3_m2 |      0 | eb_concat |   98.41 |   98.44 |   -0.02 |   -0.28 |    0.24 | 0.88 |     0.93 | False             |
| Photo        | r3_m2cl | r3_m2 |      0 | backbone  |   98.62 |   98.77 |   -0.15 |   -0.34 |    0.02 | 0.10 |     0.30 | False             |
| Photo        | r3_m2cl | r3_m2 |      1 | eb1a      |   94.96 |   94.67 |    0.29 |    0.06 |    0.51 | 0.02 |     0.09 | False             |
| Photo        | r3_m2cl | r3_m2 |      1 | eb1       |   94.83 |   94.79 |    0.05 |   -0.18 |    0.26 | 0.71 |     0.86 | False             |
| Photo        | r3_m2cl | r3_m2 |      1 | eb2a      |   96.41 |   96.33 |    0.08 |   -0.18 |    0.35 | 0.55 |     0.77 | False             |
| Photo        | r3_m2cl | r3_m2 |      1 | eb2       |   95.47 |   95.43 |    0.03 |   -0.26 |    0.32 | 0.87 |     0.93 | False             |
| Photo        | r3_m2cl | r3_m2 |      1 | eb3       |   93.63 |   93.54 |    0.09 |   -0.55 |    0.72 | 0.79 |     0.89 | False             |
| Photo        | r3_m2cl | r3_m2 |      1 | eb4       |   75.14 |   92.14 |  -17.00 |  -18.61 |  -15.37 | 0.00 |     0.00 | True              |
| Photo        | r3_m2cl | r3_m2 |      1 | eb_concat |   98.13 |   98.39 |   -0.26 |   -0.54 |    0.02 | 0.08 |     0.25 | False             |
| Photo        | r3_m2cl | r3_m2 |      1 | backbone  |   98.50 |   98.66 |   -0.16 |   -0.38 |    0.06 | 0.13 |     0.34 | False             |
| Photo        | r3_m2cl | r3_m2 |      2 | eb1a      |   95.15 |   95.02 |    0.14 |   -0.06 |    0.35 | 0.22 |     0.43 | False             |
| Photo        | r3_m2cl | r3_m2 |      2 | eb1       |   95.59 |   95.53 |    0.06 |   -0.14 |    0.25 | 0.55 |     0.77 | False             |
| Photo        | r3_m2cl | r3_m2 |      2 | eb2a      |   96.27 |   96.32 |   -0.05 |   -0.30 |    0.18 | 0.68 |     0.86 | False             |
| Photo        | r3_m2cl | r3_m2 |      2 | eb2       |   95.90 |   95.46 |    0.44 |    0.06 |    0.85 | 0.02 |     0.09 | False             |
| Photo        | r3_m2cl | r3_m2 |      2 | eb3       |   94.15 |   94.20 |   -0.05 |   -0.54 |    0.44 | 0.87 |     0.93 | False             |
| Photo        | r3_m2cl | r3_m2 |      2 | eb4       |   74.06 |   91.30 |  -17.24 |  -18.89 |  -15.55 | 0.00 |     0.00 | True              |
| Photo        | r3_m2cl | r3_m2 |      2 | eb_concat |   98.01 |   98.23 |   -0.22 |   -0.52 |    0.06 | 0.12 |     0.33 | False             |
| Photo        | r3_m2cl | r3_m2 |      2 | backbone  |   98.29 |   98.39 |   -0.10 |   -0.32 |    0.12 | 0.41 |     0.64 | False             |
| Sketch       | r3_m2cl | r3_m2 |      0 | eb1a      |   81.29 |   81.39 |   -0.10 |   -0.42 |    0.21 | 0.56 |     0.77 | False             |
| Sketch       | r3_m2cl | r3_m2 |      0 | eb1       |   82.50 |   82.40 |    0.10 |   -0.16 |    0.37 | 0.49 |     0.72 | False             |
| Sketch       | r3_m2cl | r3_m2 |      0 | eb2a      |   84.86 |   84.97 |   -0.11 |   -0.51 |    0.30 | 0.60 |     0.78 | False             |
| Sketch       | r3_m2cl | r3_m2 |      0 | eb2       |   80.87 |   80.90 |   -0.03 |   -0.42 |    0.35 | 0.84 |     0.92 | False             |
| Sketch       | r3_m2cl | r3_m2 |      0 | eb3       |   82.52 |   82.09 |    0.43 |   -0.23 |    1.02 | 0.20 |     0.41 | False             |
| Sketch       | r3_m2cl | r3_m2 |      0 | eb4       |   73.73 |   82.35 |   -8.62 |  -10.65 |   -6.63 | 0.00 |     0.00 | True              |
| Sketch       | r3_m2cl | r3_m2 |      0 | eb_concat |   92.64 |   93.53 |   -0.89 |   -1.58 |   -0.23 | 0.01 |     0.04 | True              |
| Sketch       | r3_m2cl | r3_m2 |      0 | backbone  |   95.78 |   95.92 |   -0.14 |   -0.41 |    0.12 | 0.30 |     0.52 | False             |
| Sketch       | r3_m2cl | r3_m2 |      1 | eb1a      |   81.52 |   81.45 |    0.07 |   -0.15 |    0.31 | 0.59 |     0.78 | False             |
| Sketch       | r3_m2cl | r3_m2 |      1 | eb1       |   81.57 |   81.85 |   -0.27 |   -0.57 |   -0.02 | 0.03 |     0.12 | False             |
| Sketch       | r3_m2cl | r3_m2 |      1 | eb2a      |   84.11 |   84.01 |    0.10 |   -0.15 |    0.36 | 0.49 |     0.72 | False             |
| Sketch       | r3_m2cl | r3_m2 |      1 | eb2       |   83.67 |   83.67 |    0.00 |   -0.32 |    0.31 | 0.99 |     0.99 | False             |
| Sketch       | r3_m2cl | r3_m2 |      1 | eb3       |   83.33 |   83.46 |   -0.13 |   -0.60 |    0.33 | 0.58 |     0.78 | False             |
| Sketch       | r3_m2cl | r3_m2 |      1 | eb4       |   76.15 |   83.33 |   -7.18 |   -8.93 |   -5.50 | 0.00 |     0.00 | True              |
| Sketch       | r3_m2cl | r3_m2 |      1 | eb_concat |   92.70 |   93.08 |   -0.38 |   -0.95 |    0.14 | 0.15 |     0.37 | False             |
| Sketch       | r3_m2cl | r3_m2 |      1 | backbone  |   95.43 |   95.51 |   -0.09 |   -0.37 |    0.18 | 0.55 |     0.77 | False             |
| Sketch       | r3_m2cl | r3_m2 |      2 | eb1a      |   81.22 |   80.90 |    0.32 |    0.00 |    0.65 | 0.05 |     0.17 | False             |
| Sketch       | r3_m2cl | r3_m2 |      2 | eb1       |   81.96 |   81.86 |    0.10 |   -0.15 |    0.35 | 0.41 |     0.64 | False             |
| Sketch       | r3_m2cl | r3_m2 |      2 | eb2a      |   85.16 |   85.31 |   -0.15 |   -0.43 |    0.12 | 0.29 |     0.50 | False             |
| Sketch       | r3_m2cl | r3_m2 |      2 | eb2       |   80.32 |   80.26 |    0.05 |   -0.36 |    0.48 | 0.80 |     0.89 | False             |
| Sketch       | r3_m2cl | r3_m2 |      2 | eb3       |   83.33 |   82.76 |    0.57 |    0.02 |    1.14 | 0.04 |     0.17 | False             |
| Sketch       | r3_m2cl | r3_m2 |      2 | eb4       |   70.27 |   81.65 |  -11.38 |  -13.27 |   -9.39 | 0.00 |     0.00 | True              |
| Sketch       | r3_m2cl | r3_m2 |      2 | eb_concat |   92.72 |   93.44 |   -0.71 |   -1.35 |   -0.05 | 0.03 |     0.14 | False             |
| Sketch       | r3_m2cl | r3_m2 |      2 | backbone  |   95.47 |   95.67 |   -0.20 |   -0.48 |    0.09 | 0.18 |     0.40 | False             |

*Paired bootstrap (2000 resamples) on per-sample correctness averaged over probe seeds; Benjamini-Hochberg across the 96 comparisons in this table.*

## Round 3 (lr 1e-3): ERM / M2 / M2-CL vs ImageNet-only (backbone)

| Domain       | A       | B        |   seed | layer    |   acc A |   acc B |   Delta |   CI lo |   CI hi |    p |   p (BH) | sig (BH, q=.05)   |
|:-------------|:--------|:---------|-------:|:---------|--------:|--------:|--------:|--------:|--------:|-----:|---------:|:------------------|
| Art Painting | r3_m2   | imagenet |      0 | backbone |   99.35 |   99.61 |   -0.26 |   -0.54 |    0.01 | 0.06 |     0.11 | False             |
| Art Painting | r3_m2   | imagenet |      1 | backbone |   99.25 |   99.61 |   -0.36 |   -0.65 |   -0.08 | 0.01 |     0.03 | True              |
| Art Painting | r3_m2   | imagenet |      2 | backbone |   99.23 |   99.61 |   -0.39 |   -0.69 |   -0.09 | 0.01 |     0.03 | True              |
| Art Painting | r3_m2cl | imagenet |      0 | backbone |   99.26 |   99.61 |   -0.35 |   -0.67 |   -0.07 | 0.02 |     0.05 | False             |
| Art Painting | r3_m2cl | imagenet |      1 | backbone |   99.23 |   99.61 |   -0.39 |   -0.67 |   -0.10 | 0.01 |     0.02 | True              |
| Art Painting | r3_m2cl | imagenet |      2 | backbone |   99.19 |   99.61 |   -0.42 |   -0.74 |   -0.13 | 0.00 |     0.02 | True              |
| Cartoon      | r3_m2   | imagenet |      0 | backbone |   98.47 |   98.62 |   -0.16 |   -0.67 |    0.32 | 0.54 |     0.57 | False             |
| Cartoon      | r3_m2   | imagenet |      1 | backbone |   98.30 |   98.62 |   -0.32 |   -0.80 |    0.16 | 0.20 |     0.26 | False             |
| Cartoon      | r3_m2   | imagenet |      2 | backbone |   98.36 |   98.62 |   -0.26 |   -0.78 |    0.24 | 0.30 |     0.35 | False             |
| Cartoon      | r3_m2cl | imagenet |      0 | backbone |   98.34 |   98.62 |   -0.29 |   -0.83 |    0.26 | 0.28 |     0.33 | False             |
| Cartoon      | r3_m2cl | imagenet |      1 | backbone |   98.07 |   98.62 |   -0.55 |   -1.05 |   -0.04 | 0.03 |     0.06 | False             |
| Cartoon      | r3_m2cl | imagenet |      2 | backbone |   98.21 |   98.62 |   -0.41 |   -0.91 |    0.10 | 0.12 |     0.19 | False             |
| Photo        | r3_m2   | imagenet |      0 | backbone |   98.77 |   99.13 |   -0.36 |   -0.76 |    0.04 | 0.08 |     0.15 | False             |
| Photo        | r3_m2   | imagenet |      1 | backbone |   98.66 |   99.13 |   -0.47 |   -0.88 |   -0.08 | 0.02 |     0.05 | False             |
| Photo        | r3_m2   | imagenet |      2 | backbone |   98.39 |   99.13 |   -0.74 |   -1.19 |   -0.31 | 0.00 |     0.01 | True              |
| Photo        | r3_m2cl | imagenet |      0 | backbone |   98.62 |   99.13 |   -0.51 |   -0.91 |   -0.12 | 0.01 |     0.03 | True              |
| Photo        | r3_m2cl | imagenet |      1 | backbone |   98.50 |   99.13 |   -0.63 |   -1.05 |   -0.22 | 0.01 |     0.02 | True              |
| Photo        | r3_m2cl | imagenet |      2 | backbone |   98.29 |   99.13 |   -0.84 |   -1.31 |   -0.39 | 0.00 |     0.00 | True              |
| Sketch       | r3_m2   | imagenet |      0 | backbone |   95.92 |   96.11 |   -0.19 |   -0.97 |    0.62 | 0.67 |     0.67 | False             |
| Sketch       | r3_m2   | imagenet |      1 | backbone |   95.51 |   96.11 |   -0.59 |   -1.41 |    0.26 | 0.19 |     0.26 | False             |
| Sketch       | r3_m2   | imagenet |      2 | backbone |   95.67 |   96.11 |   -0.44 |   -1.19 |    0.35 | 0.27 |     0.33 | False             |
| Sketch       | r3_m2cl | imagenet |      0 | backbone |   95.78 |   96.11 |   -0.33 |   -1.15 |    0.49 | 0.46 |     0.50 | False             |
| Sketch       | r3_m2cl | imagenet |      1 | backbone |   95.43 |   96.11 |   -0.68 |   -1.51 |    0.19 | 0.13 |     0.20 | False             |
| Sketch       | r3_m2cl | imagenet |      2 | backbone |   95.47 |   96.11 |   -0.64 |   -1.45 |    0.14 | 0.11 |     0.19 | False             |

*Paired bootstrap (2000 resamples) on per-sample correctness averaged over probe seeds; Benjamini-Hochberg across the 24 comparisons in this table.*

## Sanity check: linear CLASS probe on the same backbone features (%)

| group    | target       |   class acc source-test |   class acc target |
|:---------|:-------------|------------------------:|-------------------:|
| imagenet | art_painting |                   85.40 |              69.09 |
| imagenet | cartoon      |                   87.23 |              51.07 |
| imagenet | photo        |                   82.30 |              95.75 |
| imagenet | sketch       |                   88.95 |              45.18 |
| r3_m2    | art_painting |                   97.44 |              74.30 |
| r3_m2    | cartoon      |                   97.55 |              70.31 |
| r3_m2    | photo        |                   96.85 |              95.97 |
| r3_m2    | sketch       |                   97.01 |              64.31 |
| r3_m2cl  | art_painting |                   97.59 |              75.10 |
| r3_m2cl  | cartoon      |                   97.73 |              69.84 |
| r3_m2cl  | photo        |                   96.93 |              96.09 |
| r3_m2cl  | sketch       |                   97.05 |              65.41 |

*ImageNet features should give high class accuracy; near-chance (14%) values mean a label bug.*
