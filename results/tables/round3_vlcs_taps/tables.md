# Generated tables

Do not edit by hand: regenerate with scripts/make_tables.py

## Probe-training noise floor (domain probe, same frozen features, 5 probe seeds)

| group    | layer     |   mean_sd_pp |   max_range_pp |   n |
|:---------|:----------|-------------:|---------------:|----:|
| imagenet | backbone  |         0.24 |           0.98 |   4 |
| r3v_m2   | backbone  |         0.25 |           1.03 |  12 |
| r3v_m2   | eb1       |         0.47 |           2.23 |  12 |
| r3v_m2   | eb1a      |         0.40 |           2.56 |  12 |
| r3v_m2   | eb2       |         0.32 |           1.32 |  12 |
| r3v_m2   | eb2a      |         0.33 |           1.82 |  12 |
| r3v_m2   | eb3       |         0.21 |           1.04 |  12 |
| r3v_m2   | eb4       |         0.15 |           0.68 |  12 |
| r3v_m2   | eb_concat |         0.35 |           1.90 |  12 |
| r3v_m2cl | backbone  |         0.28 |           1.11 |  12 |
| r3v_m2cl | eb1       |         0.40 |           1.69 |  12 |
| r3v_m2cl | eb1a      |         0.39 |           3.10 |  12 |
| r3v_m2cl | eb2       |         0.34 |           1.31 |  12 |
| r3v_m2cl | eb2a      |         0.49 |           2.48 |  12 |
| r3v_m2cl | eb3       |         0.24 |           1.00 |  12 |
| r3v_m2cl | eb4       |         0.24 |           0.82 |  12 |
| r3v_m2cl | eb_concat |         0.46 |           2.48 |  12 |

*max_range_pp = largest max-min accuracy spread across probe seeds for any target.*

## VLCS round 3, active term: M2-CL (alpha 0.07) vs M2, lr 1e-3, matched seeds, per extraction block

| Domain     | A        | B      |   seed | layer     |   acc A |   acc B |   Delta |   CI lo |   CI hi |    p |   p (BH) | sig (BH, q=.05)   |
|:-----------|:---------|:-------|-------:|:----------|--------:|--------:|--------:|--------:|--------:|-----:|---------:|:------------------|
| Caltech101 | r3v_m2cl | r3v_m2 |      0 | eb1a      |   67.88 |   67.81 |    0.06 |   -0.35 |    0.47 | 0.74 |     0.90 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      0 | eb1       |   65.14 |   65.14 |    0.00 |   -0.35 |    0.38 | 0.97 |     0.98 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      0 | eb2a      |   65.77 |   65.85 |   -0.07 |   -0.63 |    0.49 | 0.82 |     0.92 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      0 | eb2       |   61.75 |   62.05 |   -0.31 |   -0.92 |    0.30 | 0.29 |     0.59 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      0 | eb3       |   68.10 |   66.99 |    1.11 |   -0.05 |    2.33 | 0.06 |     0.21 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      0 | eb4       |   67.38 |   74.36 |   -6.98 |   -8.64 |   -5.37 | 0.00 |     0.00 | True              |
| Caltech101 | r3v_m2cl | r3v_m2 |      0 | eb_concat |   78.20 |   79.81 |   -1.62 |   -2.58 |   -0.64 | 0.00 |     0.01 | True              |
| Caltech101 | r3v_m2cl | r3v_m2 |      0 | backbone  |   82.63 |   82.63 |    0.01 |   -0.30 |    0.31 | 0.99 |     0.99 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      1 | eb1a      |   65.17 |   65.27 |   -0.10 |   -0.50 |    0.29 | 0.63 |     0.86 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      1 | eb1       |   63.28 |   63.35 |   -0.07 |   -0.45 |    0.29 | 0.71 |     0.90 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      1 | eb2a      |   65.27 |   65.55 |   -0.28 |   -0.84 |    0.30 | 0.34 |     0.65 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      1 | eb2       |   65.04 |   65.36 |   -0.31 |   -0.90 |    0.27 | 0.29 |     0.59 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      1 | eb3       |   66.96 |   66.92 |    0.04 |   -1.09 |    1.19 | 0.97 |     0.98 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      1 | eb4       |   68.92 |   73.90 |   -4.98 |   -6.60 |   -3.35 | 0.00 |     0.00 | True              |
| Caltech101 | r3v_m2cl | r3v_m2 |      1 | eb_concat |   78.41 |   79.74 |   -1.32 |   -2.17 |   -0.45 | 0.00 |     0.01 | True              |
| Caltech101 | r3v_m2cl | r3v_m2 |      1 | backbone  |   82.90 |   82.93 |   -0.04 |   -0.37 |    0.30 | 0.84 |     0.92 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      2 | eb1a      |   66.38 |   66.30 |    0.08 |   -0.39 |    0.57 | 0.73 |     0.90 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      2 | eb1       |   64.19 |   64.21 |   -0.03 |   -0.47 |    0.39 | 0.93 |     0.97 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      2 | eb2a      |   66.36 |   66.84 |   -0.48 |   -1.04 |    0.09 | 0.10 |     0.26 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      2 | eb2       |   62.70 |   61.90 |    0.80 |    0.04 |    1.52 | 0.04 |     0.18 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      2 | eb3       |   66.95 |   67.30 |   -0.35 |   -1.63 |    0.93 | 0.58 |     0.86 | False             |
| Caltech101 | r3v_m2cl | r3v_m2 |      2 | eb4       |   66.35 |   73.63 |   -7.28 |   -9.04 |   -5.57 | 0.00 |     0.00 | True              |
| Caltech101 | r3v_m2cl | r3v_m2 |      2 | eb_concat |   78.59 |   80.84 |   -2.25 |   -3.32 |   -1.23 | 0.00 |     0.00 | True              |
| Caltech101 | r3v_m2cl | r3v_m2 |      2 | backbone  |   82.86 |   82.65 |    0.21 |   -0.21 |    0.64 | 0.34 |     0.65 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      0 | eb1a      |   72.64 |   72.75 |   -0.11 |   -0.46 |    0.24 | 0.57 |     0.85 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      0 | eb1       |   69.49 |   69.48 |    0.01 |   -0.41 |    0.44 | 0.96 |     0.98 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      0 | eb2a      |   73.95 |   73.65 |    0.30 |   -0.26 |    0.87 | 0.30 |     0.59 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      0 | eb2       |   72.25 |   72.33 |   -0.07 |   -0.64 |    0.53 | 0.81 |     0.92 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      0 | eb3       |   73.55 |   73.31 |    0.24 |   -0.83 |    1.39 | 0.68 |     0.88 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      0 | eb4       |   75.48 |   81.54 |   -6.06 |   -7.75 |   -4.47 | 0.00 |     0.00 | True              |
| LabelMe    | r3v_m2cl | r3v_m2 |      0 | eb_concat |   83.29 |   83.77 |   -0.48 |   -1.23 |    0.32 | 0.22 |     0.52 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      0 | backbone  |   88.63 |   88.50 |    0.13 |   -0.16 |    0.43 | 0.38 |     0.68 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      1 | eb1a      |   70.69 |   70.45 |    0.24 |   -0.13 |    0.60 | 0.23 |     0.52 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      1 | eb1       |   68.90 |   69.03 |   -0.12 |   -0.61 |    0.36 | 0.65 |     0.86 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      1 | eb2a      |   72.82 |   72.69 |    0.12 |   -0.36 |    0.59 | 0.62 |     0.86 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      1 | eb2       |   71.92 |   72.39 |   -0.46 |   -1.00 |    0.02 | 0.06 |     0.21 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      1 | eb3       |   75.71 |   74.58 |    1.12 |    0.15 |    2.06 | 0.03 |     0.15 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      1 | eb4       |   78.13 |   79.53 |   -1.40 |   -2.90 |    0.11 | 0.06 |     0.21 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      1 | eb_concat |   83.24 |   83.38 |   -0.14 |   -0.95 |    0.69 | 0.75 |     0.90 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      1 | backbone  |   88.86 |   88.93 |   -0.07 |   -0.36 |    0.22 | 0.66 |     0.87 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      2 | eb1a      |   72.12 |   72.44 |   -0.31 |   -0.77 |    0.15 | 0.20 |     0.51 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      2 | eb1       |   69.04 |   69.09 |   -0.05 |   -0.50 |    0.40 | 0.82 |     0.92 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      2 | eb2a      |   73.58 |   73.73 |   -0.15 |   -0.74 |    0.45 | 0.61 |     0.86 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      2 | eb2       |   70.36 |   70.93 |   -0.58 |   -1.20 |    0.09 | 0.09 |     0.25 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      2 | eb3       |   74.00 |   73.94 |    0.06 |   -0.97 |    1.05 | 0.92 |     0.97 | False             |
| LabelMe    | r3v_m2cl | r3v_m2 |      2 | eb4       |   73.15 |   80.83 |   -7.67 |   -9.41 |   -5.97 | 0.00 |     0.00 | True              |
| LabelMe    | r3v_m2cl | r3v_m2 |      2 | eb_concat |   82.63 |   84.13 |   -1.50 |   -2.43 |   -0.64 | 0.00 |     0.01 | True              |
| LabelMe    | r3v_m2cl | r3v_m2 |      2 | backbone  |   88.87 |   88.91 |   -0.04 |   -0.42 |    0.33 | 0.85 |     0.92 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      0 | eb1a      |   83.66 |   83.40 |    0.26 |   -0.03 |    0.53 | 0.08 |     0.25 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      0 | eb1       |   78.60 |   78.68 |   -0.08 |   -0.37 |    0.21 | 0.61 |     0.86 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      0 | eb2a      |   81.14 |   80.95 |    0.19 |   -0.31 |    0.68 | 0.47 |     0.76 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      0 | eb2       |   78.77 |   78.33 |    0.45 |   -0.06 |    0.99 | 0.09 |     0.25 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      0 | eb3       |   80.60 |   80.25 |    0.35 |   -0.44 |    1.21 | 0.41 |     0.68 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      0 | eb4       |   83.11 |   87.87 |   -4.76 |   -6.17 |   -3.42 | 0.00 |     0.00 | True              |
| SUN09      | r3v_m2cl | r3v_m2 |      0 | eb_concat |   91.20 |   92.39 |   -1.19 |   -1.90 |   -0.53 | 0.00 |     0.01 | True              |
| SUN09      | r3v_m2cl | r3v_m2 |      0 | backbone  |   92.38 |   92.34 |    0.04 |   -0.24 |    0.31 | 0.80 |     0.92 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      1 | eb1a      |   82.28 |   81.86 |    0.42 |    0.04 |    0.80 | 0.03 |     0.15 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      1 | eb1       |   79.87 |   79.48 |    0.38 |   -0.07 |    0.81 | 0.09 |     0.25 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      1 | eb2a      |   82.34 |   82.23 |    0.12 |   -0.38 |    0.60 | 0.64 |     0.86 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      1 | eb2       |   82.24 |   82.36 |   -0.13 |   -0.69 |    0.40 | 0.64 |     0.86 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      1 | eb3       |   81.50 |   80.44 |    1.06 |    0.13 |    1.99 | 0.02 |     0.12 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      1 | eb4       |   82.57 |   87.28 |   -4.71 |   -6.26 |   -3.22 | 0.00 |     0.00 | True              |
| SUN09      | r3v_m2cl | r3v_m2 |      1 | eb_concat |   92.06 |   92.36 |   -0.30 |   -0.96 |    0.38 | 0.39 |     0.68 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      1 | backbone  |   92.60 |   92.33 |    0.27 |   -0.02 |    0.56 | 0.07 |     0.23 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      2 | eb1a      |   82.66 |   82.59 |    0.07 |   -0.40 |    0.53 | 0.76 |     0.90 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      2 | eb1       |   78.82 |   79.11 |   -0.29 |   -0.80 |    0.19 | 0.23 |     0.52 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      2 | eb2a      |   82.16 |   82.08 |    0.08 |   -0.48 |    0.70 | 0.78 |     0.91 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      2 | eb2       |   80.38 |   80.18 |    0.21 |   -0.40 |    0.84 | 0.52 |     0.82 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      2 | eb3       |   82.71 |   83.11 |   -0.39 |   -1.16 |    0.49 | 0.40 |     0.68 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      2 | eb4       |   85.71 |   86.73 |   -1.02 |   -2.34 |    0.22 | 0.10 |     0.28 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      2 | eb_concat |   92.78 |   93.08 |   -0.30 |   -0.83 |    0.23 | 0.27 |     0.59 | False             |
| SUN09      | r3v_m2cl | r3v_m2 |      2 | backbone  |   92.25 |   92.41 |   -0.16 |   -0.50 |    0.17 | 0.36 |     0.67 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      0 | eb1a      |   77.02 |   77.40 |   -0.38 |   -0.95 |    0.17 | 0.18 |     0.46 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      0 | eb1       |   75.93 |   75.98 |   -0.05 |   -0.59 |    0.49 | 0.85 |     0.92 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      0 | eb2a      |   77.90 |   77.92 |   -0.03 |   -0.62 |    0.54 | 0.92 |     0.97 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      0 | eb2       |   74.64 |   74.97 |   -0.33 |   -0.95 |    0.32 | 0.28 |     0.59 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      0 | eb3       |   79.73 |   79.59 |    0.14 |   -0.74 |    0.96 | 0.74 |     0.90 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      0 | eb4       |   81.49 |   84.33 |   -2.85 |   -4.08 |   -1.49 | 0.00 |     0.00 | True              |
| VOC2007    | r3v_m2cl | r3v_m2 |      0 | eb_concat |   90.87 |   91.66 |   -0.79 |   -1.37 |   -0.24 | 0.00 |     0.02 | True              |
| VOC2007    | r3v_m2cl | r3v_m2 |      0 | backbone  |   92.23 |   92.54 |   -0.31 |   -0.66 |    0.01 | 0.06 |     0.21 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      1 | eb1a      |   78.90 |   78.75 |    0.15 |   -0.21 |    0.53 | 0.41 |     0.68 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      1 | eb1       |   73.90 |   73.64 |    0.26 |   -0.20 |    0.71 | 0.29 |     0.59 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      1 | eb2a      |   75.82 |   75.90 |   -0.08 |   -0.73 |    0.53 | 0.76 |     0.90 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      1 | eb2       |   77.33 |   77.51 |   -0.17 |   -0.74 |    0.40 | 0.56 |     0.85 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      1 | eb3       |   78.21 |   77.17 |    1.04 |    0.07 |    1.99 | 0.03 |     0.15 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      1 | eb4       |   84.88 |   85.79 |   -0.92 |   -2.20 |    0.34 | 0.15 |     0.38 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      1 | eb_concat |   91.26 |   90.97 |    0.29 |   -0.29 |    0.91 | 0.30 |     0.59 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      1 | backbone  |   92.87 |   92.61 |    0.26 |   -0.00 |    0.50 | 0.05 |     0.21 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      2 | eb1a      |   79.62 |   79.44 |    0.18 |   -0.23 |    0.55 | 0.39 |     0.68 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      2 | eb1       |   73.67 |   73.53 |    0.15 |   -0.33 |    0.58 | 0.55 |     0.85 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      2 | eb2a      |   77.30 |   77.87 |   -0.57 |   -1.12 |    0.00 | 0.05 |     0.21 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      2 | eb2       |   75.58 |   74.75 |    0.83 |    0.17 |    1.48 | 0.01 |     0.05 | True              |
| VOC2007    | r3v_m2cl | r3v_m2 |      2 | eb3       |   79.63 |   78.68 |    0.95 |   -0.05 |    1.99 | 0.06 |     0.21 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      2 | eb4       |   80.53 |   85.01 |   -4.48 |   -5.97 |   -2.96 | 0.00 |     0.00 | True              |
| VOC2007    | r3v_m2cl | r3v_m2 |      2 | eb_concat |   90.34 |   91.33 |   -1.00 |   -1.78 |   -0.22 | 0.01 |     0.06 | False             |
| VOC2007    | r3v_m2cl | r3v_m2 |      2 | backbone  |   92.80 |   92.72 |    0.08 |   -0.19 |    0.35 | 0.57 |     0.85 | False             |

*Paired bootstrap (2000 resamples) on per-sample correctness averaged over probe seeds; Benjamini-Hochberg across the 96 comparisons in this table.*

## VLCS round 3 (lr 1e-3): ERM / M2 / M2-CL vs ImageNet-only (backbone)

| Domain     | A        | B        |   seed | layer    |   acc A |   acc B |   Delta |   CI lo |   CI hi |    p |   p (BH) | sig (BH, q=.05)   |
|:-----------|:---------|:---------|-------:|:---------|--------:|--------:|--------:|--------:|--------:|-----:|---------:|:------------------|
| Caltech101 | r3v_m2   | imagenet |      0 | backbone |   82.63 |   82.78 |   -0.15 |   -1.17 |    0.77 | 0.72 |     0.89 | False             |
| Caltech101 | r3v_m2   | imagenet |      1 | backbone |   82.93 |   82.78 |    0.16 |   -0.79 |    1.07 | 0.76 |     0.89 | False             |
| Caltech101 | r3v_m2   | imagenet |      2 | backbone |   82.65 |   82.78 |   -0.12 |   -1.03 |    0.77 | 0.78 |     0.89 | False             |
| Caltech101 | r3v_m2cl | imagenet |      0 | backbone |   82.63 |   82.78 |   -0.14 |   -1.20 |    0.78 | 0.75 |     0.89 | False             |
| Caltech101 | r3v_m2cl | imagenet |      1 | backbone |   82.90 |   82.78 |    0.12 |   -0.85 |    1.04 | 0.80 |     0.89 | False             |
| Caltech101 | r3v_m2cl | imagenet |      2 | backbone |   82.86 |   82.78 |    0.09 |   -0.88 |    0.99 | 0.89 |     0.89 | False             |
| LabelMe    | r3v_m2   | imagenet |      0 | backbone |   88.50 |   88.55 |   -0.06 |   -0.79 |    0.65 | 0.88 |     0.89 | False             |
| LabelMe    | r3v_m2   | imagenet |      1 | backbone |   88.93 |   88.55 |    0.37 |   -0.35 |    1.08 | 0.31 |     0.71 | False             |
| LabelMe    | r3v_m2   | imagenet |      2 | backbone |   88.91 |   88.55 |    0.36 |   -0.36 |    1.07 | 0.35 |     0.71 | False             |
| LabelMe    | r3v_m2cl | imagenet |      0 | backbone |   88.63 |   88.55 |    0.07 |   -0.69 |    0.82 | 0.88 |     0.89 | False             |
| LabelMe    | r3v_m2cl | imagenet |      1 | backbone |   88.86 |   88.55 |    0.31 |   -0.48 |    1.04 | 0.42 |     0.71 | False             |
| LabelMe    | r3v_m2cl | imagenet |      2 | backbone |   88.87 |   88.55 |    0.31 |   -0.45 |    1.05 | 0.44 |     0.71 | False             |
| SUN09      | r3v_m2   | imagenet |      0 | backbone |   92.34 |   92.65 |   -0.31 |   -0.96 |    0.35 | 0.32 |     0.71 | False             |
| SUN09      | r3v_m2   | imagenet |      1 | backbone |   92.33 |   92.65 |   -0.32 |   -0.95 |    0.29 | 0.29 |     0.71 | False             |
| SUN09      | r3v_m2   | imagenet |      2 | backbone |   92.41 |   92.65 |   -0.24 |   -0.86 |    0.36 | 0.42 |     0.71 | False             |
| SUN09      | r3v_m2cl | imagenet |      0 | backbone |   92.38 |   92.65 |   -0.28 |   -0.94 |    0.40 | 0.39 |     0.71 | False             |
| SUN09      | r3v_m2cl | imagenet |      1 | backbone |   92.60 |   92.65 |   -0.05 |   -0.64 |    0.53 | 0.83 |     0.89 | False             |
| SUN09      | r3v_m2cl | imagenet |      2 | backbone |   92.25 |   92.65 |   -0.40 |   -1.04 |    0.24 | 0.20 |     0.71 | False             |
| VOC2007    | r3v_m2   | imagenet |      0 | backbone |   92.54 |   91.70 |    0.83 |    0.03 |    1.67 | 0.04 |     0.20 | False             |
| VOC2007    | r3v_m2   | imagenet |      1 | backbone |   92.61 |   91.70 |    0.91 |    0.15 |    1.72 | 0.03 |     0.20 | False             |
| VOC2007    | r3v_m2   | imagenet |      2 | backbone |   92.72 |   91.70 |    1.02 |    0.21 |    1.86 | 0.01 |     0.12 | False             |
| VOC2007    | r3v_m2cl | imagenet |      0 | backbone |   92.23 |   91.70 |    0.53 |   -0.31 |    1.34 | 0.21 |     0.71 | False             |
| VOC2007    | r3v_m2cl | imagenet |      1 | backbone |   92.87 |   91.70 |    1.17 |    0.39 |    2.01 | 0.01 |     0.10 | False             |
| VOC2007    | r3v_m2cl | imagenet |      2 | backbone |   92.80 |   91.70 |    1.10 |    0.26 |    1.96 | 0.01 |     0.10 | False             |

*Paired bootstrap (2000 resamples) on per-sample correctness averaged over probe seeds; Benjamini-Hochberg across the 24 comparisons in this table.*

## Sanity check: linear CLASS probe on the same backbone features (%)

| group    | target     |   class acc source-test |   class acc target |
|:---------|:-----------|------------------------:|-------------------:|
| imagenet | Caltech101 |                   78.25 |              95.83 |
| imagenet | LabelMe    |                   83.44 |              59.90 |
| imagenet | SUN09      |                   83.49 |              69.84 |
| imagenet | VOC2007    |                   81.46 |              74.56 |
| r3v_m2   | Caltech101 |                   84.13 |              94.11 |
| r3v_m2   | LabelMe    |                   87.99 |              61.06 |
| r3v_m2   | SUN09      |                   86.38 |              69.46 |
| r3v_m2   | VOC2007    |                   85.49 |              71.93 |
| r3v_m2cl | Caltech101 |                   84.11 |              94.75 |
| r3v_m2cl | LabelMe    |                   88.48 |              61.47 |
| r3v_m2cl | SUN09      |                   85.86 |              69.50 |
| r3v_m2cl | VOC2007    |                   85.34 |              72.61 |

*ImageNet features should give high class accuracy; near-chance (14%) values mean a label bug.*
