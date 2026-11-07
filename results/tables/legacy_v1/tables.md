# Generated tables

Do not edit by hand: regenerate with scripts/make_tables.py

## Target-domain test accuracy (%), erm

| Domain        |   Seed 0 |   Seed 1 |   Seed 2 |   Mean |     SD | n    |
|:--------------|---------:|---------:|---------:|-------:|-------:|:-----|
| Art Painting  |    72.85 |    74.80 |    72.22 |  73.29 |   1.35 | 3    |
| Cartoon       |    70.78 |    67.53 |    66.08 |  68.13 |   2.41 | 3    |
| Photo         |    94.67 |    95.63 |    94.91 |  95.07 |   0.50 | 3    |
| Sketch        |    66.20 |    61.49 |    65.77 |  64.49 |   2.60 | 3    |
| Macro-average |   nan    |   nan    |   nan    |  75.24 | nan    | <NA> |

*SD = sample standard deviation (ddof=1) across seeds.*

## Target-domain test accuracy (%), erm_hist_1seed

| Domain        |   Seed 0 |   Mean |   SD | n    |
|:--------------|---------:|-------:|-----:|:-----|
| Art Painting  |    73.14 |  73.14 |  nan | 1    |
| Cartoon       |    70.69 |  70.69 |  nan | 1    |
| Photo         |    94.37 |  94.37 |  nan | 1    |
| Sketch        |    66.51 |  66.51 |  nan | 1    |
| Macro-average |   nan    |  76.18 |  nan | <NA> |

*SD = sample standard deviation (ddof=1) across seeds.*

## Target-domain test accuracy (%), m2_4tap_hist

| Domain        |   Seed 0 |   Mean |   SD | n    |
|:--------------|---------:|-------:|-----:|:-----|
| Art Painting  |    74.27 |  74.27 |  nan | 1    |
| Cartoon       |    67.36 |  67.36 |  nan | 1    |
| Photo         |    95.39 |  95.39 |  nan | 1    |
| Sketch        |    61.06 |  61.06 |  nan | 1    |
| Macro-average |   nan    |  74.52 |  nan | <NA> |

*SD = sample standard deviation (ddof=1) across seeds.*

## Target-domain test accuracy (%), m2cl_4tap_hist

| Domain        |   Seed 0 |   Mean |   SD | n    |
|:--------------|---------:|-------:|-----:|:-----|
| Art Painting  |    74.22 |  74.22 |  nan | 1    |
| Cartoon       |    68.30 |  68.30 |  nan | 1    |
| Photo         |    95.45 |  95.45 |  nan | 1    |
| Sketch        |    61.03 |  61.03 |  nan | 1    |
| Macro-average |   nan    |  74.75 |  nan | <NA> |

*SD = sample standard deviation (ddof=1) across seeds.*

## Target-domain test accuracy (%), m2cl_6tap

| Domain        |   Seed 0 |   Seed 1 |   Seed 2 |   Mean |     SD | n    |
|:--------------|---------:|---------:|---------:|-------:|-------:|:-----|
| Art Painting  |    74.51 |    76.03 |    75.29 |  75.28 |   0.76 | 3    |
| Cartoon       |    68.43 |    73.04 |    68.05 |  69.84 |   2.78 | 3    |
| Photo         |    95.57 |    95.69 |    95.75 |  95.67 |   0.09 | 3    |
| Sketch        |    62.89 |    65.41 |    63.60 |  63.97 |   1.30 | 3    |
| Macro-average |   nan    |   nan    |   nan    |  76.19 | nan    | <NA> |

*SD = sample standard deviation (ddof=1) across seeds.*

## Matched-seed accuracy comparison: m2cl_6tap minus erm (pp)

| Domain        |   m2cl_6tap |   erm |   Delta |   d(seed 0) |   d(seed 1) |   d(seed 2) |   sign test p |   sign-flip p | n    |
|:--------------|------------:|------:|--------:|------------:|------------:|------------:|--------------:|--------------:|:-----|
| Art Painting  |       75.28 | 73.29 |    1.99 |        1.66 |        1.23 |        3.07 |          0.25 |          0.25 | 3    |
| Cartoon       |       69.84 | 68.13 |    1.71 |       -2.35 |        5.51 |        1.97 |          1.00 |          0.75 | 3    |
| Photo         |       95.67 | 95.07 |    0.60 |        0.90 |        0.06 |        0.84 |          0.25 |          0.25 | 3    |
| Sketch        |       63.97 | 64.49 |   -0.52 |       -3.31 |        3.92 |       -2.17 |          1.00 |          1.00 | 3    |
| Macro-average |       76.19 | 75.24 |    0.94 |      nan    |      nan    |      nan    |               |               | <NA> |

*Seeds matched by source train/val partition. With n=3 seeds the smallest attainable two-sided p is 0.250; differences are descriptive.*
