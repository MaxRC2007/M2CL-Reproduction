"""Statistics used in the paper (NumPy/SciPy only)."""
from __future__ import annotations

from itertools import product
from typing import Dict, Sequence

import numpy as np
from scipy.stats import binomtest


def paired_bootstrap(corr_a: np.ndarray, corr_b: np.ndarray, n_boot: int = 2000,
                     seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap over a SHARED test set.

    corr_a / corr_b: per-sample correctness in [0, 1] (0/1 for one probe, or the mean
    over several probe seeds). The observed delta is exactly acc_a - acc_b, so the
    point estimate always lies inside its own interval's sampling distribution.
    p = 2 * min(P[d* <= 0], P[d* >= 0]) (capped at 1); CI = 2.5/97.5 percentiles.
    """
    a = np.asarray(corr_a, dtype=np.float64)
    b = np.asarray(corr_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs identical test sets")
    rng = np.random.default_rng(seed)
    n = len(a)
    d = a - b
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = d[rng.integers(0, n, n)].mean()
    p = min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
    return {"acc_a": 100 * a.mean(), "acc_b": 100 * b.mean(), "delta_pp": 100 * d.mean(),
            "ci_lo_pp": 100 * np.percentile(boots, 2.5), "ci_hi_pp": 100 * np.percentile(boots, 97.5),
            "p": float(p)}


def benjamini_hochberg(pvals: Sequence[float], q: float = 0.05):
    """Returns (adjusted p-values, reject flags) in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj, adj <= q


def sign_test(diffs: Sequence[float]) -> float:
    """Exact two-sided binomial sign test on the number of positive differences (zeros dropped)."""
    d = [x for x in diffs if x != 0]
    if not d:
        return 1.0
    return float(binomtest(sum(x > 0 for x in d), len(d), 0.5).pvalue)


def sign_flip_test(diffs: Sequence[float]) -> float:
    """Exact two-sided sign-flip permutation test on the mean difference."""
    d = np.asarray(diffs, dtype=float)
    obs = abs(d.mean())
    stats = [abs((d * np.array(s)).mean()) for s in product([1, -1], repeat=len(d))]
    return float(np.mean([s >= obs - 1e-12 for s in stats]))


def min_attainable_p(n: int) -> float:
    return 2 / 2 ** n


def majority_rate(labels: np.ndarray) -> float:
    return float(np.bincount(labels).max() / len(labels))
