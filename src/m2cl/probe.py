"""Seeded linear probes on cached features.

Protocol (paper Sec. V-A): one linear layer, Adam lr 1e-3, 30 epochs, batch 64,
cross-entropy, raw features by default. Differences from the original scripts:
  * the probe is fully seeded (init AND minibatch order) -> bit-for-bit repeatable;
  * the reported accuracy and the bootstrap interval come from the SAME probe(s);
  * each condition is probed with several probe seeds; per-sample correctness is
    averaged over seeds before the paired bootstrap, so probe-initialisation
    noise is averaged out instead of being mistaken for an effect;
  * optional z-scoring (fit on probe-train split only) -- raw features from a
    random-init network have very different scales per dimension and make the
    raw-feature probe unstable (docs/AUDIT.md, issue B2).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn


def train_linear_probe(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, n_classes: int, seed: int,
                       epochs: int = 30, lr: float = 1e-3, batch_size: int = 64,
                       standardize: bool = False) -> np.ndarray:
    """Returns predicted labels for xte."""
    xtr = torch.as_tensor(xtr, dtype=torch.float32)
    xte = torch.as_tensor(xte, dtype=torch.float32)
    ytr = torch.as_tensor(ytr, dtype=torch.long)
    if standardize:
        mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-6)
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    probe = nn.Linear(xtr.shape[1], n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(xtr)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            loss = lossf(probe(xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return probe(xte).argmax(1).numpy()


def probe_correctness(xtr, ytr, xte, yte, n_classes: int, seeds: Sequence[int], **kw) -> np.ndarray:
    """(n_seeds, n_test) boolean matrix of per-sample correctness, one row per probe seed."""
    return np.stack([train_linear_probe(xtr, ytr, xte, n_classes, s, **kw) == yte for s in seeds])


def summarize(corr: np.ndarray) -> Dict[str, float]:
    accs = 100 * corr.mean(1)
    return {"acc_mean": float(accs.mean()),
            "acc_sd_probe_seeds": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "acc_range_probe_seeds": float(accs.max() - accs.min()),
            "acc_per_probe_seed": [float(a) for a in accs]}
