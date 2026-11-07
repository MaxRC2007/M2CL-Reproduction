"""M2-CL multilayer supervised contrastive loss (Ballas & Diou, 2024, Eqs. 1-2).

    p_l(c) = sum_{i<j, y_i=y_j=c} exp(u_i.u_j/tau) / sum_{k<m} exp(u_k.u_m/tau)
    L      = CE + alpha * sum_l sum_c -log p_l(c)

reduction="sum"  : the paper's objective (used for every corrected run).
reduction="mean" : the ORIGINAL bug (mean over classes and layers) -- kept only so
                   the audit/tests can show the ~C*M scale difference.
Ordered pairs (i != j) are used in numerator and denominator; the factor 2
relative to i<j cancels in the ratio.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def m2cl_contrastive_loss(reps: Sequence[torch.Tensor], labels: torch.Tensor, tau: float = 1.0,
                          reduction: str = "sum") -> torch.Tensor:
    if reduction not in ("sum", "mean"):
        raise ValueError(reduction)
    per_layer = []
    for u in reps:
        u = F.normalize(u.float(), dim=1)
        e = torch.exp(u @ u.t() / tau)
        off = ~torch.eye(len(u), dtype=torch.bool, device=u.device)
        denom = e[off].sum()
        terms = []
        for c in labels.unique():
            m = labels == c
            if int(m.sum()) < 2:
                continue  # class has no positive pair in this batch
            numer = (e[m][:, m] * off[m][:, m]).sum()
            terms.append(-torch.log(numer / denom + 1e-12))
        if terms:
            t = torch.stack(terms)
            per_layer.append(t.sum() if reduction == "sum" else t.mean())
    if not per_layer:
        return labels.new_zeros((), dtype=torch.float32)
    t = torch.stack(per_layer)
    return t.sum() if reduction == "sum" else t.mean()
