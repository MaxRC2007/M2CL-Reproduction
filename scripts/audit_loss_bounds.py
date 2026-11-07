"""Where does a summed loss of about 170 sit between chance and the best possible value?

  python scripts/audit_loss_bounds.py <repo root>

Uses the project's own loss on PACS-like batches of 128 with 6 representation sets.
Chance: representations with no class structure (random high-dimensional unit vectors).
Best: every sample of a class maps to the same unit vector, classes orthogonal or maximally spread.
"""
import sys

import torch

sys.path.insert(0, sys.argv[1] + "/src")
from m2cl.losses import m2cl_contrastive_loss

torch.manual_seed(0)
# PACS class counts (dog, elephant, giraffe, guitar, horse, house, person), all four domains
counts = torch.tensor([1729, 1654, 1566, 1113, 1540, 943, 1446], dtype=torch.float)
p = counts / counts.sum()
M, N, D = 6, 128, 512


def run(make, tau, n=200):
    vals = []
    for _ in range(n):
        y = torch.multinomial(p, N, replacement=True)
        vals.append(float(m2cl_contrastive_loss([make(y) for _ in range(M)], y, tau)))
    return sum(vals) / len(vals)


def random_reps(y):
    return torch.randn(len(y), D)


def collapsed_orthogonal(y):
    return torch.nn.functional.one_hot(y, D).float()


simplex = torch.eye(7) - 1 / 7  # 7 maximally spread unit directions (regular simplex)
simplex = torch.nn.functional.normalize(simplex, dim=1)


def collapsed_simplex(y):
    return torch.nn.functional.pad(simplex[y], (0, D - 7))


for tau in (1.0, 0.5, 0.07):
    print(f"tau {tau}: chance {run(random_reps, tau):.1f} | best, orthogonal classes {run(collapsed_orthogonal, tau):.1f}"
          f" | best, simplex classes {run(collapsed_simplex, tau):.1f}")
