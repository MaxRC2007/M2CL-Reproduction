"""Laptop pre-check for round 4. NOT part of the experiment, and its numbers are never used.

It imports the authors' DomainBed code at the pinned commit (as train.py does, including `wilds`), builds their
M2CL with the round-4 settings (ResNet-18, lr 1e-3, temp 1.0), and runs two training updates on random images
on the CPU, once with lparam 0.01 and once with 0.0. This catches import, model-building and loss errors a day
before the institute smoke run. It does not touch the data loaders or the GPU, which only the institute run tests.

Usage: python scripts/check_authors_cpu.py <folder with the authors' code, checked out at the pinned commit>
"""
import argparse
import importlib.metadata as md
import math
import subprocess
import sys
import time

COMMIT = "0b1bce3026d9d56369037129e36f81e7645b80e6"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("authors_dir")
    ap.add_argument("--per_domain", type=int, default=32, help="images per source domain (round 4 uses 32)")
    a = ap.parse_args()

    head = subprocess.run(["git", "-C", a.authors_dir, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", a.authors_dir, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if head != COMMIT or dirty:
        sys.exit(f"The authors' folder is at {head or '?'}{' with changes' if dirty else ''}; expected the "
                 f"unmodified commit {COMMIT}.")
    print(f"[ok] authors' code at the pinned commit {COMMIT[:7]}, unmodified")

    import torch
    vers = {}
    for p in ("torch", "torchvision", "wilds", "timm", "numpy"):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    print("[info] versions:", vers)

    sys.path.insert(0, a.authors_dir)
    from domainbed import algorithms, datasets, hparams_registry  # noqa: F401  (datasets imports wilds)
    print("[ok] the authors' algorithms, datasets and hparams modules import")

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    n = a.per_domain
    batches = [(torch.randn(n, 3, 224, 224, generator=g), torch.randint(0, 7, (n,), generator=g)) for _ in range(3)]
    for lparam in (0.01, 0.0):
        hp = hparams_registry.default_hparams("M2CL", "PACS")       # what train.py uses for hparams_seed 0
        hp.update({"resnet18": True})                                 # --hparams '{"resnet18": true}'
        hp.update(lr=0.001, temp=1.0, lparam=lparam)                  # --lr 0.001 --temp 1.0 --lparam x
        t = time.time()
        alg = algorithms.get_algorithm_class("M2CL")((3, 224, 224), 7, 3, hp)   # PACS: 7 classes, 3 source domains
        n_par = sum(p.numel() for p in alg.parameters())
        losses = []
        for _ in range(2):
            losses.append(alg.update(batches)["loss"])
        if not all(math.isfinite(x) for x in losses):
            sys.exit(f"[FAILED] lparam {lparam}: non-finite loss {losses}")
        alg.eval()
        with torch.no_grad():
            out = alg.predict(batches[0][0][:4])
        def shape(o):
            return tuple(o.shape) if hasattr(o, "shape") else [shape(x) for x in o][:3]
        shapes = shape(out)
        print(f"[ok] lparam {lparam}: {n_par / 1e6:.1f} M parameters, 2 updates on {3 * n} random images, "
              f"loss {losses[0]:.3f} -> {losses[1]:.3f}, predict output shapes {shapes}, {time.time() - t:.0f} s")
    print("All checks passed. The data loaders and the GPU are tested by the institute smoke run.")


if __name__ == "__main__":
    main()
