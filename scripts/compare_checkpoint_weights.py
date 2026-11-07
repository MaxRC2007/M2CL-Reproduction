#!/usr/bin/env python
"""Backbone weight distances between checkpoints (evidence for docs/AUDIT.md, E2).

Asks how different the 4-tap M2 and M2-CL networks really are, using as references how far
each moved from its ImageNet start and how far two 6-tap M2-CL seeds lie apart.
Only learned backbone parameters count (BatchNorm running statistics are excluded).

  python scripts/compare_checkpoint_weights.py --ckpt_root C:/Users/Madhusudan/Downloads
"""
import argparse
import json
import os
import re

import torch
import torchvision


def backbone_params(path, reference):
    state = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for key, value in state.items():
        if not (hasattr(value, "dtype") and value.dtype.is_floating_point):
            continue
        name = key.replace("module.", "").replace("layer0.0.", "conv1.").replace("layer0.1.", "bn1.")
        name = re.sub(r"^(backbone|net|encoder|resnet|model)\.", "", name)
        if name in reference and reference[name].shape == value.shape and "running" not in name:
            out[name] = value.float()
    return out


def distance(a, b):
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("no shared parameters")
    return float(torch.sqrt(sum(((a[k] - b[k]) ** 2).sum() for k in keys)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--out", default="results/audit/checkpoint_weight_distance.json")
    a = ap.parse_args()
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    imagenet = {k: v.float() for k, v in torchvision.models.resnet18(weights=weights).state_dict().items()
                if v.dtype.is_floating_point}
    results = {}
    for target in ("art_painting", "cartoon", "photo"):
        load = lambda rel: backbone_params(os.path.join(a.ckpt_root, rel), imagenet)
        m2 = load(f"m2cl_checkpoints/m2_{target}.pt")
        m2cl = load(f"m2cl_checkpoints/m2cl_{target}.pt")
        s0 = load(f"m2cl6tap/m2cl6tap_{target}_seed0.pt")
        s1 = load(f"m2cl6tap/m2cl6tap_{target}_seed1.pt")
        results[target] = {"n_params": len(set(m2) & set(m2cl)),
                           "m2_from_imagenet": distance(m2, imagenet),
                           "m2cl_from_imagenet": distance(m2cl, imagenet),
                           "m2_vs_m2cl": distance(m2, m2cl),
                           "m2cl6_seed0_vs_seed1": distance(s0, s1)}
        print(target, {k: round(v, 3) for k, v in results[target].items()})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
