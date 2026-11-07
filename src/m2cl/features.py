"""Frozen-feature extraction for the leakage probes.

A "condition" is one frozen network:
  imagenet                      ResNet-18, ImageNet weights, no PACS exposure
  random  (seed s)              ResNet-18, random init with torch seed s
  ckpt    (model, taps, path)   a trained ERM / M2 / M2-CL checkpoint

Features are extracted for ALL PACS images in one canonical (sorted) order and
saved as .npz together with class/domain labels and a hash of the sample list,
so every later paired comparison can assert row alignment.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch

from .data import get_transforms, make_loader
from .models import ERM, build_model, load_checkpoint
from .splits import PACS_DOMAINS, index_pacs, sample_ids_hash


def build_condition(kind: str, model: str = "erm", taps: int = 4, ckpt: Optional[str] = None,
                    seed: int = 0, pretrained_backbone: bool = True) -> torch.nn.Module:
    if kind == "imagenet":
        return ERM(pretrained=pretrained_backbone)
    if kind == "random":
        torch.manual_seed(seed)
        return ERM(pretrained=False)
    if kind == "ckpt":
        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        return load_checkpoint(build_model(model, 7, taps, pretrained=False), ckpt)
    raise ValueError(kind)


@torch.no_grad()
def extract_all(net: torch.nn.Module, data_root: str, batch_size: int = 64, img_size: int = 224,
                num_workers: int = 2, allow_dct: bool = False,
                subset_per_domain: int = 0) -> Dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()
    classes, by_dom = index_pacs(data_root, allow_dct=allow_dct)
    if subset_per_domain:  # audit only: same seeded sample for every condition
        by_dom = {d: [v[i] for i in sorted(np.random.RandomState(0).permutation(len(v))[:subset_per_domain])]
                  for d, v in by_dom.items()}
    samples = [s for d in PACS_DOMAINS for s in by_dom[d]]
    _, te_tf = get_transforms("baseline", img_size)
    loader = make_loader(samples, te_tf, batch_size, shuffle=False, num_workers=num_workers)
    acc: Dict[str, list] = {}
    for x, _, _ in loader:
        for k, v in net.forward_features(x.to(device)).items():
            acc.setdefault(k, []).append(v.float().cpu().numpy())
    out = {f"feat__{k}": np.concatenate(v) for k, v in acc.items()}
    out["class_labels"] = np.array([s[1] for s in samples], dtype=np.int64)
    out["domain_labels"] = np.array([s[2] for s in samples], dtype=np.int64)
    out["sample_hash"] = np.array(sample_ids_hash(samples))
    out["classes"] = np.array(classes)
    return out


def load_features(path: str) -> Dict:
    z = np.load(path, allow_pickle=False)
    return {"feats": {k[6:]: z[k] for k in z.files if k.startswith("feat__")},
            "class_labels": z["class_labels"], "domain_labels": z["domain_labels"],
            "sample_hash": str(z["sample_hash"]), "classes": list(z["classes"])}
