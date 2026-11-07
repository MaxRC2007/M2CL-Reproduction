#!/usr/bin/env python
"""Audit A0: were the draft's leakage tables measured on DCT images instead of RGB PACS?

  python scripts/audit_dct_data.py --dct_root C:/.../PACS --rgb_root C:/.../PACS/pacs_data/pacs_data

Probes the ImageNet-only ResNet-18 on both copies with the draft's protocol and prints them
next to the numbers in results/legacy_v1/imagenet_control_results.json. If the draft used the
DCT copy, the DCT column reproduces the legacy domain AND class accuracies, and the class probe
recovers on RGB.
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402

from m2cl.features import build_condition, extract_all  # noqa: E402
from m2cl.probe import probe_correctness, train_linear_probe  # noqa: E402
from m2cl.splits import PACS_DOMAINS, probe_split  # noqa: E402


def probes(F, target, seeds):
    t = PACS_DOMAINS.index(target)
    X, d, c = F["feat__backbone"], F["domain_labels"], F["class_labels"]
    src, tgt = np.where(d != t)[0], np.where(d == t)[0]
    remap = {v: i for i, v in enumerate(sorted(set(d[src].tolist())))}
    y = np.array([remap[v] for v in d[src]])
    tr, te = probe_split(y)
    dom = probe_correctness(X[src][tr], y[tr], X[src][te], y[te], len(remap), seeds).mean(1)
    cls = [np.mean(train_linear_probe(X[src], c[src], X[tgt], 7, s) == c[tgt]) for s in seeds]
    return {"domain_acc": 100 * float(dom.mean()), "domain_sd": 100 * float(dom.std(ddof=1)),
            "class_acc": 100 * float(np.mean(cls)), "class_sd": 100 * float(np.std(cls, ddof=1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dct_root", required=True)
    ap.add_argument("--rgb_root", required=True)
    ap.add_argument("--cache", default="features/audit")
    ap.add_argument("--out", default="results/audit/dct_vs_rgb.json")
    ap.add_argument("--probe_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--subset", type=int, default=0,
                    help="images per domain (0 = all); a small sample is enough on a slow CPU")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(os.path.join(ROOT, "results", "legacy_v1", "imagenet_control_results.json")) as f:
        legacy = json.load(f)["results"]

    feats = {}
    for name, root in (("dct", a.dct_root), ("rgb", a.rgb_root)):
        p = os.path.join(a.cache, f"imagenet_{name}_n{a.subset}.npz")
        if not os.path.exists(p):
            net = build_condition("imagenet")
            np.savez_compressed(p, **extract_all(net, root, allow_dct=name == "dct", subset_per_domain=a.subset))
            print(f"extracted {name}", flush=True)
        feats[name] = dict(np.load(p))

    out = {}
    print(f"{'held-out':13s}{'':8s}{'draft JSON':>12s}{'DCT copy':>12s}{'RGB copy':>12s}")
    for T in PACS_DOMAINS:
        r = {n: probes(feats[n], T, a.probe_seeds) for n in feats}
        lg = legacy[T]
        r["draft"] = {"domain_acc": 100 * lg["domain_probes"]["backbone_final"]["linear_acc"],
                      "class_acc": 100 * lg["class_probes"]["backbone_final"]["linear_acc"]}
        out[T] = r
        for k, lab in (("domain_acc", "domain"), ("class_acc", "class")):
            print(f"{T:13s}{lab:8s}{r['draft'][k]:12.2f}{r['dct'][k]:12.2f}{r['rgb'][k]:12.2f}", flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
