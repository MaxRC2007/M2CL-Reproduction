#!/usr/bin/env python
"""Re-evaluate every saved checkpoint on its held-out domain, on the RGB and DCT copies of PACS.

  python scripts/verify_checkpoints.py --manifest configs/leakage_manifest.yaml \
         --rgb_root C:/.../PACS/pacs_data/pacs_data --dct_root C:/.../PACS

Two checks at once: the checkpoint loads correctly into this code base (RGB accuracy matches the
draft), and which image copy it was trained on (only that copy gives high accuracy).
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from extract_features import expand  # noqa: E402
from m2cl.data import get_transforms, make_loader  # noqa: E402
from m2cl.features import build_condition  # noqa: E402
from m2cl.splits import index_pacs  # noqa: E402


@torch.no_grad()
def acc(net, samples, batch_size, num_workers):
    _, tf = get_transforms("baseline", 224)
    hit = n = 0
    for x, y, _ in make_loader(samples, tf, batch_size, shuffle=False, num_workers=num_workers):
        hit += (net(x)[0].argmax(1) == y).sum().item()
        n += len(y)
    return 100.0 * hit / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rgb_root", required=True)
    ap.add_argument("--dct_root", required=True)
    ap.add_argument("--dct_subset", type=int, default=500, help="images per target for the DCT check")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out", default="results/audit/checkpoint_accuracy.json")
    a = ap.parse_args()
    with open(a.manifest) as f:
        m = yaml.safe_load(f)
    entries = [e for e in expand(m["conditions"], m["checkpoint_root"]) if e["kind"] == "ckpt"]
    _, rgb = index_pacs(a.rgb_root)
    _, dct = index_pacs(a.dct_root, allow_dct=True)
    out = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = json.load(open(out)) if os.path.exists(out) else {}
    for e in entries:
        key = f"{e['group']}|{e['target']}|{e['seed']}"
        if key in res or not os.path.exists(e["ckpt"]):
            print(f"[skip] {key}", flush=True)
            continue
        net = build_condition("ckpt", e["model"], e["taps"], e["ckpt"]).eval()
        idx = np.random.RandomState(0).permutation(len(dct[e["target"]]))[:a.dct_subset]
        res[key] = {"rgb": acc(net, rgb[e["target"]], a.batch_size, a.num_workers),
                    "dct_subset": acc(net, [dct[e["target"]][i] for i in idx], a.batch_size, a.num_workers)}
        print(f"{key:40s} rgb={res[key]['rgb']:6.2f}  dct={res[key]['dct_subset']:6.2f}", flush=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
