#!/usr/bin/env python
"""Extract frozen features for every condition listed in a manifest.

  python scripts/extract_features.py --manifest configs/leakage_manifest.yaml \
         --data_root PACS --out_dir features

Manifest entries:  {group, kind: imagenet|random|ckpt, seed, target, model, taps, ckpt}
  * `group` is the condition name used in tables (e.g. m2cl6, erm_legacy, random)
  * `target` = the held-out domain the checkpoint was trained without (null for
    imagenet/random, which never saw PACS)
  * `ckpt` may contain {target} and {seed} placeholders; with `targets:`/`seeds:`
    lists one entry expands to many.
Also: --from_runs runs/  adds every completed run produced by train_lodo.py.
Existing .npz files are skipped, so the script can be re-run after interruptions.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from m2cl.features import build_condition, extract_all  # noqa: E402


def expand(entries, base_dir):
    out = []
    for e in entries:
        targets = e.get("targets", [e.get("target")])
        seeds = e.get("seeds", [e.get("seed")])
        for t in targets:
            for s in seeds:
                x = {k: v for k, v in e.items() if k not in ("targets", "seeds")}
                x["target"], x["seed"] = t, s
                if x.get("ckpt"):
                    p = x["ckpt"].format(target=t, seed=s)
                    x["ckpt"] = p if os.path.isabs(p) else os.path.join(base_dir, p)
                out.append(x)
    return out


def from_runs(runs_dir):
    out = []
    for r in sorted(glob.glob(os.path.join(runs_dir, "*", "result.json"))):
        with open(r) as f:
            res = json.load(f)
        if res.get("status") != "complete":
            continue
        c = res["config"]
        group = c["model"] + (f"_{c['arch_taps']}tap" if c["model"] != "erm" else "")
        out.append({"group": group, "kind": "ckpt", "model": c["model"], "taps": c["arch_taps"],
                    "seed": c["seed"], "target": c["target"],
                    "ckpt": os.path.join(os.path.dirname(r), "best.pt")})
    return out


def fname(e):
    return f"{e['group']}__seed{e.get('seed')}__{e.get('target') or 'all'}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--from_runs")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="features")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--checkpoint_root", help="overrides checkpoint_root in the manifest")
    a = ap.parse_args()

    entries = []
    if a.manifest:
        with open(a.manifest) as f:
            m = yaml.safe_load(f)
        base = a.checkpoint_root or m.get("checkpoint_root") or os.path.dirname(os.path.abspath(a.manifest))
        entries += expand(m["conditions"], base)
    if a.from_runs:
        entries += from_runs(a.from_runs)
    if not entries:
        sys.exit("nothing to extract: give --manifest and/or --from_runs")

    os.makedirs(a.out_dir, exist_ok=True)
    index_path = os.path.join(a.out_dir, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else []
    known = {i["file"] for i in index}
    for e in entries:
        fn = fname(e)
        path = os.path.join(a.out_dir, fn)
        if os.path.exists(path):
            print(f"[skip] {fn}")
        else:
            if e["kind"] == "ckpt" and not os.path.exists(e["ckpt"]):
                print(f"[MISSING] {e['ckpt']} -> skipped"); continue
            print(f"[extract] {fn}")
            net = build_condition(e["kind"], e.get("model", "erm"), e.get("taps", 4), e.get("ckpt"),
                                  e.get("seed") or 0, num_classes=e.get("num_classes", 7))
            feats = extract_all(net, a.data_root, a.batch_size, num_workers=a.num_workers)
            np.savez_compressed(path + ".tmp.npz", **feats)  # an interrupted save must not look complete
            os.replace(path + ".tmp.npz", path)
        if fn not in known:
            index.append({"file": fn, **{k: e.get(k) for k in ("group", "kind", "model", "taps", "seed", "target", "ckpt")}})
            known.add(fn)
            with open(index_path, "w") as f:
                json.dump(index, f, indent=1)
    print(f"index: {index_path} ({len(index)} conditions)")


if __name__ == "__main__":
    main()
