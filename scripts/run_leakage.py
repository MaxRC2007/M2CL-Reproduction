#!/usr/bin/env python
"""Domain-leakage probes on cached features (CPU is fine; ~1 s per probe).

  python scripts/run_leakage.py --features_dir features --out results/leakage/raw
  python scripts/run_leakage.py --features_dir features --out results/leakage/zscore --standardize

For every held-out target T and every condition applicable to T:
  * source images only (T is never used), stratified 70/30 split by domain with a
    fixed split seed shared by ALL conditions -> identical test set for pairing;
  * linear DOMAIN probe per layer, repeated over --probe_seeds;
  * linear CLASS probe sanity check (source-test and target accuracy) per layer.
Saves per-sample correctness so make_tables.py can do any paired comparison later
without retraining a probe.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402

from m2cl.features import load_features  # noqa: E402
from m2cl.probe import probe_correctness, summarize, train_linear_probe  # noqa: E402
from m2cl.splits import ALL_DOMAINS, PACS_DOMAINS, domains_of, probe_split  # noqa: E402
from m2cl.stats import majority_rate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="features")
    ap.add_argument("--out", default="results/leakage/raw")
    ap.add_argument("--targets", nargs="+", default=list(PACS_DOMAINS))
    ap.add_argument("--groups", nargs="+", help="restrict to these condition groups")
    ap.add_argument("--layers", nargs="+", help="default: every layer present in a feature file")
    ap.add_argument("--probe_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--no_class_probe", action="store_true")
    a = ap.parse_args()

    with open(os.path.join(a.features_dir, "index.json")) as f:
        index = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    summary = {"config": vars(a), "targets": {}}
    arrays = {}
    t_start = time.time()
    summary_path, arrays_path = os.path.join(a.out, "summary.json"), os.path.join(a.out, "correctness.npz")
    if os.path.exists(summary_path) and os.path.exists(arrays_path):  # resume after an interruption
        with open(summary_path) as f:
            old = json.load(f)
        same = all(old["config"].get(k) == getattr(a, k) for k in vars(a) if k != "targets")
        if not same:
            sys.exit(f"{a.out} holds results from different settings; use another --out")
        summary["targets"] = old["targets"]
        with np.load(arrays_path) as z:
            arrays = {k: z[k] for k in z.files}

    for T in a.targets:
        if T in summary["targets"]:
            print(f"[skip] {T}: already probed", flush=True)
            continue
        entries = [e for e in index if e.get("target") in (None, T) and (not a.groups or e["group"] in a.groups)]
        if not entries:
            continue
        ref_hash, src, tr, te, y_dom, n_dom = None, None, None, None, None, None
        tsum = {"conditions": {}}
        for e in entries:
            F = load_features(os.path.join(a.features_dir, e["file"]))
            if ref_hash is None:
                ref_hash = F["sample_hash"]
                d_all = F["domain_labels"]
                t_idx = domains_of(T).index(T)
                src = np.where(d_all != t_idx)[0]
                tgt = np.where(d_all == t_idx)[0]
                remap = {d: i for i, d in enumerate(sorted(set(d_all[src].tolist())))}
                y_dom = np.array([remap[d] for d in d_all[src]])
                n_dom = len(remap)
                tr, te = probe_split(y_dom, a.test_size, a.split_seed)
                tsum.update({"n_source": int(len(src)), "n_probe_train": int(len(tr)),
                             "n_probe_test": int(len(te)),
                             "majority_baseline": 100 * majority_rate(y_dom[te]),
                             "source_domains": [domains_of(T)[d] for d in sorted(remap)]})
                arrays[f"{T}|test_idx"] = src[te]
            elif F["sample_hash"] != ref_hash:
                raise RuntimeError(f"{e['file']}: sample order differs from other conditions -> "
                                   "re-extract all features from the same PACS copy")
            key = f"{e['group']}|{e.get('seed')}"
            layers = [l for l in (a.layers or F["feats"].keys()) if l in F["feats"]]
            csum = {"group": e["group"], "seed": e.get("seed"), "file": e["file"], "layers": {}}
            for L in layers:
                X = F["feats"][L][src]
                corr = probe_correctness(X[tr], y_dom[tr], X[te], y_dom[te], n_dom, a.probe_seeds,
                                         epochs=a.epochs, standardize=a.standardize)
                arrays[f"{T}|{key}|{L}"] = corr
                ls = {"domain": summarize(corr)}
                if not a.no_class_probe:
                    yc = F["class_labels"]
                    p = train_linear_probe(X[tr], yc[src][tr], np.concatenate([X[te], F["feats"][L][tgt]]),
                                           len(F["classes"]), seed=0, epochs=a.epochs, standardize=a.standardize)
                    ls["class_source_test_acc"] = float(100 * np.mean(p[:len(te)] == yc[src][te]))
                    ls["class_target_acc"] = float(100 * np.mean(p[len(te):] == yc[tgt]))
                csum["layers"][L] = ls
                print(f"[{T:12s}] {key:22s} {L:10s} domain={ls['domain']['acc_mean']:6.2f}"
                      f" (probe-seed sd {ls['domain']['acc_sd_probe_seeds']:.2f})"
                      + (f"  class(src/tgt)={ls['class_source_test_acc']:.1f}/{ls['class_target_acc']:.1f}"
                         if 'class_target_acc' in ls else ""), flush=True)
            tsum["conditions"][key] = csum
        summary["targets"][T] = tsum
        np.savez_compressed(arrays_path + ".tmp.npz", **arrays)  # save after each target; arrays first
        os.replace(arrays_path + ".tmp.npz", arrays_path)
        with open(summary_path + ".tmp", "w") as f:
            json.dump(summary, f, indent=1)
        os.replace(summary_path + ".tmp", summary_path)

    print(f"done in {(time.time()-t_start)/60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
