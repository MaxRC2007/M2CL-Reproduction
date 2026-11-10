"""Torch-version check for the round-3 PACS primary test (docs/AUDIT.md F28). Fixed before the runs.

In the primary test, all M2 runs used torch 2.11.0 and 15 of the 20 M2-CL runs used 2.14.1. This check retrains
the four seed-0 M2 runs of the primary test (lr 1e-3, one per PACS fold) twice on a Colab T4: once on torch 2.14.1,
and once on 2.11.0 as a same-version control for GPU nondeterminism. Each is compared with the original run.

  version shift  = acc(2.14.1 rerun) - acc(2.11.0 rerun)     per fold
  rerun noise    = acc(2.11.0 rerun) - acc(original 2.11.0)  per fold

Reading (fixed in advance; n = 4, so descriptive, no p-values):
  * NEGLIGIBLE if the mean |version shift| is at most max(2 x mean |rerun noise|, 0.25 pp) and the mean signed
    version shift is within +/-0.22 pp (half the primary effect, +0.44 pp).
  * Otherwise REPORTED: the shift is given beside the primary result as a possible bias of that size.
  Backbone distances to the original run are reported for both versions.

Usage: python scripts/analyze_torchcheck.py <torchcheck zip or folder> <archive zip 06> [--json out.json]
"""
import argparse
import json
import re
import statistics as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_round3 import Store  # noqa: E402

FOLDS = ("art_painting", "cartoon", "photo", "sketch")
PRIMARY_EFFECT = 0.44
EXPECT = {"model": "m2", "arch_taps": 6, "alpha": 0.0, "lr": 0.001, "epochs": 30, "batch_size": 128, "seed": 0,
          "augmentation": "baseline", "momentum": 0.9, "weight_decay": 0.0005}


def check(r, where, torch_prefix):
    if r is None:
        sys.exit(f"missing: {where}")
    if r.get("status") != "complete":
        sys.exit(f"not finished: {where}")
    bad = {k: r["config"].get(k) for k, v in EXPECT.items() if r["config"].get(k) != v}
    if bad:
        sys.exit(f"{where}: settings differ from the round-3 M2 config: {bad}")
    if not str(r.get("torch", "")).startswith(torch_prefix):
        sys.exit(f"{where}: ran on torch {r.get('torch')}, expected {torch_prefix}")


def distance(store, pa, pb):
    sa, sb = store.state(pa), store.state(pb)
    ks = [k for k in sa if k.startswith(("layer0.", "layer1.", "layer2.", "layer3.", "layer4."))
          and sa[k].is_floating_point() and not k.endswith(("running_mean", "running_var"))]
    return float(sum(((sa[k].float() - sb[k].float()) ** 2).sum() for k in ks) ** 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("torchcheck")
    ap.add_argument("archive06")
    ap.add_argument("--json")
    ap.add_argument("--no_weights", action="store_true")
    a = ap.parse_args()
    store = Store([a.torchcheck, a.archive06])
    rows = []
    for f in FOLDS:
        p_orig = rf"lr/lr_0\.001/m2_6tap_{f}_seed0"
        p_new = rf"torchcheck/torch_2\.14\.1/m2_6tap_{f}_seed0"
        p_ctl = rf"torchcheck/torch_2\.11\.0/m2_6tap_{f}_seed0"
        o, n, c = store.result(p_orig), store.result(p_new), store.result(p_ctl)
        check(o, p_orig, "2.11.0"), check(n, p_new, "2.14.1"), check(c, p_ctl, "2.11.0")
        row = {"fold": f, "original": o["test_acc"], "torch_2.14.1": n["test_acc"], "torch_2.11.0_rerun": c["test_acc"],
               "best_epoch": [o["best_epoch"], n["best_epoch"], c["best_epoch"]],
               "version_shift": n["test_acc"] - c["test_acc"], "rerun_noise": c["test_acc"] - o["test_acc"]}
        if not a.no_weights:
            row["dist_2.14.1_vs_original"] = distance(store, p_new, p_orig)
            row["dist_2.11.0_vs_original"] = distance(store, p_ctl, p_orig)
        rows.append(row)

    print(f"{'fold':13s} {'orig':>7s} {'2.11 re':>8s} {'2.14.1':>7s} {'shift':>7s} {'noise':>7s}  best epochs"
          + ("   dist 2.14/2.11 vs orig" if not a.no_weights else ""))
    for r in rows:
        line = (f"{r['fold']:13s} {r['original']:7.2f} {r['torch_2.11.0_rerun']:8.2f} {r['torch_2.14.1']:7.2f} "
                f"{r['version_shift']:+7.2f} {r['rerun_noise']:+7.2f}  {r['best_epoch']}")
        if not a.no_weights:
            line += f"   {r['dist_2.14.1_vs_original']:.3f} / {r['dist_2.11.0_vs_original']:.3f}"
        print(line)
    shift = [r["version_shift"] for r in rows]
    noise = [r["rerun_noise"] for r in rows]
    mean_abs_shift, mean_abs_noise = st.mean(map(abs, shift)), st.mean(map(abs, noise))
    limit = max(2 * mean_abs_noise, 0.25)
    negligible = mean_abs_shift <= limit and abs(st.mean(shift)) <= PRIMARY_EFFECT / 2
    verdict = "NEGLIGIBLE" if negligible else "REPORTED"
    print(f"\nmean version shift {st.mean(shift):+.2f} pp (mean |shift| {mean_abs_shift:.2f}); "
          f"mean rerun noise {st.mean(noise):+.2f} pp (mean |noise| {mean_abs_noise:.2f})")
    print(f"VERDICT: {verdict} (rule: mean |shift| <= {limit:.2f} and |mean shift| <= {PRIMARY_EFFECT / 2:.2f})")
    if a.json:
        json.dump({"rows": rows, "mean_shift": st.mean(shift), "mean_abs_shift": mean_abs_shift,
                   "mean_noise": st.mean(noise), "mean_abs_noise": mean_abs_noise, "limit": limit,
                   "verdict": verdict}, open(a.json, "w"), indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
