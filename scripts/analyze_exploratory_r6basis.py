"""Exploratory analyses run on 13 Nov 2026, after rounds 3-5 were complete (audit F34). None was pre-registered.

  python -I scripts/analyze_exploratory_r6basis.py --json results/exploratory/r6basis.json

1. Checkpoint selection (rounds 4 and 5, the authors' code): term on minus off at a fixed checkpoint instead of the
   validation-selected one: (a) the final step, 5000; (b) the mean over the last five checkpoints (steps 3900-5000).
2. Unit of analysis (rounds 3-5): the 12-pair interval against (a) a fold-stratified interval, which holds the four
   held-out domains fixed and pools the within-domain seed variance (df = 8), and (b) seeds as units (n = 3, df = 2).
3. Learning-rate test S2 without the four seed-0 pairs, which were seen before round 3 was specified.

Analysis 1 is the basis of round 6 (docs/PREREGISTRATION_ROUND6.md). It was chosen after the data were seen and two
variants were computed, so its p values are not confirmatory.
"""
import argparse
import json
import math
import os
import statistics as st
import sys

from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze_round3 import paired  # noqa: E402

MARGIN = 0.71


def load(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def records(run_root, lp, t, s):
    with open(os.path.join(ROOT, run_root, f"lparam_{lp}", f"test{t}_seed{s}", "results.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def fixed_checkpoint(run_root, domains, report, label):
    final, last5 = {}, {}
    for lp in ("0.01", "0.0"):
        for t, dom in enumerate(domains):
            for s in range(3):
                rec = records(run_root, lp, t, s)
                key = (dom, s)
                final.setdefault(lp, {})[key] = 100 * rec[-1][f"env{t}_in_acc"]
                last5.setdefault(lp, {})[key] = 100 * st.mean(r[f"env{t}_in_acc"] for r in rec[-5:])
                assert rec[-1]["step"] == 5000, (lp, t, s, rec[-1]["step"])
    paired(f"{label} final step", final["0.01"], final["0.0"], margin=MARGIN, report=report)
    paired(f"{label} mean of last 5 checkpoints", last5["0.01"], last5["0.0"], margin=MARGIN, report=report)


def units(name, diffs, out):
    by_fold, by_seed = {}, {}
    for k, v in diffs.items():
        f, s = k.split("/")
        by_fold.setdefault(f, []).append(v)
        by_seed.setdefault(s, []).append(v)
    xs = list(diffs.values())
    n, m = len(xs), st.mean(xs)
    h12 = stats.t.ppf(0.975, n - 1) * st.stdev(xs) / math.sqrt(n)
    pooled = st.mean(st.variance(v) for v in by_fold.values())
    df = sum(len(v) - 1 for v in by_fold.values())
    se = math.sqrt(pooled / len(next(iter(by_fold.values()))) / len(by_fold))
    hs = stats.t.ppf(0.975, df) * se
    seeds = [st.mean(v) for v in by_seed.values()]
    hu = stats.t.ppf(0.975, len(seeds) - 1) * st.stdev(seeds) / math.sqrt(len(seeds))
    out[name] = {"mean": m, "ci_12_pairs": [m - h12, m + h12], "ci_fold_stratified": [m - hs, m + hs],
                 "df_fold_stratified": df, "ci_seed_units": [m - hu, m + hu], "seed_means": seeds}
    print(f"{name}: mean {m:+.2f}; 12 pairs [{m - h12:+.2f}, {m + h12:+.2f}]; fold-stratified (df {df}) "
          f"[{m - hs:+.2f}, {m + hs:+.2f}]; seeds as units (n 3) [{m - hu:+.2f}, {m + hu:+.2f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    report = {"note": "exploratory; not pre-registered (audit F34)"}

    print("=== 1. Fixed checkpoint instead of validation selection (authors' code)")
    fixed_checkpoint("results/round4/institute/run_results",
                     ("art_painting", "cartoon", "photo", "sketch"), report, "R4 PACS")
    fixed_checkpoint("results/round5/institute/run_results",
                     ("Caltech101", "LabelMe", "SUN09", "VOC2007"), report, "R5 VLCS")

    print("\n=== 2. Unit of analysis")
    r3 = load("results/round3/round3_analysis_all.json")
    tests = {"R3 PACS M2-CL - M2": r3["H1/H2 M2-CL - M2"]["diffs"],
             "R3 VLCS M2-CL - M2": r3["S4 VLCS M2-CL - M2"]["diffs"],
             "R4 PACS lparam 0.01 - 0": load("results/round4/round4_analysis.json")["H1/H2 lparam 0.01 - 0"]["diffs"],
             "R5 VLCS lparam 0.01 - 0": load("results/round5/round5_analysis.json")["H1/H2 lparam 0.01 - 0"]["diffs"]}
    report["units"] = {}
    for name, d in tests.items():
        units(name, d, report["units"])

    print("\n=== 3. S2 without the seed-0 pairs")
    d = {k: v for k, v in r3["S2 M2 lr 1e-3 - lr 5e-4 (PACS)"]["diffs"].items() if not k.endswith("seed0")}
    keys = [tuple(k.split("/")) for k in d]
    paired("S2 lr 1e-3 - 5e-4, seeds 1-2", {k: d["/".join(k)] for k in keys}, {k: 0.0 for k in keys}, report=report)

    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1, default=str)
        print("\nwrote", a.json)


if __name__ == "__main__":
    main()
