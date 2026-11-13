"""Pre-registered analysis of round 6 (docs/PREREGISTRATION_ROUND6.md): the authors' M2CL code on PACS, lparam 0.01
vs 0, seeds 3-7, primary endpoint = test accuracy at the final step (5000).

  python scripts/analyze_round6.py <round-6 results folder or zip> [--no_weights] [--json out.json]

Runs are found under round6/lparam_<x>/test<t>_seed<s>/ (DomainBed's results.jsonl, done, model.pkl). The run checks
reuse round 4's check_run (same settings except the seeds); the test reuses round 3's paired() and decide(), with the
same +/-0.71 pp margin. No decision is made with fewer than 20 complete pairs.
"""
import argparse
import io
import json
import os
import re
import statistics as st
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_round3 import MARGIN, decide, paired  # noqa: E402
from analyze_round4 import DOMAINS, N_STEPS, check_run, distance, select  # noqa: E402

SEEDS = (3, 4, 5, 6, 7)
N_PAIRS = len(SEEDS) * len(DOMAINS)                     # 20
ARMS = {"0.01": "M2-CL (lparam 0.01)", "0.0": "term off (lparam 0)"}
ACTIVE_FRACTION, ACTIVE_MIN_PAIRS = 0.10, 17
PAPER = {"M2-CL": 83.54, "M2": 82.83}
PAPER_GAIN = round(PAPER["M2-CL"] - PAPER["M2"], 2)     # 0.71
FINAL_STEP = N_STEPS - 1                                # 5000
RUN_RE = re.compile(r"round6/lparam_(0\.01|0\.0)/test([0-3])_seed(\d+)/(results\.jsonl|done|model\.pkl)$")


class Store:
    """Reads run files from a folder or a zip with the same layout."""

    def __init__(self, src):
        self.zip = zipfile.ZipFile(src) if src.endswith(".zip") else None
        self.root = src
        names = (self.zip.namelist() if self.zip else
                 [os.path.relpath(os.path.join(r, f), src).replace(os.sep, "/")
                  for r, _, fs in os.walk(src) for f in fs])
        self.files = {}
        for n in names:
            m = RUN_RE.search(n)
            if m:
                self.files[(m.group(1), int(m.group(2)), int(m.group(3)), m.group(4))] = n

    def read(self, key, binary=False):
        n = self.files[key]
        if self.zip:
            data = self.zip.read(n)
            return data if binary else data.decode("utf-8")
        with open(os.path.join(self.root, n), "rb" if binary else "r") as f:
            return f.read()


def load_runs(store):
    runs = {}
    for (lp, t, s, kind) in sorted(store.files):
        if kind != "results.jsonl" or (lp, t, s, "done") not in store.files:
            continue
        where = f"round6/lparam_{lp}/test{t}_seed{s}"
        if s not in SEEDS:
            raise SystemExit(f"{where}: seed {s} is not one of the pre-registered seeds {SEEDS}")
        records = [json.loads(line) for line in store.read((lp, t, s, kind)).splitlines() if line.strip()]
        check_run(records, lp, t, s, where)          # also requires the last record to be step 5000
        last = records[-1]
        assert last["step"] == FINAL_STEP, (where, last["step"])
        _, sel_test, sel_step = select(records, t)
        runs[(lp, t, s)] = {"final": 100 * last[f"env{t}_in_acc"],
                            "last5": 100 * st.mean(r[f"env{t}_in_acc"] for r in records[-5:]),
                            "selected": sel_test, "selected_step": sel_step,
                            "mem_gb": max(r.get("mem_gb", 0) for r in records)}
    return runs


def weights(store, key):
    import torch
    sd = torch.load(io.BytesIO(store.read(key + ("model.pkl",), binary=True)), map_location="cpu",
                    weights_only=False)["model_dict"]
    return {k: v.double() for k, v in sd.items()
            if v.is_floating_point() and "running_" not in k and "ext" not in k.lower()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--no_weights", action="store_true", help="skip the weight-distance manipulation check")
    ap.add_argument("--json")
    a = ap.parse_args()
    store = Store(a.source)
    runs = load_runs(store)
    report = {"n_runs_complete": len(runs), "runs": {f"lparam_{k[0]}/test{k[1]}_seed{k[2]}": v
                                                      for k, v in sorted(runs.items())}}
    print(f"{len(runs)} of {2 * N_PAIRS} round-6 runs complete")

    def arm(lp, field):
        return {(DOMAINS[t], s): v[field] for (l2, t, s), v in runs.items() if l2 == lp}

    print("\n=== PRIMARY (pre-registered): authors' M2CL, lparam 0.01 - lparam 0, PACS, seeds 3-7, final step 5000")
    res = paired("H1/H2 final step, lparam 0.01 - 0", arm("0.01", "final"), arm("0.0", "final"), margin=MARGIN,
                 report=report)
    report["decision"] = decide(res, N_PAIRS)
    print("DECISION:", report["decision"])
    if res and res.get("ci95") and res["n"] == N_PAIRS:
        lo, hi = (float(x) for x in res["ci95"])
        report["ci_contains_paper_gain"] = bool(lo <= PAPER_GAIN <= hi)
        report["ci_contains_zero"] = bool(lo <= 0 <= hi)
        print(f"95% CI [{lo:+.2f}, {hi:+.2f}] {'contains' if report['ci_contains_paper_gain'] else 'excludes'} the "
              f"paper's PACS gain of +{PAPER_GAIN:.2f} pp and {'contains' if report['ci_contains_zero'] else 'excludes'}"
              " zero (descriptive)")

    print("\n=== Manipulation check")
    if not a.no_weights:
        pairs, ref, cache = {}, {}, {}

        def w(key):
            if key not in cache:
                cache[key] = weights(store, key)
            return cache[key]
        for t in range(4):
            off = [s for s in SEEDS if ("0.0", t, s) in runs]
            between = [distance(w(("0.0", t, x)), w(("0.0", t, y))) for i, x in enumerate(off) for y in off[i + 1:]]
            if between:
                ref[DOMAINS[t]] = st.median(between)
            for s in SEEDS:
                if ("0.0", t, s) in runs and ("0.01", t, s) in runs:
                    pairs[(DOMAINS[t], s)] = distance(w(("0.0", t, s)), w(("0.01", t, s)))
            cache.clear()
        ratios = {k: d / ref[k[0]] for k, d in pairs.items() if k[0] in ref}
        active = sum(r > ACTIVE_FRACTION for r in ratios.values())
        report["manipulation_weight"] = {"pair_distance": {f"{k[0]}/seed{k[1]}": d for k, d in pairs.items()},
                                         "between_seed_median": ref,
                                         "ratio": {f"{k[0]}/seed{k[1]}": r for k, r in ratios.items()},
                                         "active_pairs": active, "pairs_checked": len(ratios)}
        if ratios:
            print(f"paired distance / between-seed median: {min(ratios.values()):.3f}-{max(ratios.values()):.3f}; "
                  f"active (> {ACTIVE_FRACTION:.0%}) in {active} of {len(ratios)} pairs "
                  f"(pre-registered: at least {ACTIVE_MIN_PAIRS} of {N_PAIRS})")

    print("\n=== Secondary (descriptive, no decision)")
    paired("validation-selected checkpoint, lparam 0.01 - 0", arm("0.01", "selected"), arm("0.0", "selected"),
           margin=MARGIN, report=report)
    paired("mean of the last 5 checkpoints, lparam 0.01 - 0", arm("0.01", "last5"), arm("0.0", "last5"),
           margin=MARGIN, report=report)
    print("round 4 (seeds 0-2), final step, exploratory: +0.74 pp [+0.15, +1.32] (audit F34); not pooled")
    for lp, label in ARMS.items():
        per = {}
        for t, d in enumerate(DOMAINS):
            xs = [runs[(lp, t, s)]["final"] for s in SEEDS if (lp, t, s) in runs]
            if xs:
                per[d] = (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0, len(xs))
        macro = st.mean(v[0] for v in per.values()) if len(per) == 4 else None
        report.setdefault("means_final_step", {})[label] = {"per_domain": per, "macro": macro}
        cells = ", ".join(f"{d} {m:.2f}+/-{sd:.2f} (n={n})" for d, (m, sd, n) in per.items())
        print(f"{label}, final step: {cells}" + (f" | macro {macro:.2f}" if macro is not None else ""))
    print(f"paper, ResNet-18 PACS macro: M2-CL {PAPER['M2-CL']}, M2 {PAPER['M2']} (validation-selected, a separate "
          "M2 model)")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1, default=str)
        print("\nwrote", a.json)


if __name__ == "__main__":
    main()
