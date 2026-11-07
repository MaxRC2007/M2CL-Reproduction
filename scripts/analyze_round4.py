"""Pre-registered analysis of round 4 (docs/PREREGISTRATION_ROUND4.md): the authors' M2CL code, lparam 0.01 vs 0.

  python scripts/analyze_round4.py <round-4 results folder or zip> [--no_weights] [--json out.json]

Runs are found under round4/lparam_<x>/test<t>_seed<s>/ (DomainBed's results.jsonl, done, model.pkl).
Model selection is DomainBed's training-domain validation (IIDAccuracySelectionMethod): the checkpoint with the
highest mean source-domain out-split accuracy (first on ties); its test-domain in-split accuracy is reported.
The primary test reuses round 3's paired() and decide(), with the same +/-0.71 pp margin.
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

DOMAINS = ("art_painting", "cartoon", "photo", "sketch")  # test_envs 0-3: DomainBed sorts the folder names
SEEDS = (0, 1, 2)
ARMS = {"0.01": "M2-CL (lparam 0.01)", "0.0": "term off (lparam 0)"}
EXPECT = {"algorithm": "M2CL", "dataset": "PACS", "lr": 0.001, "temp": 1.0, "hparams_seed": 0,
          "holdout_fraction": 0.2, "task": "domain_generalization"}
N_STEPS, CHECKPOINT_FREQ = 5001, 300
ACTIVE_FRACTION, ACTIVE_MIN_PAIRS = 0.10, 10
PAPER = {"M2-CL": 83.54, "M2": 82.83}
RUN_RE = re.compile(r"round4/lparam_(0\.01|0\.0)/test([0-3])_seed(\d+)/(results\.jsonl|done|model\.pkl)$")


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


def select(records, test_env):
    """DomainBed IIDAccuracySelectionMethod for one run: (val %, test %, step)."""
    best = None
    for r in records:
        val = st.mean(r[f"env{j}_out_acc"] for j in range(4) if j != test_env)
        if best is None or val > best[0]:
            best = (val, r[f"env{test_env}_in_acc"], r["step"])
    return best[0] * 100, best[1] * 100, best[2]


def check_run(records, lparam, test_env, seed, where):
    """Every record must have the pre-registered settings; the run must reach the last step."""
    steps = [r["step"] for r in records]
    want_steps = sorted(set(range(0, N_STEPS, CHECKPOINT_FREQ)) | {N_STEPS - 1})
    if steps != want_steps:
        raise SystemExit(f"{where}: checkpoint steps {steps[:3]}...{steps[-2:]} differ from the plan {want_steps[-2:]}")
    for r in records:
        a, h = r["args"], r["hparams"]
        bad = {k: a.get(k) for k, v in EXPECT.items() if a.get(k) != v}
        if a.get("lparam") != float(lparam) or a.get("test_envs") != [test_env] or a.get("seed") != seed \
                or a.get("trial_seed") != seed or a.get("steps") not in (None, N_STEPS) \
                or a.get("checkpoint_freq") not in (None, CHECKPOINT_FREQ):
            bad.update(lparam=a.get("lparam"), test_envs=a.get("test_envs"), seed=a.get("seed"),
                       trial_seed=a.get("trial_seed"), steps=a.get("steps"))
        if h.get("resnet18") is not True or h.get("batch_size") != 32 or h.get("data_augmentation") is not True:
            bad.update(resnet18=h.get("resnet18"), batch_size=h.get("batch_size"))
        if bad:
            raise SystemExit(f"{where}: settings differ from the pre-registration: {bad}")


def load_runs(store):
    runs = {}
    for (lp, t, s, kind) in sorted(store.files):
        if kind != "results.jsonl" or (lp, t, s, "done") not in store.files:
            continue
        where = f"round4/lparam_{lp}/test{t}_seed{s}"
        records = [json.loads(line) for line in store.read((lp, t, s, kind)).splitlines() if line.strip()]
        check_run(records, lp, t, s, where)
        val, test, step = select(records, t)
        runs[(lp, t, s)] = {"val": val, "test": test, "step": step,
                            "mem_gb": max(r.get("mem_gb", 0) for r in records)}
    return runs


def weights(store, key):
    import torch
    sd = torch.load(io.BytesIO(store.read(key + ("model.pkl",), binary=True)), map_location="cpu",
                    weights_only=False)["model_dict"]
    return {k: v.double() for k, v in sd.items()
            if v.is_floating_point() and "running_" not in k and "ext" not in k.lower()}


def distance(a, b):
    return sum(float(((a[k] - b[k]) ** 2).sum()) for k in a) ** 0.5


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
    print(f"{len(runs)} of 24 round-4 runs complete")

    arm = {lp: {(DOMAINS[t], s): v["test"] for (l2, t, s), v in runs.items() if l2 == lp} for lp in ARMS}
    print("\n=== PRIMARY (pre-registered): authors' M2CL, lparam 0.01 - lparam 0, PACS, seeds 0-2")
    res = paired("H1/H2 lparam 0.01 - 0", arm["0.01"], arm["0.0"], margin=MARGIN, report=report)
    report["decision"] = decide(res, 12)
    print("DECISION:", report["decision"])

    print("\n=== Manipulation checks")
    print("gradient share at the ImageNet start: 0.109-0.127 (audit F12; pre-registered: at least 0.05)")
    if not a.no_weights:
        pairs, ref = {}, {}
        cache = {}

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
                  f"(pre-registered: at least {ACTIVE_MIN_PAIRS} of 12)")

    print("\n=== Secondary (descriptive): test accuracy, mean +/- SD over seeds")
    for lp, label in ARMS.items():
        per = {}
        for t, d in enumerate(DOMAINS):
            xs = [runs[(lp, t, s)]["test"] for s in SEEDS if (lp, t, s) in runs]
            if xs:
                per[d] = (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0, len(xs))
        macro = st.mean(v[0] for v in per.values()) if len(per) == 4 else None
        report.setdefault("means", {})[label] = {"per_domain": per, "macro": macro}
        cells = ", ".join(f"{d} {m:.2f}+/-{sd:.2f} (n={n})" for d, (m, sd, n) in per.items())
        print(f"{label}: {cells}" + (f" | macro {macro:.2f}" if macro is not None else ""))
    print(f"paper, ResNet-18 PACS macro: M2-CL {PAPER['M2-CL']}, M2 {PAPER['M2']} (the lparam 0 arm is the same "
          "network without the term, not the authors' separate M2 model)")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1, default=str)
        print("\nwrote", a.json)


if __name__ == "__main__":
    main()
