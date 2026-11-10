#!/usr/bin/env python
"""Build every paper table from raw outputs (NumPy/pandas only, no GPU, no torch).

Accuracy tables  (from training runs or a CSV of per-run accuracies):
  python scripts/make_tables.py --runs runs --acc_compare m2cl_6tap:erm m2cl_6tap:m2_6tap --out results/tables
  python scripts/make_tables.py --accuracy_csv results/legacy_v1/paper_accuracy_runs.csv \
                                --acc_compare m2cl_6tap:erm --out results/tables/legacy_v1
Leakage tables   (from scripts/run_leakage.py output):
  python scripts/make_tables.py --leakage results/leakage/raw --comparisons configs/comparisons.yaml --out results/tables
Writes tables.md (all tables, human-readable) and one LaTeX file per table.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from m2cl.splits import ALL_DOMAINS as PACS_DOMAINS  # PACS, then VLCS; absent targets are skipped  # noqa: E402
from m2cl.stats import benjamini_hochberg, paired_bootstrap, sign_flip_test, sign_test  # noqa: E402

NICE = {"art_painting": "Art Painting", "cartoon": "Cartoon", "photo": "Photo", "sketch": "Sketch",
        "Caltech101": "Caltech101", "LabelMe": "LabelMe", "SUN09": "SUN09", "VOC2007": "VOC2007"}


class Out:
    def __init__(self, out_dir):
        self.dir, self.md = out_dir, []
        os.makedirs(out_dir, exist_ok=True)

    def add(self, name, title, df, note=""):
        df = df.copy()
        for col in ("n", "n init", "seed"):
            if col in df.columns and df[col].dtype != object:
                df[col] = df[col].astype("Int64")
        for col in ("p", "p (BH)", "sign test p", "sign-flip p"):
            if col in df.columns:
                df[col] = df[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        self.md += [f"## {title}", "", df.to_markdown(index=False, floatfmt=".2f", missingval=""), ""]
        if note:
            self.md += [f"*{note}*", ""]
        with open(os.path.join(self.dir, f"{name}.tex"), "w") as f:
            f.write(df.to_latex(index=False, float_format="%.2f", escape=True, na_rep="", caption=title, label=f"tab:{name}"))
        df.to_csv(os.path.join(self.dir, f"{name}.csv"), index=False)

    def save(self):
        with open(os.path.join(self.dir, "tables.md"), "w") as f:
            f.write("# Generated tables\n\nDo not edit by hand: regenerate with scripts/make_tables.py\n\n"
                    + "\n".join(self.md))
        print(f"wrote {self.dir}/tables.md")


# ───────────────────────── accuracy ─────────────────────────
def load_runs(dirs):
    rows = []
    for d in dirs:
        for r in glob.glob(os.path.join(d, "*", "result.json")):
            res = json.load(open(r))
            if res.get("status") != "complete":
                continue
            c = res["config"]
            g = c["model"] + (f"_{c['arch_taps']}tap" if c["model"] != "erm" else "")
            rows.append({"group": g, "target": c["target"], "seed": c["seed"], "test_acc": res["test_acc"],
                         "best_epoch": res.get("best_epoch")})
    return pd.DataFrame(rows)


def acc_tables(df, pairs, out):
    for g, sub in df.groupby("group"):
        seeds = sorted(sub.seed.unique())
        rows = []
        for t in PACS_DOMAINS:
            s = sub[sub.target == t].set_index("seed").test_acc
            if s.empty:
                continue
            rows.append({"Domain": NICE[t], **{f"Seed {k}": s.get(k, np.nan) for k in seeds},
                         "Mean": s.mean(), "SD": s.std(ddof=1) if len(s) > 1 else np.nan, "n": len(s)})
        t_df = pd.DataFrame(rows)
        if len(t_df) == 4:
            t_df.loc[len(t_df)] = {"Domain": "Macro-average", "Mean": t_df.Mean.mean()}
        out.add(f"acc_{g}", f"Target-domain test accuracy (%), {g}", t_df,
                "SD = sample standard deviation (ddof=1) across seeds.")
    for pair in pairs:
        a, b = pair.split(":")
        A, B = df[df.group == a], df[df.group == b]
        rows, macro_a, macro_b = [], [], []
        for t in PACS_DOMAINS:
            sa, sb = A[A.target == t].set_index("seed").test_acc, B[B.target == t].set_index("seed").test_acc
            common = sorted(set(sa.index) & set(sb.index))
            if not common:
                continue
            d = [sa[k] - sb[k] for k in common]
            macro_a.append(sa[common].mean()); macro_b.append(sb[common].mean())
            rows.append({"Domain": NICE[t], a: sa[common].mean(), b: sb[common].mean(), "Delta": np.mean(d),
                         **{f"d(seed {k})": x for k, x in zip(common, d)},
                         "sign test p": sign_test(d), "sign-flip p": sign_flip_test(d), "n": len(common)})
        if not rows:
            continue
        t_df = pd.DataFrame(rows)
        if len(rows) == 4:
            t_df.loc[len(t_df)] = {"Domain": "Macro-average", a: np.mean(macro_a), b: np.mean(macro_b),
                                   "Delta": np.mean(macro_a) - np.mean(macro_b)}
        n = int(t_df.n.min())
        out.add(f"acc_{a}_vs_{b}", f"Matched-seed accuracy comparison: {a} minus {b} (pp)", t_df,
                f"Seeds matched by source train/val partition. With n={n} seeds the smallest attainable "
                f"two-sided p is {2/2**n:.3f}; differences are descriptive.")


# ───────────────────────── leakage ─────────────────────────
def leak_load(d):
    S = json.load(open(os.path.join(d, "summary.json")))
    Z = np.load(os.path.join(d, "correctness.npz"))
    return S, Z


def per_sample(Z, T, key, L):
    k = f"{T}|{key}|{L}"
    return Z[k].mean(0) if k in Z.files else None  # mean over probe seeds


def leak_tables(S, Z, comps, out, n_boot):
    # 1) probe noise floor: same features, different probe seeds
    rows = []
    for T, ts in S["targets"].items():
        for key, c in ts["conditions"].items():
            for L, ls in c["layers"].items():
                rows.append({"group": c["group"], "layer": L, "target": T,
                             "sd": ls["domain"]["acc_sd_probe_seeds"], "range": ls["domain"]["acc_range_probe_seeds"]})
    if rows:
        df = pd.DataFrame(rows).groupby(["group", "layer"]).agg(
            mean_sd_pp=("sd", "mean"), max_range_pp=("range", "max"), n=("sd", "size")).reset_index()
        out.add("leak_noise_floor", "Probe-training noise floor (domain probe, same frozen features, "
                f"{len(S['config']['probe_seeds'])} probe seeds)", df,
                "max_range_pp = largest max-min accuracy spread across probe seeds for any target.")

    # 2) architectural bias / pretraining amplification (Table II style), backbone layer
    rows = []
    for T in PACS_DOMAINS:
        ts = S["targets"].get(T)
        if not ts:
            continue
        rnd = [c["layers"]["backbone"]["domain"]["acc_mean"] for c in ts["conditions"].values()
               if c["group"] == "random" and "backbone" in c["layers"]]
        img = [c["layers"]["backbone"]["domain"]["acc_mean"] for c in ts["conditions"].values()
               if c["group"] == "imagenet" and "backbone" in c["layers"]]
        if not rnd or not img:
            continue
        maj = ts["majority_baseline"]
        rows.append({"Held-out": NICE[T], "Random-init mean": np.mean(rnd),
                     "Random-init SD": np.std(rnd, ddof=1) if len(rnd) > 1 else np.nan, "n init": len(rnd),
                     "Majority": maj, "Arch. bias": np.mean(rnd) - maj, "ImageNet": img[0],
                     "Pretrain amp.": img[0] - np.mean(rnd)})
    if rows:
        out.add("leak_arch_bias", "Domain-probe accuracy (%) at the backbone feature: random init vs ImageNet",
                pd.DataFrame(rows), "Each random-init value is first averaged over probe seeds; "
                "SD is across weight-initialisation seeds (ddof=1).")

    # 3) comparison families with BH correction inside each family
    for fam in comps or []:
        layers = fam["layers"] if isinstance(fam["layers"], list) else [fam["layers"]]
        rows = []
        for T in PACS_DOMAINS:
            ts = S["targets"].get(T)
            if not ts:
                continue
            conds = ts["conditions"]
            for A, B in fam["pairs"]:
                ka = sorted([k for k, c in conds.items() if c["group"] == A], key=lambda k: str(conds[k]["seed"]))
                kb = [k for k, c in conds.items() if c["group"] == B]
                for k_a in ka:
                    sa = conds[k_a]["seed"]
                    match = [k for k in kb if conds[k]["seed"] == sa] or (kb if len(kb) == 1 else [])
                    if not match:
                        continue
                    for L in layers:
                        ca, cb = per_sample(Z, T, k_a, L), per_sample(Z, T, match[0], L)
                        if ca is None or cb is None:
                            continue
                        r = paired_bootstrap(ca, cb, n_boot=n_boot, seed=0)
                        rows.append({"Domain": NICE[T], "A": A, "B": B, "seed": sa, "layer": L,
                                     "acc A": r["acc_a"], "acc B": r["acc_b"], "Delta": r["delta_pp"],
                                     "CI lo": r["ci_lo_pp"], "CI hi": r["ci_hi_pp"], "p": r["p"]})
        if not rows:
            print(f"[warn] family {fam['name']}: no matching conditions"); continue
        df = pd.DataFrame(rows)
        df["p (BH)"], df["sig (BH, q=.05)"] = benjamini_hochberg(df.p.values)
        out.add(f"leak_{fam['name']}", fam.get("title", fam["name"]), df,
                f"Paired bootstrap ({n_boot} resamples) on per-sample correctness averaged over probe seeds; "
                f"Benjamini-Hochberg across the {len(df)} comparisons in this table.")

    # 4) class-probe sanity check
    rows = []
    for T, ts in S["targets"].items():
        for key, c in ts["conditions"].items():
            b = c["layers"].get("backbone", {})
            if "class_target_acc" in b:
                rows.append({"group": c["group"], "target": T, "class acc source-test": b["class_source_test_acc"],
                             "class acc target": b["class_target_acc"]})
    if rows:
        df = pd.DataFrame(rows).groupby(["group", "target"]).mean(numeric_only=True).reset_index()
        out.add("leak_class_sanity", "Sanity check: linear CLASS probe on the same backbone features (%)", df,
                "ImageNet features should give high class accuracy; near-chance (14%) values mean a label bug.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--accuracy_csv")
    ap.add_argument("--acc_compare", nargs="*", default=[])
    ap.add_argument("--leakage")
    ap.add_argument("--comparisons")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    out = Out(a.out)
    frames = [load_runs(a.runs)] if a.runs else []
    if a.accuracy_csv:
        frames.append(pd.read_csv(a.accuracy_csv))
    if frames:
        df = pd.concat(frames, ignore_index=True)
        if not df.empty:
            acc_tables(df, a.acc_compare, out)
    if a.leakage:
        import yaml
        S, Z = leak_load(a.leakage)
        comps = yaml.safe_load(open(a.comparisons))["families"] if a.comparisons else []
        leak_tables(S, Z, comps, out, a.n_boot)
    out.save()


if __name__ == "__main__":
    main()
