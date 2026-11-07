"""Pre-registered analysis of round 3 (docs/PREREGISTRATION.md). Fixed before any round-3 result exists.

  python scripts/analyze_round3.py <zip or folder> [<zip or folder> ...] [--data_root PACS] [--grad] [--json out.json]

Give every zip or folder that holds the runs: the round-3 download(s), lr.zip (M2 lr 1e-3 seed 0),
and m2_v2.zip (M2 lr 5e-4, for S2). Runs are found by their folder path under m2cl_v2/:
  lr/lr_0.001/m2_6tap_*            M2,    lr 1e-3   (PACS)
  round3/m2cl/m2cl_6tap_*          M2-CL, alpha 0.07, lr 1e-3 (PACS)
  round3/erm/erm_*                 ERM,   lr 1e-3   (PACS)
  round3_vlcs/{m2,m2cl,erm}/...    the same three on VLCS
  runs/m2_6tap_*                   M2,    lr 5e-4   (PACS, rounds 1-2)
"""
import argparse
import io
import itertools
import json
import os
import re
import statistics as st
import sys
import zipfile

from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
PACS = ("art_painting", "cartoon", "photo", "sketch")
VLCS = ("Caltech101", "LabelMe", "SUN09", "VOC2007")
MARGIN = 0.71  # Ballas & Diou (2024), Table I, ResNet-18 PACS: 83.54 - 82.83
PRIMARY_SEEDS = (0, 1, 2)
ROBUST_SEEDS = (3, 4)
INERT_DIST = 0.033  # largest same-seed backbone distance at alpha 0.003 (AUDIT F9)

EXPECTED = {  # pre-registered settings every run must have
    "m2": dict(model="m2", arch_taps=6, alpha=0.0, tau=1.0, lr=0.001, batch_size=128, epochs=30, val_fraction=0.2),
    "m2cl": dict(model="m2cl", arch_taps=6, alpha=0.07, tau=1.0, lr=0.001, batch_size=128, epochs=30, val_fraction=0.2),
    "erm": dict(model="erm", lr=0.001, batch_size=128, epochs=30, val_fraction=0.2),
    "m2_old": dict(model="m2", arch_taps=6, alpha=0.0, lr=0.0005),
}


class Store:
    """Every run folder (holding result.json) across zips and folders, keyed by its path."""

    def __init__(self, sources):
        self.runs = {}
        for src in sources:
            if zipfile.is_zipfile(src):
                z = zipfile.ZipFile(src)
                for i in z.infolist():
                    n = i.filename.replace("\\", "/")
                    if n.endswith("/result.json"):
                        self._add(n[: -len("/result.json")], ("zip", z, n[: -len("result.json")]))
            else:
                for root, _, files in os.walk(src):
                    if "result.json" in files:
                        self._add(root.replace("\\", "/"), ("dir", None, root.replace("\\", "/") + "/"))

    def _add(self, key, where):
        self.runs.setdefault(key, where)

    def _find(self, pattern):
        hits = [k for k in self.runs if re.search(pattern + "$", k)]
        if len(hits) > 1:
            same = {json.dumps(self.result_at(k), sort_keys=True) for k in hits}
            if len(same) > 1:
                raise ValueError(f"{pattern}: different runs in {hits}")
        return hits[0] if hits else None

    def result_at(self, key):
        kind, z, prefix = self.runs[key]
        raw = z.read(prefix + "result.json") if kind == "zip" else open(prefix + "result.json", "rb").read()
        return json.loads(raw)

    def result(self, pattern):
        k = self._find(pattern)
        return None if k is None else self.result_at(k)

    def state(self, pattern):
        import torch
        kind, z, prefix = self.runs[self._find(pattern)]
        raw = z.read(prefix + "best.pt") if kind == "zip" else open(prefix + "best.pt", "rb").read()
        ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        return ck.get("model_state", ck)


def check(r, kind, where):
    c = r["config"]
    bad = {k: (c.get(k), v) for k, v in EXPECTED[kind].items() if c.get(k) != v}
    if bad or r.get("status") != "complete":
        raise SystemExit(f"{where}: not the pre-registered settings or not complete: {bad}")


def collect(store, pattern_fmt, kind, domains, seeds):
    out = {}
    for f in domains:
        for s in seeds:
            pat = pattern_fmt.format(f=re.escape(f), s=s)
            r = store.result(pat)
            if r is not None:
                check(r, kind, pat)
                out[(f, s)] = r["test_acc"]
    return out


def paired(name, a, b, margin=None, report=None):
    keys = sorted(set(a) & set(b))
    if not keys:
        print(f"{name}: no complete pairs yet")
        return None
    d = [a[k] - b[k] for k in keys]
    n, m = len(d), st.mean(d)
    res = {"n": n, "mean": m, "diffs": {f"{k[0]}/seed{k[1]}": x for k, x in zip(keys, d)}}
    if n >= 2:
        sd = st.stdev(d)
        se = sd / n ** 0.5 if sd > 0 else 1e-12
        t95 = stats.t.ppf(0.975, n - 1)
        res.update(sd=sd, ci95=(m - t95 * se, m + t95 * se), p_t=float(2 * stats.t.sf(abs(m) / se, n - 1)))
        if n <= 20:
            res["p_signflip"] = sum(abs(sum(x * y for x, y in zip(c, d))) >= abs(sum(d)) - 1e-9
                                    for c in itertools.product([1, -1], repeat=n)) / 2 ** n
        if margin is not None:
            res["p_tost"] = float(max(stats.t.sf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1)))
    per_fold = {}
    for f in sorted({k[0] for k in keys}):
        per_fold[f] = st.mean(a[k] - b[k] for k in keys if k[0] == f)
    res["per_fold"] = per_fold
    line = f"{name}: n={n} mean {m:+.2f} pp"
    if "ci95" in res:
        line += f", 95% CI {res['ci95'][0]:+.2f} to {res['ci95'][1]:+.2f}, t p={res['p_t']:.4f}"
    if "p_signflip" in res:
        line += f", sign-flip p={res['p_signflip']:.4f}"
    if "p_tost" in res:
        line += f", TOST(+/-{margin}) p={res['p_tost']:.4f}"
    print(line)
    print("   per fold:", {k: round(v, 2) for k, v in per_fold.items()})
    if report is not None:
        report[name] = res
    return res


def decide(res, n_planned):
    if res is None or res["n"] < n_planned:
        return f"INCOMPLETE: {0 if res is None else res['n']} of {n_planned} pairs; no decision"
    effect = res.get("p_signflip", 1) < 0.05
    equiv = res.get("p_tost", 1) < 0.05
    if effect and equiv:
        return "EFFECT and EQUIVALENCE: a real difference, smaller than the paper's 0.71 pp"
    if effect:
        side = "excludes" if not (res["ci95"][0] <= MARGIN <= res["ci95"][1]) else "includes"
        return f"EFFECT: sign-flip p < 0.05, mean {res['mean']:+.2f} pp; the 95% CI {side} 0.71"
    if equiv:
        return "EQUIVALENCE: no effect as large as the paper's claimed 0.71 pp gain"
    return "INCONCLUSIVE: neither an effect nor equivalence at the pre-registered thresholds"


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--data_root")
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    store = Store(a.sources)
    report = {}

    print("=== PRIMARY (pre-registered): PACS, M2-CL alpha 0.07 vs M2, lr 1e-3, seeds 0-2")
    m2 = collect(store, r"lr/lr_0\.001/m2_6tap_{f}_seed{s}", "m2", PACS, PRIMARY_SEEDS)
    cl = collect(store, r"round3/m2cl/m2cl_6tap_{f}_seed{s}", "m2cl", PACS, PRIMARY_SEEDS)
    prim = paired("H1/H2 M2-CL - M2", cl, m2, MARGIN, report)
    report["decision"] = decide(prim, 12)
    print("DECISION:", report["decision"])

    print("\n=== Manipulation checks")
    if prim is not None:
        dists = {}
        for (f, s) in sorted(set(m2) & set(cl)):
            sa = store.state(rf"lr/lr_0\.001/m2_6tap_{re.escape(f)}_seed{s}")
            sb = store.state(rf"round3/m2cl/m2cl_6tap_{re.escape(f)}_seed{s}")
            ks = [k for k in sa if k.startswith(("layer0.", "layer1.", "layer2.", "layer3.", "layer4."))
                  and sa[k].is_floating_point() and not k.endswith(("running_mean", "running_var"))]
            dists[f"{f}/seed{s}"] = float(sum(((sa[k].float() - sb[k].float()) ** 2).sum() for k in ks) ** 0.5)
        above = sum(v > INERT_DIST for v in dists.values())
        report["manipulation_distance"] = {"distances": dists, "above_inert": above, "n": len(dists)}
        print(f"backbone distance M2-CL vs M2: {above} of {len(dists)} pairs above the inert level {INERT_DIST} "
              f"(pre-registered: at least 10 of 12); range {min(dists.values()):.3f}-{max(dists.values()):.3f}")
    if a.grad and a.data_root and store._find(r"round3/m2cl/m2cl_6tap_art_painting_seed0"):
        import torch
        from m2cl.data import make_lodo_loaders
        from m2cl.losses import m2cl_contrastive_loss
        from m2cl.models import build_model
        torch.manual_seed(0)
        loaders, _ = make_lodo_loaders(a.data_root, "art_painting", "baseline", 224, 128, 0.2, 0, 0)
        model = build_model("m2cl", 7, 6, False)
        model.load_state_dict(store.state(r"round3/m2cl/m2cl_6tap_art_painting_seed0"))
        model.train()
        bb = [p for n, p in model.named_parameters() if n.startswith("layer")]
        shares = []
        for i, (x, y, _) in enumerate(loaders["train"]):
            logits, reps = model(x)
            g = lambda loss: float(sum((t ** 2).sum() for t in torch.autograd.grad(loss, bb, retain_graph=True, allow_unused=True) if t is not None) ** 0.5)
            shares.append(0.07 * g(m2cl_contrastive_loss(reps, y, 1.0, "sum")) / g(torch.nn.functional.cross_entropy(logits, y)))
            if i == 1:
                break
        report["manipulation_grad_share"] = shares
        print(f"gradient share at M2-CL Art seed 0: {[round(s, 3) for s in shares]} (pre-registered: at least 0.05)")

    print("\n=== SECONDARY (Holm-corrected across S2-S4)")
    secondary = []
    print("S1: per-fold differences are listed under the primary result above.")
    m2_old = collect(store, r"runs/m2_6tap_{f}_seed{s}", "m2_old", PACS, PRIMARY_SEEDS)
    secondary.append(("S2 M2 lr 1e-3 - lr 5e-4 (PACS)", paired("S2 M2 lr 1e-3 - lr 5e-4 (PACS)", m2, m2_old, None, report)))
    erm = collect(store, r"round3/erm/erm_{f}_seed{s}", "erm", PACS, PRIMARY_SEEDS)
    secondary.append(("S3 M2 - ERM (PACS, lr 1e-3)", paired("S3 M2 - ERM (PACS, lr 1e-3)", m2, erm, None, report)))
    secondary.append(("S3 M2-CL - ERM (PACS, lr 1e-3)", paired("S3 M2-CL - ERM (PACS, lr 1e-3)", cl, erm, None, report)))
    v_m2 = collect(store, r"round3_vlcs/m2/m2_6tap_{f}_seed{s}", "m2", VLCS, PRIMARY_SEEDS)
    v_cl = collect(store, r"round3_vlcs/m2cl/m2cl_6tap_{f}_seed{s}", "m2cl", VLCS, PRIMARY_SEEDS)
    v_erm = collect(store, r"round3_vlcs/erm/erm_{f}_seed{s}", "erm", VLCS, PRIMARY_SEEDS)
    secondary.append(("S4 VLCS M2-CL - M2", paired("S4 VLCS M2-CL - M2", v_cl, v_m2, MARGIN, report)))
    secondary.append(("S4 VLCS M2 - ERM", paired("S4 VLCS M2 - ERM", v_m2, v_erm, None, report)))
    secondary.append(("S4 VLCS M2-CL - ERM", paired("S4 VLCS M2-CL - ERM", v_cl, v_erm, None, report)))
    tested = [(n, r) for n, r in secondary if r is not None and "p_signflip" in r]
    if tested:
        adj = holm([r["p_signflip"] for _, r in tested])
        report["holm"] = {n: p for (n, _), p in zip(tested, adj)}
        print("Holm-adjusted sign-flip p:", {n: round(p, 4) for n, p in report["holm"].items()})

    print("\n=== Means per method (macro-average over folds, seeds 0-2)")
    for name, runs, doms in (("PACS ERM lr1e-3", erm, PACS), ("PACS M2 lr1e-3", m2, PACS), ("PACS M2-CL lr1e-3", cl, PACS),
                             ("VLCS ERM", v_erm, VLCS), ("VLCS M2", v_m2, VLCS), ("VLCS M2-CL", v_cl, VLCS)):
        per = {f: [runs[(f, s)] for s in PRIMARY_SEEDS if (f, s) in runs] for f in doms}
        if all(len(v) == 3 for v in per.values()):
            means = {f: st.mean(v) for f, v in per.items()}
            sds = {f: st.stdev(v) for f, v in per.items()}
            print(f"{name}: " + ", ".join(f"{f} {means[f]:.2f}+/-{sds[f]:.2f}" for f in doms) + f" | macro {st.mean(means.values()):.2f}")
            report[f"means {name}"] = {"per_fold": means, "sd": sds, "macro": st.mean(means.values())}
        else:
            print(f"{name}: {sum(map(len, per.values()))} of 12 runs")

    print("\n=== D. Robustness, seeds 3-4 (reported separately, never pooled into the primary test)")
    paired("D M2-CL - M2 (PACS, seeds 3-4)", collect(store, r"round3/m2cl/m2cl_6tap_{f}_seed{s}", "m2cl", PACS, ROBUST_SEEDS),
           collect(store, r"lr/lr_0\.001/m2_6tap_{f}_seed{s}", "m2", PACS, ROBUST_SEEDS), MARGIN, report)

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(report, fh, indent=1, default=list)
        print("\nwrote", a.json)


if __name__ == "__main__":
    main()
