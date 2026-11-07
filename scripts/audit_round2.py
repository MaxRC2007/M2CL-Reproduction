"""Analyse the round-2 runs: dose-response, learning rate 1e-3, and ERM with the corrected split.

  python scripts/audit_round2.py <Downloads folder> <RGB PACS root> [--grad]

Reads dose.zip, lr.zip, erm.zip, m2_v2.zip and m2cl_v2.zip from the Downloads folder in memory.
--grad also measures the contrastive gradient share of each dose checkpoint on two batches of 128.
"""
import io
import itertools
import json
import os
import re
import statistics as st
import sys
import zipfile

import torch
from scipy import stats

DL, DATA = sys.argv[1], sys.argv[2]
GRAD = "--grad" in sys.argv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
FOLDS = ("art_painting", "cartoon", "photo", "sketch")
BACKBONE = ("layer0.", "layer1.", "layer2.", "layer3.", "layer4.")


class Runs:
    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
        self.files = {}
        for i in self.z.infolist():
            n = i.filename.replace("\\", "/")
            m = re.search(r"([^/]+)/([^/]+)$", n)
            if m:
                key = n.rsplit("/", 1)[0]
                self.files.setdefault(key, {})[m[2]] = i

    def find(self, pattern):
        hits = [k for k in self.files if re.search(pattern, k)]
        assert len(hits) == 1, (pattern, hits)
        return hits[0]

    def result(self, pattern):
        return json.loads(self.z.read(self.files[self.find(pattern)]["result.json"]))

    def state(self, pattern):
        ck = torch.load(io.BytesIO(self.z.read(self.files[self.find(pattern)]["best.pt"])), map_location="cpu", weights_only=True)
        return ck.get("model_state", ck)


dose, lr, erm = Runs(f"{DL}/dose.zip"), Runs(f"{DL}/lr.zip"), Runs(f"{DL}/erm.zip")
m2, m2cl = Runs(f"{DL}/m2_v2.zip"), Runs(f"{DL}/m2cl_v2.zip")
imnet = torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"), map_location="cpu", weights_only=True)


def bb_keys(sd):
    return [k for k in sd if k.startswith(BACKBONE) and sd[k].is_floating_point() and not k.endswith(("running_mean", "running_var"))]


def dist(a, b):
    ks = bb_keys(a)
    assert len(ks) == 60
    return float(torch.sqrt(sum(((a[k].float() - b[k].float()) ** 2).sum() for k in ks)))


def tv(k):
    return k.replace("layer0.0.", "conv1.").replace("layer0.1.", "bn1.")


def from_imagenet(sd):
    return float(torch.sqrt(sum(((sd[k].float() - imnet[tv(k)].float()) ** 2).sum() for k in bb_keys(sd))))


# ---------------- A. dose-response ----------------
print("== A. Dose-response, Art held out, seed 0 (lr 5e-4, tau 1)")
m2_art = m2.result(r"m2_6tap_art_painting_seed0$")
m2_sd = m2.state(r"m2_6tap_art_painting_seed0$")
cl3_sd = m2cl.state(r"m2cl_6tap_art_painting_seed0$")
cl3 = m2cl.result(r"m2cl_6tap_art_painting_seed0$")
rows = [("0 (M2)", m2_art, m2_sd), ("0.003", cl3, cl3_sd)]
states = {}
for a in ("0.01", "0.1", "0.3", "1.0"):
    pat = rf"alpha_{re.escape(a)}/m2cl_6tap_art_painting_seed0$"
    rows.append((a, dose.result(pat), dose.state(pat)))
    states[a] = rows[-1][2]
print("alpha | test | best val | best ep | train acc last | cl first -> last | CE last | backbone dist to M2 | from ImageNet")
for a, r, sd in rows:
    h = r["history"]
    alpha = r["config"]["alpha"]
    ce_last = h[-1]["loss"] - alpha * h[-1]["cl_loss"]
    print(f"{a:7s} | {r['test_acc']:.2f} | {r['best_val']:.2f} | {r['best_epoch']:2d} | {h[-1]['train_acc']:.1f} | "
          f"{h[0]['cl_loss']:.1f} -> {h[-1]['cl_loss']:.1f} | {ce_last:.4f} | {dist(sd, m2_sd):.3f} | {from_imagenet(sd):.3f}")
m2_seeds = [m2.state(rf"m2_6tap_art_painting_seed{s}$") for s in range(3)]
print("M2 Art seed-to-seed backbone distance:", [round(dist(m2_seeds[i], m2_seeds[j]), 3) for i, j in ((0, 1), (0, 2), (1, 2))])

if GRAD:
    from m2cl.data import make_lodo_loaders
    from m2cl.losses import m2cl_contrastive_loss
    from m2cl.models import build_model
    torch.manual_seed(0)
    loaders, _ = make_lodo_loaders(DATA, "art_painting", "baseline", 224, 128, 0.2, 0, 0)
    batches = []
    for i, (x, y, _) in enumerate(loaders["train"]):
        batches.append((x, y))
        if i == 1:
            break

    def gn(loss, ps):
        g = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
        return float(sum((t ** 2).sum() for t in g if t is not None) ** 0.5)

    print("\nGradient share alpha*|grad CL|/|grad CE| at each best checkpoint (2 batches of 128, train mode)")
    for name, sd, alpha in [("0.003", cl3_sd, 0.003)] + [(a, states[a], float(a)) for a in states]:
        model = build_model("m2cl", 7, 6, False)
        model.load_state_dict(sd)
        model.train()
        bb = [p for n, p in model.named_parameters() if n.startswith("layer")]
        eb = [p for n, p in model.named_parameters() if n.startswith("eb")]
        out = []
        for x, y in batches:
            logits, reps = model(x)
            cl = m2cl_contrastive_loss(reps, y, 1.0, "sum")
            ce = torch.nn.functional.cross_entropy(logits, y)
            out.append((alpha * gn(cl, bb) / gn(ce, bb), alpha * gn(cl, eb) / gn(ce, eb), cl.item()))
        print(f"  alpha {name}: backbone {[round(o[0], 3) for o in out]}, blocks {[round(o[1], 3) for o in out]}, "
              f"loss {[round(o[2], 1) for o in out]}", flush=True)


# ---------------- B. learning rate ----------------
print("\n== B. M2, seed 0: lr 1e-3 vs 5e-4")
d_lr = []
for f in FOLDS:
    new, old = lr.result(rf"m2_6tap_{f}_seed0$"), m2.result(rf"m2_6tap_{f}_seed0$")
    sd_new, sd_old = lr.state(rf"m2_6tap_{f}_seed0$"), m2.state(rf"m2_6tap_{f}_seed0$")
    d_lr.append(new["test_acc"] - old["test_acc"])
    print(f"{f:12s} 5e-4 {old['test_acc']:.2f} (ep {old['best_epoch']}) -> 1e-3 {new['test_acc']:.2f} (ep {new['best_epoch']}) "
          f"diff {d_lr[-1]:+.2f} | best val {old['best_val']:.2f} -> {new['best_val']:.2f} | "
          f"final CE {old['history'][-1]['loss']:.4f} -> {new['history'][-1]['loss']:.4f} | from ImageNet {from_imagenet(sd_old):.3f} -> {from_imagenet(sd_new):.3f}")
seed0_old = st.mean(m2.result(rf"m2_6tap_{f}_seed0$")["test_acc"] for f in FOLDS)
seed0_new = st.mean(lr.result(rf"m2_6tap_{f}_seed0$")["test_acc"] for f in FOLDS)
print(f"macro, seed 0: {seed0_old:.2f} -> {seed0_new:.2f} ({seed0_new - seed0_old:+.2f}); mean diff {st.mean(d_lr):+.2f}")
m2_sd_fold = {f: st.stdev(m2.result(rf"m2_6tap_{f}_seed{s}$")["test_acc"] for s in range(3)) for f in FOLDS}
print("M2 seed SD per fold at 5e-4:", {f: round(v, 2) for f, v in m2_sd_fold.items()})


# ---------------- C. ERM ----------------
print("\n== C. ERM, corrected split")
table = {}
for f in FOLDS:
    for s in range(3):
        table[("erm", f, s)] = erm.result(rf"erm_{f}_seed{s}$")["test_acc"]
        table[("m2", f, s)] = m2.result(rf"m2_6tap_{f}_seed{s}$")["test_acc"]
        table[("m2cl", f, s)] = m2cl.result(rf"m2cl_6tap_{f}_seed{s}$")["test_acc"]
for model in ("erm", "m2", "m2cl"):
    means = {f: st.mean(table[(model, f, s)] for s in range(3)) for f in FOLDS}
    sds = {f: st.stdev(table[(model, f, s)] for s in range(3)) for f in FOLDS}
    print(f"{model:5s} " + "  ".join(f"{f[:6]} {means[f]:.2f}±{sds[f]:.2f}" for f in FOLDS) + f"  macro {st.mean(means.values()):.2f}")
print("ERM per seed:", {f: [round(table[('erm', f, s)], 2) for s in range(3)] for f in FOLDS})


def paired(a, b):
    d = [table[(a, f, s)] - table[(b, f, s)] for f in FOLDS for s in range(3)]
    n, m, sd = len(d), st.mean(d), st.stdev(d)
    se = sd / n ** 0.5
    t95 = stats.t.ppf(0.975, n - 1)
    p_flip = sum(abs(sum(x * y for x, y in zip(c, d))) >= abs(sum(d)) - 1e-9 for c in itertools.product([1, -1], repeat=n)) / 2 ** n
    per_fold = {f: round(st.mean(table[(a, f, s)] - table[(b, f, s)] for s in range(3)), 2) for f in FOLDS}
    print(f"{a} - {b}: mean {m:+.2f}, 95% CI {m - t95 * se:+.2f} to {m + t95 * se:+.2f}, sign-flip p {p_flip:.3f}, "
          f"higher {sum(x > 0.005 for x in d)}/12; per fold {per_fold}")


paired("m2", "erm")
paired("m2cl", "erm")
