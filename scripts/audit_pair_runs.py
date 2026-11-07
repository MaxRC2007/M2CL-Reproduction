"""Pair the 12 M2 and 12 M2-CL 6-tap runs: test accuracy, validation curves and weight distances.

Reads both zips in memory. best.pt is loaded with weights_only=True; resume checkpoints are not unpickled,
only searched as raw bytes for the stored loader state.
"""
import hashlib
import io
import itertools
import json
import os
import re
import statistics as st
import sys
import zipfile

import torch

OLD, NEW = sys.argv[1], sys.argv[2]
BACKBONE = ("layer0.", "layer1.", "layer2.", "layer3.", "layer4.")
FOLDS = ("art_painting", "cartoon", "photo", "sketch")


def index(path):
    z = zipfile.ZipFile(path)
    runs = {}
    for i in z.infolist():
        m = re.search(r"(?:^|/)runs/([^/]+)/([^/]+)$", i.filename.replace("\\", "/"))
        if m:
            runs.setdefault(m[1], {})[m[2]] = i
    return z, runs


zo, ro = index(OLD)
zn, rn = index(NEW)

print("== Same M2-CL runs in both zips")
for name in sorted(rn):
    if name in ro and "best.pt" in ro[name]:
        a = hashlib.sha256(zo.read(ro[name]["best.pt"])).hexdigest()
        b = hashlib.sha256(zn.read(rn[name]["best.pt"])).hexdigest()
        print(f"  {name}: best.pt {'identical' if a == b else 'DIFFERENT'}")
    elif name in ro:
        print(f"  {name}: old zip has only {sorted(ro[name])}")

print("== Resume checkpoint of m2cl sketch seed0 in the old zip (raw bytes, not unpickled)")
for f in ("latest.pt", "latest_prev.pt"):
    with zipfile.ZipFile(io.BytesIO(zo.read(ro["m2cl_6tap_sketch_seed0"][f]))) as inner:
        pkl = [n for n in inner.namelist() if n.endswith("data.pkl")][0]
        raw = inner.read(pkl)
        print(f"  {f}: stores train_loader state: {b'train_loader' in raw}")


def result(z, runs, name):
    return json.loads(z.read(runs[name]["result.json"]))


def state(z, runs, name):
    ck = torch.load(io.BytesIO(z.read(runs[name]["best.pt"])), map_location="cpu", weights_only=True)
    return ck["model_state"] if "model_state" in ck else ck


m2 = {(f, s): (zo, ro, f"m2_6tap_{f}_seed{s}") for f in FOLDS for s in range(3)}
cl = {(f, s): (zn, rn, f"m2cl_6tap_{f}_seed{s}") for f in FOLDS for s in range(3)}

s0 = state(*m2[("art_painting", 0)])
keys = list(s0)
print("== Model state:", len(keys), "tensors; first", keys[:4], "last", keys[-4:])
prefixes = sorted({k.split(".")[0] for k in keys})
print("   top-level modules:", prefixes)
bb = [k for k in keys if k.startswith(BACKBONE) and s0[k].is_floating_point()
      and not k.endswith(("running_mean", "running_var"))]
print("   learned backbone tensors:", len(bb))
assert len(bb) == 60, bb

imnet_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
imnet = torch.load(imnet_path, map_location="cpu", weights_only=True) if os.path.exists(imnet_path) else None


def dist(a, b, ks):
    return float(torch.sqrt(sum(((a[k].float() - b[k].float()) ** 2).sum() for k in ks)))


def tv(k):  # this model's key -> torchvision resnet18 key
    return k.replace("layer0.0.", "conv1.").replace("layer0.1.", "bn1.")


def from_imagenet(sd):
    ks = [k for k in bb if tv(k) in imnet]
    return dist({k: sd[k] for k in ks}, {k: imnet[tv(k)] for k in ks}, ks), len(ks)


others = [k for k in keys if not k.startswith(BACKBONE) and s0[k].is_floating_point()
          and not k.endswith(("running_mean", "running_var"))]

print()
print("fold seed | M2 test  M2-CL test   diff | best ep | max val gap | backbone dist pair | rest dist pair | M2 from ImageNet")
rows = []
for f in FOLDS:
    for s in range(3):
        rm, rc = result(*m2[(f, s)]), result(*cl[(f, s)])
        vm = [h["val_mean"] for h in rm["history"]]
        vc = [h["val_mean"] for h in rc["history"]]
        sm, sc = state(*m2[(f, s)]), state(*cl[(f, s)])
        d_bb = dist(sm, sc, bb)
        d_rest = dist(sm, sc, others)
        d_im = from_imagenet(sm)[0] if imnet else float("nan")
        diff = rc["test_acc"] - rm["test_acc"]
        rows.append((f, s, rm["test_acc"], rc["test_acc"], diff, rm["best_epoch"], rc["best_epoch"], d_bb, d_rest, d_im))
        print(f"{f:12s} {s} | {rm['test_acc']:6.2f} {rc['test_acc']:6.2f} {diff:+6.2f} | {rm['best_epoch']:2d}/{rc['best_epoch']:2d} | "
              f"{max(abs(a - b) for a, b in zip(vm, vc)):.2f} | {d_bb:.3f} | {d_rest:.3f} | {d_im:.3f}")

print()
d = [r[4] for r in rows]
print(f"mean diff {st.mean(d):+.3f}, higher {sum(x > 0.005 for x in d)}, lower {sum(x < -0.005 for x in d)}, ties {sum(abs(x) <= 0.005 for x in d)}")
same = [r for r in rows if r[5] == r[6]]
print("same best epoch:", [(r[0], r[1], round(r[4], 2), round(r[7], 3)) for r in same])
for f in FOLDS:
    a = [r[2] for r in rows if r[0] == f]
    b = [r[3] for r in rows if r[0] == f]
    print(f"{f:12s} M2 {st.mean(a):.2f}±{st.stdev(a):.2f}  M2-CL {st.mean(b):.2f}±{st.stdev(b):.2f}")
print("macro M2", round(st.mean(st.mean([r[2] for r in rows if r[0] == f]) for f in FOLDS), 2),
      "M2-CL", round(st.mean(st.mean([r[3] for r in rows if r[0] == f]) for f in FOLDS), 2))

# Sign-flip test on the 12 paired differences (exact, 4096 sign patterns)
obs = abs(sum(d))
p = sum(abs(sum(x * y for x, y in zip(c, d))) >= obs - 1e-9 for c in itertools.product([1, -1], repeat=12)) / 4096
print(f"exact paired sign-flip p over 12 pairs: {p:.3f}")

print()
print("== Seed-to-seed backbone distance (M2 vs M2, M2-CL vs M2-CL), same fold")
for f in FOLDS:
    sm = [state(*m2[(f, s)]) for s in range(3)]
    sc = [state(*cl[(f, s)]) for s in range(3)]
    dm = [dist(sm[i], sm[j], bb) for i, j in ((0, 1), (0, 2), (1, 2))]
    dc = [dist(sc[i], sc[j], bb) for i, j in ((0, 1), (0, 2), (1, 2))]
    print(f"{f:12s} M2 {[round(x, 3) for x in dm]}  M2-CL {[round(x, 3) for x in dc]}")
if imnet:
    print("ImageNet tensors compared:", from_imagenet(s0)[1])
