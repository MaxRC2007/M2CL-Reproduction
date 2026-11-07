#!/usr/bin/env python
"""End-to-end check of the whole pipeline on a tiny synthetic PACS (~2-4 min on CPU, offline).

  python scripts/smoke_test.py

Builds fake PACS -> trains ERM and 6-tap M2-CL for 1 epoch -> extracts features ->
runs leakage probes -> builds tables -> runs the unit tests. Everything is written
to a temporary folder; nothing in the repo or your real data is touched.
If this prints "SMOKE TEST PASSED", the real runs will execute without code errors.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


def fake_pacs(root, n=8, size=64):
    rng = np.random.default_rng(0)
    for di, d in enumerate(DOMAINS):
        for ci, c in enumerate(CLASSES):
            os.makedirs(os.path.join(root, d, c), exist_ok=True)
            for i in range(n):
                img = rng.integers(0, 60, (size, size, 3)).astype(np.uint8)
                img[..., di % 3] += 120                              # domain = colour cast
                img[ci * 8:(ci + 1) * 8, :, :] = 255                 # class = stripe position
                Image.fromarray(img).save(os.path.join(root, d, c, f"{i}.jpg"))


def run(args):
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run([PY] + args, check=True, cwd=ROOT)


def main():
    tmp = tempfile.mkdtemp(prefix="m2cl_smoke_")
    data, runs, feats = (os.path.join(tmp, x) for x in ("PACS", "runs", "features"))
    leak, tables = os.path.join(tmp, "leak"), os.path.join(tmp, "tables")
    fake_pacs(data)
    common = ["--data_root", data, "--out_dir", runs, "--targets", "sketch", "--seeds", "0", "1",
              "--epochs", "1", "--img_size", "64", "--batch_size", "32", "--pretrained", "false",
              "--num_workers", "0"]
    run(["scripts/train_lodo.py", "--model", "erm"] + common)
    run(["scripts/train_lodo.py", "--model", "m2cl", "--arch_taps", "6", "--alpha", "0.003"] + common)
    run(["scripts/train_lodo.py", "--model", "m2", "--arch_taps", "6", "--alpha", "0"] + common)
    run(["scripts/train_lodo.py", "--model", "erm"] + common)  # must skip (already complete)
    man = os.path.join(tmp, "manifest.yaml")
    with open(man, "w") as f:  # 'imagenet' replaced by a fixed random net so the test stays offline
        f.write("conditions:\n  - {group: imagenet, kind: random, seed: 99}\n"
                "  - {group: random, kind: random, seeds: [0, 1]}\n")
    run(["scripts/extract_features.py", "--manifest", man, "--from_runs", runs, "--data_root", data,
         "--out_dir", feats, "--num_workers", "0"])
    run(["scripts/run_leakage.py", "--features_dir", feats, "--out", leak, "--targets", "sketch",
         "--probe_seeds", "0", "1", "--epochs", "3"])
    run(["scripts/make_tables.py", "--runs", runs, "--acc_compare", "m2cl_6tap:erm", "m2cl_6tap:m2_6tap",
         "--leakage", leak, "--comparisons", "configs/comparisons.yaml", "--n_boot", "200", "--out", tables])
    md = open(os.path.join(tables, "tables.md")).read()
    for must in ("Matched-seed accuracy comparison", "noise floor", "random init vs ImageNet",
                 "M2-CL vs M2", "Sanity check"):
        assert must in md, f"missing table: {must}"
    run(["-m", "pytest", "-q", "tests"])
    print(f"\nSMOKE TEST PASSED  (outputs in {tmp})")


if __name__ == "__main__":
    main()
