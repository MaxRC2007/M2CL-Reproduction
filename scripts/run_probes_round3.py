#!/usr/bin/env python
"""Exploratory leakage re-probe of the round-3 checkpoints (active contrastive term; audit F24). Laptop CPU.

  python scripts/run_probes_round3.py --data_root C:/Users/Madhusudan/Downloads/PACS/pacs_data/pacs_data

Stages: features (features_r3/) -> M2/M2-CL probes at every tap -> ERM probes (backbone) -> tables.
Safe to restart: finished feature files are skipped. Progress: results/leakage/round3.log.
Holds off idle sleep while running, without changing any Windows setting; closing the lid still sleeps.
"""
import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
FEAT, OUT = "features_r3", "results/leakage/round3"
TAPS = ["eb1a", "eb1", "eb2a", "eb2", "eb3", "eb4", "eb_concat", "backbone"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--probe_threads", default="2")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs("results/leakage", exist_ok=True)
    log = open("results/leakage/round3.log", "a", buffering=1)
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # continuous | system required

    def say(msg):
        log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    def run(name, args, threads=None):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if threads:
            env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
        say(f"START {name}")
        code = subprocess.run([PY, "-I"] + args, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        say(f"{'DONE' if code == 0 else f'FAILED (exit {code})'} {name}")
        return code == 0

    os.makedirs(FEAT, exist_ok=True)
    src = "features/imagenet__seedNone__all.npz"            # the RGB ImageNet control, extracted on 3 Nov
    if os.path.exists(src) and not os.path.exists(os.path.join(FEAT, os.path.basename(src))):
        shutil.copy2(src, FEAT)
    if not run("features", ["scripts/extract_features.py", "--manifest", "configs/leakage_manifest_round3.yaml",
                            "--data_root", a.data_root, "--out_dir", FEAT]):
        sys.exit(1)
    comps = ["--comparisons", "configs/comparisons_round3.yaml"]
    run("M2 / M2-CL probes, every tap", ["scripts/run_leakage.py", "--features_dir", FEAT, "--out", OUT + "_taps",
                                         "--groups", "r3_m2", "r3_m2cl", "imagenet", "--layers", *TAPS],
        a.probe_threads)
    run("ERM probes, backbone", ["scripts/run_leakage.py", "--features_dir", FEAT, "--out", OUT + "_erm",
                                 "--groups", "r3_erm", "imagenet", "--layers", "backbone"], a.probe_threads)
    run("tables, taps", ["scripts/make_tables.py", "--leakage", OUT + "_taps"] + comps +
        ["--out", "results/tables/round3_taps"])
    run("tables, ERM", ["scripts/make_tables.py", "--leakage", OUT + "_erm"] + comps +
        ["--out", "results/tables/round3_erm"])
    say("ALL STAGES FINISHED")


if __name__ == "__main__":
    main()
