#!/usr/bin/env python
"""Re-run the draft's leakage tables on RGB PACS on this laptop; safe to restart at any point.

  python scripts/run_overnight.py --data_root C:/Users/Madhusudan/Downloads/Madhu/Honors/data_pacs/pacs_data/pacs_data \
         --dct_root C:/Users/Madhusudan/Downloads/Madhu/Honors/data_pacs

Stages: features -> backbone probes -> tables -> tap probes -> tables -> checkpoint accuracy.
Finished work is skipped on restart. Progress is in results/leakage/overnight.log.
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
V1 = ["art_painting", "cartoon", "photo"]  # no 4-tap M2 / M2-CL Sketch checkpoints exist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dct_root", required=True, help="only used to identify each checkpoint's training images")
    ap.add_argument("--probe_threads", default="2")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs("results/leakage", exist_ok=True)
    log = open("results/leakage/overnight.log", "a", buffering=1)
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # continuous | system required

    def say(msg):
        log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    def run(name, args, threads=None):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if threads:
            env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
        say(f"START {name}")
        code = subprocess.run([PY] + args, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        say(f"{'DONE' if code == 0 else f'FAILED (exit {code})'} {name}")
        return code == 0

    # the ImageNet features are already extracted on RGB by scripts/audit_dct_data.py
    cached, wanted = "features/audit/imagenet_rgb_n0.npz", "features/imagenet__seedNone__all.npz"
    if os.path.exists(cached) and not os.path.exists(wanted):
        os.makedirs("features", exist_ok=True)
        shutil.copy2(cached, wanted)

    comps = ["--comparisons", "configs/comparisons.yaml"]
    if not run("features", ["scripts/extract_features.py", "--manifest", "configs/leakage_manifest.yaml",
                            "--data_root", a.data_root, "--out_dir", "features"]):
        sys.exit(1)
    if run("backbone probes", ["scripts/run_leakage.py", "--features_dir", "features", "--out",
                               "results/leakage/raw_backbone", "--layers", "backbone"], a.probe_threads):
        run("backbone tables", ["scripts/make_tables.py", "--leakage", "results/leakage/raw_backbone"] + comps +
            ["--out", "results/tables/raw_backbone"])
    if run("tap probes", ["scripts/run_leakage.py", "--features_dir", "features", "--out",
                          "results/leakage/raw_taps_v1", "--targets"] + V1 +
           ["--groups", "m2_4tap_v1", "m2cl_4tap_v1", "--layers", "eb1", "eb2", "eb3", "eb4", "eb_concat", "backbone"],
           a.probe_threads):
        run("tap tables", ["scripts/make_tables.py", "--leakage", "results/leakage/raw_taps_v1"] + comps +
            ["--out", "results/tables/raw_taps_v1"])
    run("checkpoint accuracy", ["scripts/verify_checkpoints.py", "--manifest", "configs/leakage_manifest.yaml",
                                "--rgb_root", a.data_root, "--dct_root", a.dct_root])
    say("ALL STAGES FINISHED")


if __name__ == "__main__":
    main()
