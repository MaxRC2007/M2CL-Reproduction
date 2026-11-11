#!/usr/bin/env python
"""Round 5: the authors' own M2-CL code on VLCS, lparam 0.01 vs 0 (docs/PREREGISTRATION_ROUND5.md).

  python scripts/run_round5_authors.py --out ~/m2cl_round5 --data_dir ~/data --smoke         # first: short test
  python scripts/run_round5_authors.py --out ~/m2cl_round5 --data_dir ~/data --parallel 2    # then: all 24 runs

Round 4's runner with the dataset changed. It reuses the authors' code that round 4 cloned (commit checked, never
modified) and the verified VLCS copy that round 3 used (image counts checked; downloaded from the same mirror if
missing), then runs `python -m domainbed.scripts.train` once per run. Run it inside tmux. Safe to stop and start
again: finished runs are kept; an unfinished run (the code cannot resume) is deleted and restarted from scratch.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_round4_authors as r4  # noqa: E402  (round 4's helpers, unchanged)
from run_part_c import EXPECTED as VLCS_COUNTS, VLCS_T, check_gpu, image_counts, log  # noqa: E402

COMMIT = r4.COMMIT
RESNET18_URL = r4.RESNET18_URL
LPARAMS, SEEDS, TEST_ENVS = r4.LPARAMS, r4.SEEDS, r4.TEST_ENVS
N_RUNS = len(LPARAMS) * len(SEEDS) * len(TEST_ENVS)
N_STEPS_FULL = 5001
FIXED = ["--dataset", "VLCS", "--algorithm", "M2CL", "--hparams_seed", "0", "--hparams", '{"resnet18": true}',
         "--lr", "0.001", "--temp", "1.0"]


def find_vlcs(base):
    """The folder that DomainBed gets as --data_dir: it must contain VLCS/ with exactly the four domains and
    DomainBed's image counts (DomainBed treats every sub-folder of VLCS/ as a domain)."""
    if not os.path.isdir(base):
        return None
    for r, d, _ in os.walk(base):
        if "VLCS" not in d:
            continue
        vlcs = os.path.join(r, "VLCS")
        subdirs = sorted(e.name for e in os.scandir(vlcs) if e.is_dir())
        if subdirs != sorted(VLCS_T):
            log(f"{vlcs} has folders {subdirs}; DomainBed needs exactly {sorted(VLCS_T)}")
            continue
        counts = image_counts(vlcs)
        if counts == VLCS_COUNTS:
            return r
        log(f"{vlcs} has the wrong image counts {counts}; expected {VLCS_COUNTS}")
    return None


def get_vlcs(data_dir):
    root = find_vlcs(data_dir)
    if root:
        return root
    import run_part_c
    got = run_part_c.download_vlcs(data_dir)          # same verified mirror and checks as round 3
    if os.path.basename(got.rstrip("/\\")) != "VLCS":
        sys.exit(f"The VLCS download landed in {got}, not in a folder named VLCS, which DomainBed needs. "
                 "Send this line to the project owner.")
    return os.path.dirname(got.rstrip("/\\"))


def run_dir(out, lparam, test_env, seed):
    return os.path.join(out, "round5", f"lparam_{lparam}", f"test{test_env}_seed{seed}")


def train_command(lparam, test_env, seed, data_root, out_dir, extra=()):
    return [sys.executable, "-u", "-m", "domainbed.scripts.train", "--data_dir", data_root, *FIXED,
            "--lparam", lparam, "--test_envs", str(test_env), "--trial_seed", str(seed), "--seed", str(seed),
            "--output_dir", out_dir, *extra]


def acquire_lock(out):
    path = os.path.join(out, ".round5.lock")
    if os.path.exists(path):
        with open(path) as f:
            pid = int(f.read().strip() or 0)
        alive = False
        if pid and os.name == "posix":
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                pass
        if alive:
            sys.exit(f"Another run_round5_authors.py (process {pid}) is already using {out}. Stop it first, or wait.")
    with open(path, "w") as f:
        f.write(str(os.getpid()))
    return path


def record_truncated(vlcs, out):
    import run_part_c
    run_part_c.record_truncated(vlcs, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="results folder to send back (e.g. ~/m2cl_round5)")
    ap.add_argument("--data_dir", required=True, help="where VLCS and the authors' code are, or will be put")
    ap.add_argument("--parallel", type=int, default=1, help="runs at once on the one GPU (decide from the smoke test)")
    ap.add_argument("--smoke", action="store_true", help="one short run (101 steps) in <out>_smoke, to test the setup")
    ap.add_argument("--no_gpu_check", action="store_true", help=argparse.SUPPRESS)  # testing this script only
    a = ap.parse_args()
    out, data_dir = os.path.abspath(os.path.expanduser(a.out)), os.path.abspath(os.path.expanduser(a.data_dir))
    assert a.parallel >= 1
    os.makedirs(data_dir, exist_ok=True)

    import torch
    gpu = {} if a.no_gpu_check else check_gpu()
    missing = [p for p, v in r4.versions().items() if v is None and p in ("wilds", "timm")]
    if missing:
        sys.exit(f"The authors' code imports {missing}. Install them first: pip install -r requirements_round4.txt")
    authors = r4.authors_code(data_dir)
    data_root = get_vlcs(data_dir)
    vlcs = os.path.join(data_root, "VLCS")
    log(f"Authors' code: {authors} (commit {COMMIT[:7]}, unmodified) | VLCS: {vlcs} "
        f"| images per domain: {image_counts(vlcs)}")
    torch.hub.load_state_dict_from_url(RESNET18_URL, progress=False)

    if a.smoke:
        smoke_out = out.rstrip("/\\") + "_smoke"
        shutil.rmtree(smoke_out, ignore_errors=True)
        os.makedirs(smoke_out)
        d = os.path.join(smoke_out, "round5", "lparam_0.01", "test0_seed0")
        job = {"name": "smoke_lparam0.01_test0_seed0", "dir": d,
               "cmd": train_command("0.01", 0, 0, data_root, d, ("--steps", "101", "--checkpoint_freq", "50"))}
        t0 = time.time()
        failed = r4.schedule([job], 1, smoke_out, authors)
        if failed:
            sys.exit("The smoke test failed; see the lines above. Send them to the project owner.")
        with open(os.path.join(d, "results.jsonl")) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        step_s = recs[-1]["step_time"]
        total_min = (time.time() - t0) / 60
        eval_min = max(0.0, total_min - 101 * step_s / 60) / len(recs)
        per_run_h = (N_STEPS_FULL * step_s / 60 + 18 * eval_min) / 60
        mem = max(r["mem_gb"] for r in recs)
        log(f"Smoke test passed: {step_s:.2f} s per step, about {eval_min:.1f} min per evaluation, peak GPU memory "
            f"{mem:.1f} GB. A full run is {N_STEPS_FULL} steps and 18 evaluations: about {per_run_h:.1f} h per run, "
            f"{per_run_h * N_RUNS:.0f} h for all {N_RUNS} one at a time. Use --parallel 2 if 2 x {mem:.1f} GB fits "
            "in the GPU's memory with room to spare. The smoke run is in " + smoke_out + ": a test only, never "
            "analysed.")
        return

    os.makedirs(os.path.join(out, "env"), exist_ok=True)
    lock = acquire_lock(out)
    try:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        cpu = platform.processor()
        try:
            with open("/proc/cpuinfo") as f:
                cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
        except (OSError, StopIteration):
            pass
        env = {"platform": "institute", "round": 5, "started_utc": stamp, **gpu, "packages": r4.versions(),
               "python": platform.python_version(), "os": platform.platform(), "cpu": cpu,
               "cpu_count": os.cpu_count(), "parallel": a.parallel, "authors_commit": COMMIT,
               "authors_code": authors, "vlcs": vlcs, "vlcs_counts": image_counts(vlcs),
               "command_fixed_args": FIXED}
        with open(os.path.join(out, "env", f"institute_round5_{stamp}.json"), "w") as f:
            json.dump(env, f, indent=1)
        record_truncated(vlcs, out)

        jobs = []
        for t in TEST_ENVS:          # fold and seed outermost, so each pair finishes together
            for s in SEEDS:
                for lp in LPARAMS:
                    d = run_dir(out, lp, t, s)
                    if not r4.finished(d):
                        jobs.append({"name": f"lparam{lp}_test{t}_seed{s}", "dir": d,
                                     "cmd": train_command(lp, t, s, data_root, d)})
        log(f"{N_RUNS - len(jobs)} of {N_RUNS} round-5 runs already finished; {len(jobs)} to do, "
            f"{a.parallel} at a time.")
        failed = r4.schedule(jobs, a.parallel, out, authors)
        if failed:
            sys.exit(f"{len(failed)} run(s) failed. Send the lines above to the project owner; finished runs are kept.")

        log(f"All {N_RUNS} runs finished. Running the pre-registered analysis:")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analyze_round5.py"), out,
                        "--json", os.path.join(out, "round5_analysis_institute.json")], cwd=ROOT)
        log(f"Done. Zip the whole folder {out} (as m2cl_round5.zip) and send it back.")
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
