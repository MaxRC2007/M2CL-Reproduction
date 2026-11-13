#!/usr/bin/env python
"""Round 6: the authors' own M2-CL code on PACS, lparam 0.01 vs 0, new seeds 3-7 (docs/PREREGISTRATION_ROUND6.md).

  python scripts/run_round6_authors.py --out ~/m2cl_round6 --data_dir ~/data --smoke         # first: short test
  python scripts/run_round6_authors.py --out ~/m2cl_round6 --data_dir ~/data --parallel 2    # then: all 40 runs

Round 4's runner with the seeds changed and nothing else. It reuses the authors' code that round 4 cloned (commit
checked, never modified) and the PACS copy round 4 verified (image counts checked; downloaded from the same mirror if
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
from run_part_c import check_gpu, log  # noqa: E402

COMMIT = r4.COMMIT
RESNET18_URL = r4.RESNET18_URL
LPARAMS, TEST_ENVS = r4.LPARAMS, r4.TEST_ENVS
SEEDS = (3, 4, 5, 6, 7)
N_RUNS = len(LPARAMS) * len(SEEDS) * len(TEST_ENVS)
N_STEPS_FULL = 5001
FIXED = r4.FIXED                       # PACS, M2CL, hparams_seed 0, resnet18, lr 0.001, temp 1.0


def run_dir(out, lparam, test_env, seed):
    return os.path.join(out, "round6", f"lparam_{lparam}", f"test{test_env}_seed{seed}")


def train_command(lparam, test_env, seed, data_root, out_dir, extra=()):
    return r4.train_command(lparam, test_env, seed, data_root, out_dir, extra)


def acquire_lock(out):
    path = os.path.join(out, ".round6.lock")
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
            sys.exit(f"Another run_round6_authors.py (process {pid}) is already using {out}. Stop it first, or wait.")
    with open(path, "w") as f:
        f.write(str(os.getpid()))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="results folder to send back (e.g. ~/m2cl_round6)")
    ap.add_argument("--data_dir", required=True, help="where PACS and the authors' code are, or will be put")
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
    data_root = r4.find_pacs(data_dir) or r4.download_pacs(data_dir)
    pacs = os.path.join(data_root, "PACS")
    log(f"Authors' code: {authors} (commit {COMMIT[:7]}, unmodified) | PACS: {pacs} "
        f"| images per domain: {r4.pacs_counts(pacs)}")
    torch.hub.load_state_dict_from_url(RESNET18_URL, progress=False)

    if a.smoke:
        smoke_out = out.rstrip("/\\") + "_smoke"
        shutil.rmtree(smoke_out, ignore_errors=True)
        os.makedirs(smoke_out)
        d = os.path.join(smoke_out, "round6", "lparam_0.01", "test0_seed3")
        job = {"name": "smoke_lparam0.01_test0_seed3", "dir": d,
               "cmd": train_command("0.01", 0, 3, data_root, d, ("--steps", "101", "--checkpoint_freq", "50"))}
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
        env = {"platform": "institute", "round": 6, "started_utc": stamp, **gpu, "packages": r4.versions(),
               "python": platform.python_version(), "os": platform.platform(), "cpu": cpu,
               "cpu_count": os.cpu_count(), "parallel": a.parallel, "authors_commit": COMMIT,
               "authors_code": authors, "pacs": pacs, "pacs_counts": r4.pacs_counts(pacs), "seeds": list(SEEDS),
               "command_fixed_args": FIXED}
        with open(os.path.join(out, "env", f"institute_round6_{stamp}.json"), "w") as f:
            json.dump(env, f, indent=1)

        jobs = []
        for t in TEST_ENVS:          # fold and seed outermost, so each pair finishes together
            for s in SEEDS:
                for lp in LPARAMS:
                    d = run_dir(out, lp, t, s)
                    if not r4.finished(d):
                        jobs.append({"name": f"lparam{lp}_test{t}_seed{s}", "dir": d,
                                     "cmd": train_command(lp, t, s, data_root, d)})
        log(f"{N_RUNS - len(jobs)} of {N_RUNS} round-6 runs already finished; {len(jobs)} to do, "
            f"{a.parallel} at a time.")
        failed = r4.schedule(jobs, a.parallel, out, authors)
        if failed:
            sys.exit(f"{len(failed)} run(s) failed. Send the lines above to the project owner; finished runs are kept.")

        log(f"All {N_RUNS} runs finished. Running the pre-registered analysis:")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analyze_round6.py"), out,
                        "--json", os.path.join(out, "round6_analysis_institute.json")], cwd=ROOT)
        log(f"Done. Zip the whole folder {out} (as m2cl_round6.zip) and send it back.")
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
