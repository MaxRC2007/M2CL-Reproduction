#!/usr/bin/env python
"""Round 4: the authors' own M2-CL code on PACS, lparam 0.01 vs 0 (docs/PREREGISTRATION_ROUND4.md).

  python scripts/run_round4_authors.py --out ~/m2cl_round4 --data_dir ~/data --smoke         # first: short test
  python scripts/run_round4_authors.py --out ~/m2cl_round4 --data_dir ~/data --parallel 2    # then: all 24 runs

It downloads the authors' repository at the pre-registered commit (never modifies it), downloads DomainBed's PACS
and checks its image counts, then runs `python -m domainbed.scripts.train` once per run. Run it inside tmux.
Safe to stop and start again: finished runs are kept; an unfinished run (the code cannot resume) is deleted and
restarted from scratch with the same seeds.
"""
import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from run_part_c import _Progress, check_gpu, log  # noqa: E402  (same helpers as Part C)


def acquire_lock(out):
    path = os.path.join(out, ".round4.lock")
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
            sys.exit(f"Another run_round4_authors.py (process {pid}) is already using {out}. Stop it first, or wait.")
    with open(path, "w") as f:
        f.write(str(os.getpid()))
    return path

REPO = "https://github.com/aristotelisballas/m2cl.git"
COMMIT = "0b1bce3026d9d56369037129e36f81e7645b80e6"
ARCHIVE = f"https://github.com/aristotelisballas/m2cl/archive/{COMMIT}.zip"
PACS_MIRROR = "https://huggingface.co/datasets/PTeterwak/DomainBed_OOP/resolve/main/all/PACS.tar.gz"
PACS_COUNTS = {"art_painting": 2048, "cartoon": 2344, "photo": 1670, "sketch": 3929}   # 9,991 images
IMG_EXTS = (".jpg", ".jpeg", ".png")
RESNET18_URL = "https://download.pytorch.org/models/resnet18-5c106cde.pth"            # the authors' m2cl_18.py
LPARAMS = ("0.01", "0.0")
SEEDS = (0, 1, 2)
TEST_ENVS = (0, 1, 2, 3)
N_RUNS = len(LPARAMS) * len(SEEDS) * len(TEST_ENVS)
FIXED = ["--dataset", "PACS", "--algorithm", "M2CL", "--hparams_seed", "0", "--hparams", '{"resnet18": true}',
         "--lr", "0.001", "--temp", "1.0"]


# ---------------------------------------------------------------- authors' code
def authors_code(base):
    """Clone (or reuse) the authors' repository at COMMIT and check that no tracked file is modified."""
    d = os.path.join(base, f"authors_m2cl_{COMMIT[:7]}")
    if not os.path.isdir(d):
        tmp = d + ".partial"
        shutil.rmtree(tmp, ignore_errors=True)
        if shutil.which("git"):
            log(f"Cloning {REPO} at {COMMIT[:7]}")
            subprocess.run(["git", "clone", "-q", REPO, tmp], check=True)
            subprocess.run(["git", "-C", tmp, "checkout", "-q", COMMIT], check=True)
        else:
            log(f"git not found; downloading {ARCHIVE}")
            os.makedirs(tmp)
            with urllib.request.urlopen(ARCHIVE, timeout=60) as resp:
                zf = zipfile.ZipFile(__import__("io").BytesIO(resp.read()))
            top = zf.namelist()[0].split("/")[0]
            zf.extractall(tmp)
            for name in os.listdir(os.path.join(tmp, top)):
                shutil.move(os.path.join(tmp, top, name), tmp)
            os.rmdir(os.path.join(tmp, top))
            with open(os.path.join(tmp, "COMMIT_FROM_ARCHIVE"), "w") as f:
                f.write(COMMIT)
        os.replace(tmp, d)
    if os.path.isdir(os.path.join(d, ".git")):
        head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", d, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        if head != COMMIT or dirty:
            sys.exit(f"The authors' code in {d} is not the pre-registered unmodified commit {COMMIT[:7]} "
                     f"(HEAD {head[:7]}, modified files: {dirty or 'none'}). Delete that folder and run again.")
    if not os.path.exists(os.path.join(d, "domainbed", "scripts", "train.py")):
        sys.exit(f"{d} does not contain domainbed/scripts/train.py. Delete it and run again.")
    return d


# ---------------------------------------------------------------- PACS
def pacs_counts(pacs):
    return {dom: sum(f.lower().endswith(IMG_EXTS) for _, _, fs in os.walk(os.path.join(pacs, dom)) for f in fs)
            for dom in PACS_COUNTS}


def find_pacs(base):
    """The folder that DomainBed gets as --data_dir: it must contain PACS/ with exactly the four domains."""
    if not os.path.isdir(base):
        return None
    for r, d, _ in os.walk(base):
        if "PACS" in d:
            pacs = os.path.join(r, "PACS")
            subdirs = sorted(e.name for e in os.scandir(pacs) if e.is_dir())
            if subdirs != sorted(PACS_COUNTS):
                log(f"{pacs} has folders {subdirs}; DomainBed needs exactly {sorted(PACS_COUNTS)}")
                continue
            counts = pacs_counts(pacs)
            if counts == PACS_COUNTS:
                return r
            log(f"{pacs} has the wrong image counts {counts}; expected {PACS_COUNTS}")
    return None


def download_pacs(data_dir):
    final, tmp = os.path.join(data_dir, "domainbed_pacs"), os.path.join(data_dir, "domainbed_pacs.partial")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    log(f"Downloading PACS (0.18 GB) from {PACS_MIRROR}")
    with urllib.request.urlopen(PACS_MIRROR, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or 179_024_815
        with tarfile.open(fileobj=_Progress(resp, total), mode="r|gz") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(tmp, filter="data")
            else:
                tar.extractall(tmp)
    if find_pacs(tmp) is None:
        sys.exit(f"The download in {tmp} is not DomainBed's PACS with counts {PACS_COUNTS}. Delete it and retry.")
    os.replace(tmp, final)
    return find_pacs(final)


# ---------------------------------------------------------------- runs
def run_dir(out, lparam, test_env, seed):
    return os.path.join(out, "round4", f"lparam_{lparam}", f"test{test_env}_seed{seed}")


def finished(d):
    return os.path.exists(os.path.join(d, "done")) and os.path.exists(os.path.join(d, "results.jsonl"))


def train_command(lparam, test_env, seed, data_root, out_dir, extra=()):
    return [sys.executable, "-u", "-m", "domainbed.scripts.train", "--data_dir", data_root, *FIXED,
            "--lparam", lparam, "--test_envs", str(test_env), "--trial_seed", str(seed), "--seed", str(seed),
            "--output_dir", out_dir, *extra]


def selected_test_acc(d):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from analyze_round4 import select
    with open(os.path.join(d, "results.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    test_env = records[0]["args"]["test_envs"][0]
    return select(records, test_env)[1], max(r.get("mem_gb", 0) for r in records)


def schedule(jobs, parallel, out, authors, launch=None):
    """Run jobs with at most `parallel` at once; returns the failed jobs. A failure stops new starts."""
    env = dict(os.environ, PYTHONPATH=authors + os.pathsep + os.environ.get("PYTHONPATH", ""))
    launch = launch or (lambda job, logf: subprocess.Popen(job["cmd"], stdout=logf, stderr=subprocess.STDOUT,
                                                           cwd=authors, env=env))
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    pending, running, failed = list(jobs), [], []
    while pending or running:
        while pending and len(running) < parallel and not failed:
            job = pending.pop(0)
            if os.path.isdir(job["dir"]):
                log(f"[restart] {job['name']}: removing its unfinished output (the code cannot resume)")
                shutil.rmtree(job["dir"])
            logf = open(os.path.join(out, "logs", job["name"] + ".log"), "a")
            log(f"[start] {job['name']}  ({len(pending)} waiting)")
            running.append((job, launch(job, logf), logf, time.time()))
        time.sleep(job_poll_seconds)
        for item in list(running):
            job, proc, logf, t0 = item
            code = proc.poll()
            if code is None:
                continue
            running.remove(item)
            logf.close()
            minutes = (time.time() - t0) / 60
            if code == 0 and finished(job["dir"]):
                acc, mem = selected_test_acc(job["dir"])
                log(f"[done]  {job['name']}: {acc:.2f}% (training-domain validation) in {minutes:.0f} min, "
                    f"peak GPU memory {mem:.1f} GB")
            else:
                failed.append(job)
                log(f"[FAILED] {job['name']} (exit code {code}) after {minutes:.0f} min. Last lines of "
                    f"{os.path.join(out, 'logs', job['name'] + '.log')}:")
                with open(os.path.join(out, "logs", job["name"] + ".log")) as f:
                    print("".join(f.readlines()[-20:]), flush=True)
                if pending:
                    log("No new runs will start; the running ones finish first.")
        if failed and not running:
            break
    return failed


job_poll_seconds = 10


def versions():
    out = {}
    for pkg in ("torch", "torchvision", "numpy", "Pillow", "wilds", "timm", "scipy"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="results folder to send back (e.g. ~/m2cl_round4)")
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
    missing = [p for p, v in versions().items() if v is None and p in ("wilds", "timm")]
    if missing:
        sys.exit(f"The authors' code imports {missing}. Install them first: pip install -r requirements_round4.txt")
    authors = authors_code(data_dir)
    data_root = find_pacs(data_dir) or download_pacs(data_dir)
    log(f"Authors' code: {authors} (commit {COMMIT[:7]}, unmodified) | PACS: {os.path.join(data_root, 'PACS')} "
        f"| images per domain: {pacs_counts(os.path.join(data_root, 'PACS'))}")
    # Download the ImageNet weights the authors' model loads, once, so parallel runs never race on the cache file
    torch.hub.load_state_dict_from_url(RESNET18_URL, progress=False)

    if a.smoke:
        smoke_out = out.rstrip("/\\") + "_smoke"
        shutil.rmtree(smoke_out, ignore_errors=True)
        os.makedirs(smoke_out)
        d = os.path.join(smoke_out, "round4", "lparam_0.01", "test0_seed0")
        job = {"name": "smoke_lparam0.01_test0_seed0", "dir": d,
               "cmd": train_command("0.01", 0, 0, data_root, d, ("--steps", "101", "--checkpoint_freq", "50"))}
        t0 = time.time()
        failed = schedule([job], 1, smoke_out, authors)
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
        env = {"platform": "institute", "round": 4, "started_utc": stamp, **gpu, "packages": versions(),
               "python": platform.python_version(), "os": platform.platform(), "cpu": cpu,
               "cpu_count": os.cpu_count(), "parallel": a.parallel, "authors_commit": COMMIT,
               "authors_code": authors, "command_fixed_args": FIXED}
        with open(os.path.join(out, "env", f"institute_round4_{stamp}.json"), "w") as f:
            json.dump(env, f, indent=1)

        jobs = []
        for t in TEST_ENVS:          # fold and seed outermost, so each pair finishes together
            for s in SEEDS:
                for lp in LPARAMS:
                    d = run_dir(out, lp, t, s)
                    if not finished(d):
                        jobs.append({"name": f"lparam{lp}_test{t}_seed{s}", "dir": d,
                                     "cmd": train_command(lp, t, s, data_root, d)})
        log(f"{N_RUNS - len(jobs)} of {N_RUNS} round-4 runs already finished; {len(jobs)} to do, "
            f"{a.parallel} at a time.")
        failed = schedule(jobs, a.parallel, out, authors)
        if failed:
            sys.exit(f"{len(failed)} run(s) failed. Send the lines above to the project owner; finished runs are kept.")

        log(f"All {N_RUNS} runs finished. Running the pre-registered analysis:")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analyze_round4.py"), out,
                        "--json", os.path.join(out, "round4_analysis_institute.json")], cwd=ROOT)
        log(f"Done. Zip the whole folder {out} (as m2cl_round4.zip) and send it back.")
    finally:
        os.remove(lock)


N_STEPS_FULL = 5001

if __name__ == "__main__":
    main()
