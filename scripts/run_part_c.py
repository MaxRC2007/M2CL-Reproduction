#!/usr/bin/env python
"""Round 3, Part C (VLCS replication) on a local Linux GPU machine (pre-registration, Amendment 2).

Same configs, run names, data checks and analysis as the Colab and Kaggle notebooks; only the plumbing differs.

  python scripts/run_part_c.py --out ~/m2cl_v2 --data_dir ~/data --smoke         # first: 1-epoch test run
  python scripts/run_part_c.py --out ~/m2cl_v2 --data_dir ~/data --parallel 2    # then: all 36 runs

Run it inside tmux (see docs/RUN_INSTITUTE.md) so it survives logging out. It is safe to stop (Ctrl+C) and
start again: finished runs are skipped and unfinished ones resume from their last saved epoch.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

VLCS_T = ("Caltech101", "LabelMe", "SUN09", "VOC2007")
EXPECTED = {"Caltech101": 1415, "LabelMe": 2656, "SUN09": 3282, "VOC2007": 3376}   # DomainBed: 10,729 images
IMG_EXTS = (".jpg", ".jpeg", ".png")                                                 # as in src/m2cl/splits.py
MIRROR = "https://huggingface.co/datasets/PTeterwak/DomainBed_OOP/resolve/main/all/VLCS.tar.gz"  # docs/AUDIT.md F15
METHODS = ("m2", "m2cl", "erm")
SEEDS = (0, 1, 2)
CFG = {m: os.path.join(ROOT, "configs", f"round3_{m}.yaml") for m in METHODS}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- GPU
def check_gpu():
    import torch
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible to PyTorch. Check `nvidia-smi` and that the CUDA build of PyTorch is installed.")
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        sys.exit(f"{torch.cuda.get_device_name(0)} needs {arch}, which this PyTorch ({torch.__version__}) was not "
                 f"built for (it has {torch.cuda.get_arch_list()}). For Blackwell GPUs install PyTorch 2.7 or newer "
                 "built for CUDA 12.8 (docs/RUN_INSTITUTE.md).")
    try:  # a real kernel under mixed precision, as training uses
        with torch.autocast("cuda"):
            x = torch.nn.functional.conv2d(torch.randn(2, 3, 32, 32, device="cuda"),
                                           torch.randn(8, 3, 3, 3, device="cuda"))
        torch.cuda.synchronize()
        assert torch.isfinite(x).all()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"A test kernel failed on the GPU: {e}")
    props = torch.cuda.get_device_properties(0)
    log(f"GPU: {props.name} ({arch}, {props.total_memory / 1e9:.0f} GB) | torch {torch.__version__} "
        f"| CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")
    return {"gpu": props.name, "capability": arch, "gpu_memory_gb": round(props.total_memory / 1e9, 1),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


# ---------------------------------------------------------------- VLCS
def image_counts(root):
    return {t: sum(f.lower().endswith(IMG_EXTS) for _, _, fs in os.walk(os.path.join(root, t)) for f in fs)
            for t in VLCS_T}


def find_vlcs(base):
    if not os.path.isdir(base):
        return None
    for r, d, _ in os.walk(base):
        if set(VLCS_T) <= set(d):
            counts = image_counts(r)
            if counts == EXPECTED:
                return r
            log(f"{r} has the wrong image counts {counts}; expected {EXPECTED}")
    return None


class _Progress:
    def __init__(self, f, total):
        self.f, self.total, self.done, self.next = f, total, 0, 0

    def read(self, n=-1):
        b = self.f.read(n)
        self.done += len(b)
        if self.done >= self.next:
            log(f"  downloaded {self.done / 1e9:.2f} of {self.total / 1e9:.2f} GB")
            self.next += 250e6
        return b


def download_vlcs(data_dir):
    final, tmp = os.path.join(data_dir, "VLCS_raw"), os.path.join(data_dir, "VLCS_raw.partial")
    os.makedirs(tmp, exist_ok=True)
    log(f"Downloading VLCS (3.8 GB) from {MIRROR}")
    with urllib.request.urlopen(MIRROR, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or 3_838_010_386
        with tarfile.open(fileobj=_Progress(resp, total), mode="r|gz") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(tmp, filter="data")       # refuses absolute paths and links outside the folder
            else:
                tar.extractall(tmp)
    root = find_vlcs(tmp)
    if root is None:
        sys.exit(f"The download in {tmp} does not have DomainBed's image counts {EXPECTED}. Delete it and retry.")
    os.replace(tmp, final)
    return find_vlcs(final)


def record_truncated(vlcs, out):
    if os.path.exists(os.path.join(out, "env", "vlcs_truncated_images.json")):
        return
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    bad = []
    for r, _, fs in os.walk(vlcs):
        for f in sorted(fs):
            if f.lower().endswith(IMG_EXTS):
                try:
                    Image.open(os.path.join(r, f)).convert("RGB")
                except OSError as e:
                    bad.append([os.path.relpath(os.path.join(r, f), vlcs).replace(os.sep, "/"), str(e)])
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with open(os.path.join(out, "env", "vlcs_truncated_images.json"), "w") as f:
        json.dump(bad, f, indent=1)
    log(f"Truncated VLCS images: {len(bad)} (decoded as far as each file goes, as in DomainBed; "
        "listed in env/vlcs_truncated_images.json)")


# ---------------------------------------------------------------- runs
def run_name(method, target, seed):
    import yaml
    from m2cl.engine import RunConfig
    with open(CFG[method]) as f:
        cfg = {k: v for k, v in yaml.safe_load(f).items() if k not in ("targets", "seeds")}
    return RunConfig(target=target, seed=seed, **cfg).run_name


def run_dir(out, method, target, seed):
    return os.path.join(out, "round3_vlcs", method, run_name(method, target, seed))


def finished(d):
    try:
        with open(os.path.join(d, "result.json")) as f:
            return json.load(f).get("status") == "complete"
    except (OSError, ValueError):
        return False


def train_command(method, target, seed, vlcs, out_dir, extra=()):
    return [sys.executable, "-u", os.path.join(ROOT, "scripts", "train_lodo.py"), "--config", CFG[method],
            "--targets", target, "--seeds", str(seed), "--data_root", vlcs, "--out_dir", out_dir, *extra]


def schedule(jobs, parallel, out, launch=None):
    """Run jobs (method, target, seed) with at most `parallel` at once. Returns the list of failed jobs."""
    launch = launch or (lambda job, logf: subprocess.Popen(job["cmd"], stdout=logf, stderr=subprocess.STDOUT,
                                                           cwd=ROOT))
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    pending, running, failed = list(jobs), [], []
    while pending or running:
        while pending and len(running) < parallel and not failed:
            job = pending.pop(0)
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
                with open(os.path.join(job["dir"], "result.json")) as f:
                    acc = json.load(f)["test_acc"]
                log(f"[done]  {job['name']}: {acc:.2f}% in {minutes:.0f} min")
            else:
                failed.append(job)
                log(f"[FAILED] {job['name']} (exit code {code}) after {minutes:.0f} min. Last lines of "
                    f"{os.path.join(out, 'logs', job['name'] + '.log')}:")
                with open(os.path.join(out, "logs", job["name"] + ".log")) as f:
                    print("".join(f.readlines()[-15:]), flush=True)
                if pending:
                    log("No new runs will start; the running ones finish first.")
        if failed and not running:
            break
    return failed


job_poll_seconds = 10


def acquire_lock(out):
    path = os.path.join(out, ".part_c.lock")
    if os.path.exists(path):
        with open(path) as f:
            pid = int(f.read().strip() or 0)
        alive = False
        if pid and hasattr(os, "kill") and os.name == "posix":
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                pass
        if alive:
            sys.exit(f"Another run_part_c.py (process {pid}) is already using {out}. Two copies would train the "
                     "same runs twice. Stop it first, or wait for it to finish.")
    with open(path, "w") as f:
        f.write(str(os.getpid()))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="results folder (the m2cl_v2 folder that gets sent back)")
    ap.add_argument("--data_dir", required=True, help="where VLCS is, or will be downloaded to")
    ap.add_argument("--parallel", type=int, default=1, help="runs at once on the one GPU (default 1; 2 is fine "
                    "with 16 GB of GPU memory and 8 or more CPU cores)")
    ap.add_argument("--smoke", action="store_true", help="one 1-epoch run in <out>_smoke, to test the setup")
    ap.add_argument("--no_gpu_check", action="store_true", help=argparse.SUPPRESS)  # testing this script only
    a = ap.parse_args()
    out, data_dir = os.path.abspath(os.path.expanduser(a.out)), os.path.abspath(os.path.expanduser(a.data_dir))
    assert a.parallel >= 1

    import torch
    import torchvision
    gpu = {} if a.no_gpu_check else check_gpu()
    with open(os.path.join(ROOT, "src", "m2cl", "data.py"), encoding="utf-8") as f:
        assert "ImageFile.LOAD_TRUNCATED_IMAGES = True" in f.read(), \
            "This code is older than 7 Nov 2026 (no truncated-image line): use the current repository or zip."

    vlcs = find_vlcs(data_dir) or download_vlcs(data_dir)
    log(f"VLCS root: {vlcs} | images per domain: {image_counts(vlcs)}")
    # Download the ImageNet weights once, so parallel runs never write the same cache file at the same time
    torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    if a.smoke:
        smoke_out = out.rstrip("/\\") + "_smoke"
        os.makedirs(smoke_out, exist_ok=True)
        job = {"name": "smoke_m2cl_Caltech101", "dir": os.path.join(smoke_out, run_name("m2cl", "Caltech101", 0)),
               "cmd": train_command("m2cl", "Caltech101", 0, vlcs, smoke_out, ("--epochs", "1", "--no_resume"))}
        t0 = time.time()
        failed = schedule([job], 1, smoke_out)
        if failed:
            sys.exit("The smoke test failed; see the lines above.")
        minutes = (time.time() - t0) / 60
        log(f"Smoke test passed: one epoch of M2-CL took {minutes:.1f} min including start-up. A full run is "
            f"30 epochs, so expect roughly {minutes * 30 / 60:.1f} h per run and "
            f"{minutes * 30 * 36 / 60 / a.parallel:.0f} h for all 36 (with --parallel {a.parallel}; parallel runs "
            f"are each a little slower). The smoke run is in {smoke_out}: it is a test only, never analysed.")
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
        env = {"platform": "institute", "parts": ["C"], "started_utc": stamp, **gpu, "torch": torch.__version__,
               "torchvision": torchvision.__version__, "python": platform.python_version(), "os": platform.platform(),
               "cpu": cpu, "cpu_count": os.cpu_count(), "parallel": a.parallel}
        with open(os.path.join(out, "env", f"institute_{stamp}.json"), "w") as f:
            json.dump(env, f, indent=1)
        record_truncated(vlcs, out)

        # Fold and seed are the outer loops, so whole sets of M2 / M2-CL / ERM finish together.
        jobs = []
        for t in VLCS_T:
            for s in SEEDS:
                for m in METHODS:
                    d = run_dir(out, m, t, s)
                    if not finished(d):
                        jobs.append({"name": os.path.basename(d), "dir": d,
                                     "cmd": train_command(m, t, s, vlcs, os.path.dirname(d))})
        log(f"{36 - len(jobs)} of 36 Part C runs already finished; {len(jobs)} to do, {a.parallel} at a time.")
        failed = schedule(jobs, a.parallel, out)
        if failed:
            sys.exit(f"{len(failed)} run(s) failed. Fix the cause and start the script again; finished runs are kept.")

        log("All 36 runs finished. Running the pre-registered analysis (VLCS part):")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analyze_round3.py"), out,
                        "--json", os.path.join(out, "round3_analysis_institute.json")], cwd=ROOT)
        log(f"Done. Send the whole folder {out} back (zip it as m2cl_v2_C.zip).")
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
