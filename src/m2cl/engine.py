"""Training / evaluation for one leave-one-domain-out run.

Protocol (locked configuration of the paper, Table IV):
  SGD(lr=5e-4, momentum=0.9, weight_decay=5e-4), cosine annealing over `epochs`,
  batch 128, AMP on CUDA, model selection by mean per-domain SOURCE validation
  accuracy, target domain evaluated exactly once with the selected weights.
Every epoch is checkpointed (latest.pt, with the previous one kept as latest_prev.pt), so a
disconnect loses at most about one epoch, and a resumed run continues the same RNG streams
as an uninterrupted one (tests/test_resume.py).
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .data import make_lodo_loaders
from .losses import m2cl_contrastive_loss
from .models import build_model


@dataclass
class RunConfig:
    target: str = "sketch"
    model: str = "m2cl"          # erm | m2 | m2cl
    arch_taps: int = 6
    alpha: float = 0.003
    tau: float = 1.0
    loss_reduction: str = "sum"  # "mean" reproduces the original bug
    augmentation: str = "baseline"
    epochs: int = 30
    batch_size: int = 128
    img_size: int = 224
    lr: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 5e-4
    val_fraction: float = 0.2
    seed: int = 0
    pretrained: bool = True
    num_workers: int = 2

    @property
    def run_name(self) -> str:
        taps = f"_{self.arch_taps}tap" if self.model != "erm" else ""
        return f"{self.model}{taps}_{self.target}_seed{self.seed}"


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return 100.0 * correct / max(total, 1)


def _save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _save_latest(obj, latest, previous):
    """Keep the last good checkpoint until the new one is fully written."""
    tmp = latest + ".tmp"
    torch.save(obj, tmp)
    if os.path.exists(latest):
        os.replace(latest, previous)
    os.replace(tmp, latest)


def _load_resume_checkpoint(paths, cfg_dict):
    """Newest readable checkpoint with this exact config, or None. A damaged file never stops a run."""
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:  # e.g. a file cut short by a disconnect
            print(f"[warn] cannot read {path} ({type(e).__name__}); trying the previous checkpoint")
            continue
        if ck.get("config") == cfg_dict:
            return ck
        print(f"[warn] {path} has different settings; ignoring it")
    return None


def train_run(cfg: RunConfig, data_root: str, out_dir: str, resume: bool = True) -> Dict:
    device = _device()
    run_dir = os.path.join(out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    result_path = os.path.join(run_dir, "result.json")
    if resume and os.path.exists(result_path):
        with open(result_path) as f:
            res = json.load(f)
        if res.get("status") == "complete" and res.get("config") == asdict(cfg):
            print(f"[skip] {cfg.run_name} already complete: target acc {res['test_acc']:.2f}%")
            return res

    set_seed(cfg.seed)
    loaders, info = make_lodo_loaders(data_root, cfg.target, cfg.augmentation, cfg.img_size,
                                      cfg.batch_size, cfg.val_fraction, cfg.seed, cfg.num_workers)
    model = build_model(cfg.model, len(info["classes"]), cfg.arch_taps, cfg.pretrained).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device=device.type, enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss()

    latest = os.path.join(run_dir, "latest.pt")
    previous = os.path.join(run_dir, "latest_prev.pt")
    start, best_val, best_state, history = 1, -1.0, None, []
    ck = _load_resume_checkpoint([latest, previous], asdict(cfg)) if resume else None
    if ck is not None:
        model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["optimizer_state"])
        sched.load_state_dict(ck["scheduler_state"]); scaler.load_state_dict(ck["scaler_state"])
        start, best_val, best_state, history = ck["epoch"] + 1, ck["best_val"], ck["best_model_state"], ck["history"]
        random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        if torch.cuda.is_available() and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        if ck["rng"].get("train_loader") is not None:  # the shuffle order continues, not restarts
            loaders["train"].generator.set_state(ck["rng"]["train_loader"])
        print(f"[resume] {cfg.run_name} from epoch {start}")

    print(f"[run] {cfg.run_name} | train={info['n_train']} val={info['n_val']} test={info['n_test']} | {device}")
    for epoch in range(start, cfg.epochs + 1):
        model.train()
        t0, tot_loss, tot_cl, correct, seen = time.time(), 0.0, 0.0, 0, 0
        for x, y, _ in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, reps = model(x)
                loss = ce(logits, y)
            if reps is not None and cfg.alpha > 0:
                cl = m2cl_contrastive_loss(reps, y, cfg.tau, cfg.loss_reduction)  # fp32
                loss = loss + cfg.alpha * cl
                tot_cl += cl.item() * len(y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        sched.step()
        val = {d: evaluate(model, l, device) for d, l in loaders["val"].items()}
        val_mean = float(np.mean(list(val.values())))
        if val_mean > best_val:
            best_val = val_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "loss": tot_loss / seen, "cl_loss": tot_cl / seen,
                        "train_acc": 100 * correct / seen, "val_mean": val_mean, "val": val})
        _save_latest({"config": asdict(cfg), "epoch": epoch, "model_state": model.state_dict(),
                      "optimizer_state": opt.state_dict(), "scheduler_state": sched.state_dict(),
                      "scaler_state": scaler.state_dict(), "best_val": best_val, "best_model_state": best_state,
                      "history": history,
                      "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                              "torch": torch.get_rng_state(),
                              "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                              "train_loader": loaders["train"].generator.get_state()}},
                     latest, previous)
        print(f"  ep {epoch:02d}/{cfg.epochs} loss={tot_loss/seen:.4f} cl={tot_cl/seen:.3f} "
              f"train={100*correct/seen:.1f} val={val_mean:.2f} ({time.time()-t0:.0f}s)")

    model.load_state_dict(best_state)
    test_acc = evaluate(model, loaders["test"], device)  # the ONLY touch of the target domain
    _save({"config": asdict(cfg), "model_state": best_state, "best_val": best_val}, os.path.join(run_dir, "best.pt"))
    best_epoch = int(np.argmax([h["val_mean"] for h in history])) + 1
    res = {"status": "complete", "run_name": cfg.run_name, "config": asdict(cfg), "test_acc": test_acc,
           "best_val": best_val, "best_epoch": best_epoch, "history": history, "data": info,
           "torch": torch.__version__, "device": str(device)}
    tmp = result_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, result_path)
    for leftover in (latest, previous, latest + ".tmp"):  # only once the result is safely written
        if os.path.exists(leftover):
            os.remove(leftover)
    print(f"[done] {cfg.run_name}: target {cfg.target} = {test_acc:.2f}% (best epoch {best_epoch})")
    return res
