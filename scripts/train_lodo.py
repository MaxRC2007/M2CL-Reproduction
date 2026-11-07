#!/usr/bin/env python
"""Train ERM / M2 / M2-CL with leave-one-domain-out on PACS.

Examples
  # locked corrected M2-CL, all 4 targets x 3 seeds
  python scripts/train_lodo.py --config configs/m2cl_6tap.yaml --data_root /content/PACS --out_dir runs
  # the missing ablation: 6-tap M2 without the contrastive loss
  python scripts/train_lodo.py --config configs/m2_6tap.yaml --data_root /content/PACS --out_dir runs
  # one quick run
  python scripts/train_lodo.py --model erm --targets sketch --seeds 0 --data_root PACS --out_dir runs
Runs that already have a complete result.json with the same config are skipped,
interrupted runs resume from runs/<run>/latest.pt.
"""
import argparse
import os
import sys
from dataclasses import fields

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import yaml  # noqa: E402

from m2cl.engine import RunConfig, train_run  # noqa: E402
from m2cl.splits import ALL_DOMAINS, detect_domains  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML with RunConfig fields + optional targets/seeds")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--targets", nargs="+", choices=ALL_DOMAINS)
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--no_resume", action="store_true")
    for f in fields(RunConfig):
        if f.name in ("target", "seed"):
            continue
        t = (lambda s: s.lower() in ("1", "true", "yes")) if f.type in (bool, "bool") else \
            {"int": int, "float": float, "str": str}.get(f.type if isinstance(f.type, str) else f.type.__name__, str)
        ap.add_argument(f"--{f.name}", type=t)
    a = ap.parse_args()

    cfg = {}
    if a.config:
        with open(a.config) as fh:
            cfg = yaml.safe_load(fh) or {}
    targets = a.targets or cfg.pop("targets", list(detect_domains(a.data_root)))
    seeds = a.seeds if a.seeds is not None else cfg.pop("seeds", [0, 1, 2])
    cfg.pop("targets", None); cfg.pop("seeds", None)
    for f in fields(RunConfig):
        v = getattr(a, f.name, None)
        if v is not None:
            cfg[f.name] = v

    for t in targets:
        for s in seeds:
            train_run(RunConfig(target=t, seed=s, **cfg), a.data_root, a.out_dir, resume=not a.no_resume)


if __name__ == "__main__":
    main()
