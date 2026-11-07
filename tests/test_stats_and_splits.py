"""Torch-free tests: statistics, splits and the audit claims (run anywhere)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from m2cl.splits import PACS_DOMAINS, index_pacs, legacy_val_split, probe_split  # noqa: E402
from m2cl.stats import (benjamini_hochberg, min_attainable_p, paired_bootstrap,  # noqa: E402
                        sign_flip_test, sign_test)

SIZES = {"art_painting": 2048, "cartoon": 2344, "photo": 1670, "sketch": 3929}


def test_legacy_split_leaks_val_into_train():
    """Audit A1: original notebook -> ~80% of 2nd/3rd source domain val images are train images."""
    for tgt in PACS_DOMAINS:
        src = [d for d in PACS_DOMAINS if d != tgt]
        ov = legacy_val_split(SIZES, src, 0.2, seed=0)
        assert ov[src[0]] == 0.0
        assert ov[src[1]] > 0.7 and ov[src[2]] > 0.7


def test_fixed_split_is_disjoint(tmp_path):
    from m2cl.splits import index_pacs, split_train_val
    for d in PACS_DOMAINS:
        for c in ("dog", "horse"):
            p = tmp_path / d / c
            p.mkdir(parents=True)
            for i in range(10):
                (p / f"{i}.jpg").write_bytes(b"")
    classes, by = index_pacs(str(tmp_path))
    assert classes == ["dog", "horse"]
    for seed in (0, 1, 2):
        tr, val = split_train_val(by, PACS_DOMAINS[:3], 0.2, seed)
        tr_paths = {s[0] for s in tr}
        for v in val.values():
            assert not ({s[0] for s in v} & tr_paths)
        assert len(tr) + sum(map(len, val.values())) == 60


def test_probe_split_stratified_and_deterministic():
    y = np.repeat([0, 1, 2], [500, 300, 200])
    tr, te = probe_split(y, 0.3, 42)
    tr2, te2 = probe_split(y, 0.3, 42)
    assert (te == te2).all() and not (set(tr) & set(te))
    assert np.allclose(np.bincount(y[te]) / len(te), [0.5, 0.3, 0.2], atol=0.01)


def test_paired_bootstrap_point_estimate_matches_accuracies():
    rng = np.random.default_rng(0)
    a, b = rng.random(800) < 0.9, rng.random(800) < 0.88
    r = paired_bootstrap(a, b, 2000)
    assert r["delta_pp"] == pytest.approx(100 * (a.mean() - b.mean()))
    assert r["ci_lo_pp"] <= r["delta_pp"] <= r["ci_hi_pp"]
    assert paired_bootstrap(a, a, 500)["p"] == 1.0


def test_sign_tests_reproduce_draft_table_viii():
    assert sign_test([1.66, 1.22, 3.08]) == pytest.approx(0.25)
    assert sign_flip_test([1.66, 1.22, 3.08]) == pytest.approx(0.25)
    assert sign_test([-2.35, 5.50, 1.96]) == pytest.approx(1.0)
    # the draft labels this "sign-permutation" but reports the sign-test value
    assert sign_flip_test([-2.35, 5.50, 1.96]) == pytest.approx(0.75)
    assert min_attainable_p(3) == 0.25


def test_bh_matches_manual():
    p = [0.01, 0.04, 0.03, 0.5]
    adj, rej = benjamini_hochberg(p)
    assert np.allclose(adj, [0.04, 0.16 / 3, 0.16 / 3, 0.5])  # step-up: p_(i)*m/i, cumulative min
    assert list(rej) == [True, False, False, False]


def test_make_tables_leakage_end_to_end(tmp_path):
    """Feeds make_tables.py a fake run_leakage.py output (no torch needed)."""
    import json
    import subprocess
    rng = np.random.default_rng(0)
    n, arrays, conds = 600, {}, {}
    groups = [("imagenet", None, 0.88), ("random", 0, 0.66), ("random", 1, 0.67),
              ("m2cl_6tap", 0, 0.90), ("m2_6tap", 0, 0.86)]
    for g, s, p in groups:
        key = f"{g}|{s}"
        layers = {}
        for L in (["backbone"] + (["eb1", "eb_concat"] if g.startswith("m2") else [])):
            corr = rng.random((3, n)) < p
            arrays[f"sketch|{key}|{L}"] = corr
            accs = 100 * corr.mean(1)
            layers[L] = {"domain": {"acc_mean": accs.mean(), "acc_sd_probe_seeds": accs.std(ddof=1),
                                    "acc_range_probe_seeds": np.ptp(accs), "acc_per_probe_seed": list(accs)},
                         "class_source_test_acc": 90.0, "class_target_acc": 70.0}
        conds[key] = {"group": g, "seed": s, "file": "x", "layers": layers}
    summ = {"config": {"probe_seeds": [0, 1, 2]},
            "targets": {"sketch": {"majority_baseline": 47.0, "conditions": conds}}}
    (tmp_path / "summary.json").write_text(json.dumps(summ))
    np.savez_compressed(tmp_path / "correctness.npz", **arrays)
    root = os.path.join(os.path.dirname(__file__), "..")
    out = tmp_path / "tables"
    subprocess.run([sys.executable, "scripts/make_tables.py", "--leakage", str(tmp_path), "--comparisons",
                    "configs/comparisons.yaml", "--n_boot", "200", "--out", str(out)], check=True, cwd=root)
    md = (out / "tables.md").read_text()
    for must in ("noise floor", "random init vs ImageNet", "M2-CL vs M2", "Multi-seed ERM", "Sanity check"):
        assert must in md


def test_index_pacs_rejects_dct_copy(tmp_path):
    # Audit A0: the draft's probes silently read dct_*.jpg files instead of RGB PACS.
    for d in PACS_DOMAINS:
        (tmp_path / d / "dog").mkdir(parents=True)
        (tmp_path / d / "dog" / "dct_pic_001.jpg").write_bytes(b"")
    with pytest.raises(ValueError, match="dct_"):
        index_pacs(str(tmp_path))
    assert len(index_pacs(str(tmp_path), allow_dct=True)[1]["sketch"]) == 1
