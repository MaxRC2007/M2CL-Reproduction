"""Round-5 runner and analysis, tested against a stand-in for DomainBed's train.py (no GPU, no download)."""
import json
import os
import sys
import textwrap

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze_round5 as an  # noqa: E402
import run_round4_authors as r4  # noqa: E402
import run_round5_authors as rr  # noqa: E402

FAKE_TRAIN = textwrap.dedent('''
    """Stand-in for domainbed.scripts.train: same arguments, same output files, made-up accuracies."""
    import argparse, json, os, torch
    p = argparse.ArgumentParser()
    for k in ("lr", "temp", "lparam"): p.add_argument("--" + k, type=float)
    for k in ("data_dir", "dataset", "algorithm", "hparams", "output_dir"): p.add_argument("--" + k)
    for k in ("hparams_seed", "trial_seed", "seed", "steps", "checkpoint_freq"): p.add_argument("--" + k, type=int)
    p.add_argument("--test_envs", type=int, nargs="+")
    a = p.parse_args()
    assert a.dataset == "VLCS" and os.path.isdir(os.path.join(a.data_dir, "VLCS"))
    hp = {"resnet18": False, "batch_size": 32, "data_augmentation": True, "lr": 5e-5}
    hp.update(json.loads(a.hparams)); hp.update(lr=a.lr, temp=a.temp, lparam=a.lparam)
    args = dict(vars(a), holdout_fraction=0.2, task="domain_generalization", uda_holdout_fraction=0)
    n, f = a.steps or 5001, a.checkpoint_freq or 300
    os.makedirs(a.output_dir, exist_ok=True)
    t, s = a.test_envs[0], a.seed
    with open(os.path.join(a.output_dir, "results.jsonl"), "w") as out:
        for step in [x for x in range(n) if x % f == 0 or x == n - 1]:
            r = {"step": step, "step_time": 0.01, "mem_gb": 3.0, "hparams": hp, "args": args}
            for j in range(4):
                r[f"env{j}_in_acc"] = 0.60 + 0.01 * t + 0.002 * s + 0.002 * (a.lparam > 0) + step / 1e6
                r[f"env{j}_out_acc"] = 0.90 + step / 1e5 - (step == 600) * 0.5
            out.write(json.dumps(r) + "\\n")
    g = torch.Generator().manual_seed(100 * t + s)
    w = torch.randn(50, generator=g)
    sd = {"network.layer1.weight": w + (0.3 if a.lparam > 0 else 0.0),
          "network.ext_block1.comp.weight": torch.randn(5, generator=g) + 5 * a.lparam,
          "network.bn1.running_mean": torch.randn(3, generator=g)}
    torch.save({"model_dict": sd, "args": args}, os.path.join(a.output_dir, "model.pkl"))
    open(os.path.join(a.output_dir, "done"), "w").write("done")
''')


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    authors = tmp_path / "authors"
    (authors / "domainbed" / "scripts").mkdir(parents=True)
    (authors / "domainbed" / "__init__.py").write_text("")
    (authors / "domainbed" / "scripts" / "__init__.py").write_text("")
    (authors / "domainbed" / "scripts" / "train.py").write_text(FAKE_TRAIN)
    data = tmp_path / "data" / "VLCS_raw"
    (data / "VLCS").mkdir(parents=True)
    monkeypatch.setattr(r4, "authors_code", lambda base: str(authors))
    monkeypatch.setattr(rr, "get_vlcs", lambda base: str(data))
    monkeypatch.setattr(rr, "image_counts", lambda p: dict(rr.VLCS_COUNTS))
    monkeypatch.setattr(rr, "record_truncated", lambda vlcs, out: None)
    monkeypatch.setattr(r4, "versions", lambda: {"wilds": "2.0.0", "timm": "0.9.16"})
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: {})
    monkeypatch.setattr(r4, "job_poll_seconds", 0.05)
    return tmp_path


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_round5_authors.py", *args, "--no_gpu_check"])
    rr.main()


def test_command_differs_from_round4_only_in_the_dataset():
    c5 = rr.train_command("0.01", 2, 1, "/d", "/o")
    c4 = r4.train_command("0.01", 2, 1, "/d", "/o")
    i = c5.index("--dataset")
    assert c5[i + 1] == "VLCS" and c4[i + 1] == "PACS"
    assert c5[:i + 1] + c5[i + 2:] == c4[:i + 1] + c4[i + 2:]
    assert "--steps" not in c5 and "--batch_size" not in c5 and "--checkpoint_freq" not in c5


def test_find_vlcs_checks_name_folders_and_counts(tmp_path):
    vlcs = tmp_path / "VLCS_raw" / "VLCS"
    for dom, n in rr.VLCS_COUNTS.items():
        (vlcs / dom / "dog").mkdir(parents=True)
        for i in range(n):
            (vlcs / dom / "dog" / f"{i}.jpg").write_bytes(b"")
    assert rr.find_vlcs(str(tmp_path)) == str(tmp_path / "VLCS_raw")
    (vlcs / "extra").mkdir()                       # DomainBed would treat it as a fifth domain
    assert rr.find_vlcs(str(tmp_path)) is None
    (vlcs / "extra").rmdir()
    os.remove(vlcs / "SUN09" / "dog" / "0.jpg")
    assert rr.find_vlcs(str(tmp_path)) is None


def test_domain_order_is_domainbeds_sorted_order():
    assert list(an.DOMAINS) == sorted(rr.VLCS_T) and an.PAPER_GAIN == 1.67


def test_full_round_with_stand_in_trainer(fake_env, monkeypatch, capfd):
    out = fake_env / "out"
    run_main(monkeypatch, "--out", str(out), "--data_dir", str(fake_env / "data"), "--parallel", "3")
    text = capfd.readouterr().out
    assert text.count("[done]") == 24 and "DECISION:" in text and "paper's VLCS gain" in text
    rep = json.load(open(out / "round5_analysis_institute.json"))
    assert rep["n_runs_complete"] == 24
    res = rep["H1/H2 lparam 0.01 - 0"]
    assert res["n"] == 12 and abs(res["mean"] - 0.2) < 1e-6        # the stand-in adds 0.2 pp at lparam 0.01
    assert rep["ci_excludes_paper_gain"] is True
    mw = rep["manipulation_weight"]
    assert mw["pairs_checked"] == 12 and mw["active_pairs"] == 12
    env = json.load(open(next((out / "env").glob("institute_round5_*.json"))))
    assert env["round"] == 5 and "VLCS" in env["command_fixed_args"]
    assert not os.path.exists(out / ".round5.lock")
    run_main(monkeypatch, "--out", str(out), "--data_dir", str(fake_env / "data"))
    assert "24 of 24 round-5 runs already finished" in capfd.readouterr().out


def test_unfinished_run_is_restarted_from_scratch(fake_env, monkeypatch, capfd):
    out = fake_env / "out"
    d = rr.run_dir(str(out), "0.0", 0, 0)
    os.makedirs(d)
    open(os.path.join(d, "results.jsonl"), "w").write('{"step": 0}\n')   # interrupted: no "done"
    run_main(monkeypatch, "--out", str(out), "--data_dir", str(fake_env / "data"))
    assert "[restart] lparam0.0_test0_seed0" in capfd.readouterr().out
    lines = open(os.path.join(d, "results.jsonl")).read().splitlines()
    assert len(lines) == 18 and json.loads(lines[0])["args"]["lparam"] == 0.0


def test_analysis_refuses_other_settings_and_round4_files(fake_env, monkeypatch, capfd):
    out = fake_env / "out"
    run_main(monkeypatch, "--out", str(out), "--data_dir", str(fake_env / "data"), "--parallel", "4")
    capfd.readouterr()
    p = os.path.join(rr.run_dir(str(out), "0.01", 3, 2), "results.jsonl")
    recs = [json.loads(x) for x in open(p)]
    for r in recs:
        r["args"]["dataset"] = "PACS"
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(SystemExit, match="differ from the pre-registration"):
        monkeypatch.setattr(sys, "argv", ["analyze_round5.py", str(out), "--no_weights"])
        an.main()
    # round-4 folders are never read as round 5
    assert an.RUN_RE.search("round4/lparam_0.01/test0_seed0/results.jsonl") is None


def test_smoke_reports_an_estimate(fake_env, monkeypatch, capfd):
    run_main(monkeypatch, "--out", str(fake_env / "out"), "--data_dir", str(fake_env / "data"), "--smoke")
    text = capfd.readouterr().out
    assert "Smoke test passed" in text and "for all 24" in text
    assert not os.path.exists(fake_env / "out" / "round5")
