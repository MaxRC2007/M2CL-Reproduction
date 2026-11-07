"""Round-4 runner and analysis, tested against a stand-in for DomainBed's train.py (no GPU, no download)."""
import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze_round4 as an  # noqa: E402
import run_round4_authors as rr  # noqa: E402

FAKE_TRAIN = textwrap.dedent('''
    """Stand-in for domainbed.scripts.train: same arguments, same output files, made-up accuracies."""
    import argparse, json, os, torch
    p = argparse.ArgumentParser()
    for k in ("lr", "temp", "lparam"): p.add_argument("--" + k, type=float)
    for k in ("data_dir", "dataset", "algorithm", "hparams", "output_dir"): p.add_argument("--" + k)
    for k in ("hparams_seed", "trial_seed", "seed", "steps", "checkpoint_freq"): p.add_argument("--" + k, type=int)
    p.add_argument("--test_envs", type=int, nargs="+")
    a = p.parse_args()
    assert os.path.isdir(os.path.join(a.data_dir, "PACS"))
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
                r[f"env{j}_in_acc"] = 0.60 + 0.01 * t + 0.002 * s + 0.005 * (a.lparam > 0) + step / 1e6
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
    data = tmp_path / "data" / "domainbed_pacs"
    (data / "PACS").mkdir(parents=True)
    monkeypatch.setattr(rr, "authors_code", lambda base: str(authors))
    monkeypatch.setattr(rr, "find_pacs", lambda base: str(data))
    monkeypatch.setattr(rr, "pacs_counts", lambda p: dict(rr.PACS_COUNTS))
    monkeypatch.setattr(rr, "versions", lambda: {"wilds": "2.0.0", "timm": "0.9.16"})
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: {})
    monkeypatch.setattr(rr, "job_poll_seconds", 0.05)
    return tmp_path


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_round4_authors.py", *args, "--no_gpu_check"])
    rr.main()


def test_command_has_the_preregistered_settings():
    cmd = rr.train_command("0.01", 2, 1, "/d", "/o")
    s = " ".join(cmd)
    for part in ("-m domainbed.scripts.train", "--dataset PACS", "--algorithm M2CL", "--hparams_seed 0",
                 '{"resnet18": true}', "--lr 0.001", "--temp 1.0", "--lparam 0.01", "--test_envs 2",
                 "--trial_seed 1", "--seed 1"):
        assert part in s
    assert "--steps" not in cmd and "--batch_size" not in cmd and "--checkpoint_freq" not in cmd


def test_find_pacs_checks_folders_and_counts(tmp_path):
    pacs = tmp_path / "x" / "PACS"
    for dom, n in rr.PACS_COUNTS.items():
        (pacs / dom / "dog").mkdir(parents=True)
        for i in range(n):
            (pacs / dom / "dog" / f"{i}.jpg").write_bytes(b"")
    assert rr.find_pacs(str(tmp_path)) == str(tmp_path / "x")
    (pacs / "extra").mkdir()                       # DomainBed would treat it as a fifth domain
    assert rr.find_pacs(str(tmp_path)) is None
    (pacs / "extra").rmdir()
    os.remove(pacs / "sketch" / "dog" / "0.jpg")
    assert rr.find_pacs(str(tmp_path)) is None


def test_selection_is_domainbed_training_domain_validation():
    recs = [{"step": 0, **{f"env{j}_out_acc": v for j, v in enumerate([0.1, 0.5, 0.5, 0.5])},
             **{f"env{j}_in_acc": 0.3 for j in range(4)}},
            {"step": 300, **{f"env{j}_out_acc": v for j, v in enumerate([0.9, 0.7, 0.7, 0.7])},
             **{f"env{j}_in_acc": 0.4 + j / 100 for j in range(4)}},
            {"step": 600, **{f"env{j}_out_acc": v for j, v in enumerate([0.0, 0.7, 0.7, 0.7])},
             **{f"env{j}_in_acc": 0.9 for j in range(4)}}]
    val, test, step = an.select(recs, 0)           # tie at 0.7: the first checkpoint wins; env0 val is ignored
    assert step == 300 and abs(test - 40.0) < 1e-9 and abs(val - 70.0) < 1e-9


def test_full_round_with_stand_in_trainer(fake_env, monkeypatch, capfd):
    out = fake_env / "out"
    run_main(monkeypatch, "--out", str(out), "--data_dir", str(fake_env / "data"), "--parallel", "3")
    text = capfd.readouterr().out
    assert text.count("[done]") == 24 and "DECISION:" in text
    rep = json.load(open(out / "round4_analysis_institute.json"))
    assert rep["n_runs_complete"] == 24
    res = rep["H1/H2 lparam 0.01 - 0"]
    assert res["n"] == 12 and abs(res["mean"] - 0.5) < 1e-6        # the stand-in adds 0.5 pp at lparam 0.01
    assert rep["decision"].startswith("EFFECT")
    mw = rep["manipulation_weight"]
    assert mw["pairs_checked"] == 12 and mw["active_pairs"] == 12
    assert not os.path.exists(out / ".round4.lock")
    # a second start finds everything finished and trains nothing
    run_main(monkeypatch, "--out", str(out), "--data_dir", str(fake_env / "data"))
    assert "24 of 24 round-4 runs already finished" in capfd.readouterr().out


def test_unfinished_run_is_restarted_from_scratch(fake_env, monkeypatch, capfd):
    out = fake_env / "out"
    d = rr.run_dir(str(out), "0.0", 0, 0)
    os.makedirs(d)
    open(os.path.join(d, "results.jsonl"), "w").write('{"step": 0}\n')   # interrupted: no "done"
    run_main(monkeypatch, "--out", str(out), "--data_dir", str(fake_env / "data"))
    text = capfd.readouterr().out
    assert "[restart] lparam0.0_test0_seed0" in text
    lines = open(os.path.join(d, "results.jsonl")).read().splitlines()
    assert len(lines) == 18 and json.loads(lines[0])["args"]["lparam"] == 0.0


def test_analysis_refuses_other_settings(fake_env, monkeypatch, capfd):
    out = fake_env / "out"
    run_main(monkeypatch, "--out", str(out), "--data_dir", str(fake_env / "data"), "--parallel", "4")
    capfd.readouterr()
    p = os.path.join(rr.run_dir(str(out), "0.01", 3, 2), "results.jsonl")
    recs = [json.loads(x) for x in open(p)]
    for r in recs:
        r["args"]["lr"] = 5e-5
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(SystemExit, match="differ from the pre-registration"):
        monkeypatch.setattr(sys, "argv", ["analyze_round4.py", str(out), "--no_weights"])
        an.main()


def test_smoke_reports_an_estimate(fake_env, monkeypatch, capfd):
    run_main(monkeypatch, "--out", str(fake_env / "out"), "--data_dir", str(fake_env / "data"), "--smoke")
    text = capfd.readouterr().out
    assert "Smoke test passed" in text and "for all 24" in text
    assert not os.path.exists(fake_env / "out" / "round4")       # the smoke run never lands in the real folder


def test_failed_run_stops_new_starts(fake_env, monkeypatch, capfd):
    train = os.path.join(rr.authors_code(""), "domainbed", "scripts", "train.py")
    src = open(train).read()
    open(train, "w").write("import sys\nif '--test_envs' in sys.argv and sys.argv[sys.argv.index('--test_envs')+1]"
                           " == '1': sys.exit('boom')\n" + src)
    with pytest.raises(SystemExit, match="failed"):
        run_main(monkeypatch, "--out", str(fake_env / "out"), "--data_dir", str(fake_env / "data"))
    text = capfd.readouterr().out
    assert "[FAILED]" in text and "boom" in text
    assert not os.path.exists(fake_env / "out" / ".round4.lock")
