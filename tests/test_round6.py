"""Round-6 runner and analysis, tested against a stand-in for DomainBed's train.py (no GPU, no download)."""
import json
import os
import sys
import textwrap

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze_round6 as an  # noqa: E402
import run_round4_authors as r4  # noqa: E402
import run_round6_authors as rr  # noqa: E402

# The stand-in adds 0.2 pp at every checkpoint when lparam > 0, and another 0.3 pp at the final step only.
# Validation accuracy peaks at step 4800, so the selected checkpoint differs from the final one:
# final-step difference 0.5 pp, validation-selected 0.2 pp, mean of the last five checkpoints 0.26 pp.
FAKE_TRAIN = textwrap.dedent('''
    """Stand-in for domainbed.scripts.train: same arguments, same output files, made-up accuracies."""
    import argparse, json, os, torch
    p = argparse.ArgumentParser()
    for k in ("lr", "temp", "lparam"): p.add_argument("--" + k, type=float)
    for k in ("data_dir", "dataset", "algorithm", "hparams", "output_dir"): p.add_argument("--" + k)
    for k in ("hparams_seed", "trial_seed", "seed", "steps", "checkpoint_freq"): p.add_argument("--" + k, type=int)
    p.add_argument("--test_envs", type=int, nargs="+")
    a = p.parse_args()
    assert a.dataset == "PACS" and os.path.isdir(os.path.join(a.data_dir, "PACS"))
    hp = {"resnet18": False, "batch_size": 32, "data_augmentation": True, "lr": 5e-5}
    hp.update(json.loads(a.hparams)); hp.update(lr=a.lr, temp=a.temp, lparam=a.lparam)
    args = dict(vars(a), holdout_fraction=0.2, task="domain_generalization", uda_holdout_fraction=0)
    n, f = a.steps or 5001, a.checkpoint_freq or 300
    os.makedirs(a.output_dir, exist_ok=True)
    t, s, on = a.test_envs[0], a.seed, a.lparam > 0
    with open(os.path.join(a.output_dir, "results.jsonl"), "w") as out:
        for step in [x for x in range(n) if x % f == 0 or x == n - 1]:
            r = {"step": step, "step_time": 0.01, "mem_gb": 3.0, "hparams": hp, "args": args}
            for j in range(4):
                r[f"env{j}_in_acc"] = (0.60 + 0.01 * t + 0.002 * s + 0.002 * on + 0.003 * on * (step == n - 1)
                                       + step / 1e6)
                r[f"env{j}_out_acc"] = 0.90 + step / 1e5 - 0.01 * (step == n - 1)
            out.write(json.dumps(r) + "\\n")
    g = torch.Generator().manual_seed(100 * t + s)
    w = torch.randn(50, generator=g)
    sd = {"network.layer1.weight": w + (0.3 if on else 0.0),
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
    monkeypatch.setattr(r4, "authors_code", lambda base: str(authors))
    monkeypatch.setattr(r4, "find_pacs", lambda base: str(data))
    monkeypatch.setattr(r4, "pacs_counts", lambda p: dict(r4.PACS_COUNTS))
    monkeypatch.setattr(r4, "versions", lambda: {"wilds": "2.0.0", "timm": "0.9.16"})
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: {})
    monkeypatch.setattr(r4, "job_poll_seconds", 0.05)
    return tmp_path


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_round6_authors.py", *args, "--no_gpu_check"])
    rr.main()


def test_command_is_round4s_and_seeds_are_new():
    assert rr.train_command("0.01", 2, 5, "/d", "/o") == r4.train_command("0.01", 2, 5, "/d", "/o")
    assert rr.FIXED == r4.FIXED and "PACS" in rr.FIXED
    assert rr.SEEDS == (3, 4, 5, 6, 7) and not set(rr.SEEDS) & set(r4.SEEDS)
    assert rr.N_RUNS == 40 and an.N_PAIRS == 20 and an.ACTIVE_MIN_PAIRS == 17 and an.FINAL_STEP == 5000
    assert an.SEEDS == rr.SEEDS and an.PAPER_GAIN == 0.71


def test_full_round_uses_the_final_step(fake_env, monkeypatch, capfd):
    out = fake_env / "out"
    run_main(monkeypatch, "--out", str(out), "--data_dir", str(fake_env / "data"), "--parallel", "4")
    text = capfd.readouterr().out
    assert text.count("[done]") == 40 and "DECISION:" in text and "paper's PACS gain" in text
    rep = json.load(open(out / "round6_analysis_institute.json"))
    assert rep["n_runs_complete"] == 40
    prim = rep["H1/H2 final step, lparam 0.01 - 0"]
    assert prim["n"] == 20 and abs(prim["mean"] - 0.5) < 1e-6          # final step, not the selected checkpoint
    assert abs(rep["validation-selected checkpoint, lparam 0.01 - 0"]["mean"] - 0.2) < 1e-6
    assert abs(rep["mean of the last 5 checkpoints, lparam 0.01 - 0"]["mean"] - 0.26) < 1e-6
    assert all(v["selected_step"] == 4800 for v in rep["runs"].values())
    assert rep["decision"].startswith("EFFECT")
    assert rep["ci_contains_zero"] is False and rep["ci_contains_paper_gain"] is False
    mw = rep["manipulation_weight"]
    assert mw["pairs_checked"] == 20 and mw["active_pairs"] == 20
    env = json.load(open(next((out / "env").glob("institute_round6_*.json"))))
    assert env["round"] == 6 and env["seeds"] == [3, 4, 5, 6, 7] and "PACS" in env["command_fixed_args"]
    assert not os.path.exists(out / ".round6.lock")
    run_main(monkeypatch, "--out", str(out), "--data_dir", str(fake_env / "data"))
    assert "40 of 40 round-6 runs already finished" in capfd.readouterr().out


def test_unfinished_run_is_restarted_from_scratch(fake_env, monkeypatch, capfd):
    out = fake_env / "out"
    d = rr.run_dir(str(out), "0.0", 0, 3)
    os.makedirs(d)
    open(os.path.join(d, "results.jsonl"), "w").write('{"step": 0}\n')   # interrupted: no "done"
    run_main(monkeypatch, "--out", str(out), "--data_dir", str(fake_env / "data"), "--parallel", "4")
    assert "[restart] lparam0.0_test0_seed3" in capfd.readouterr().out
    lines = open(os.path.join(d, "results.jsonl")).read().splitlines()
    assert len(lines) == 18 and json.loads(lines[0])["args"]["lparam"] == 0.0


def test_no_decision_before_all_pairs(fake_env, monkeypatch, capfd):
    out = fake_env / "out"
    run_main(monkeypatch, "--out", str(out), "--data_dir", str(fake_env / "data"), "--parallel", "4")
    capfd.readouterr()
    os.remove(os.path.join(rr.run_dir(str(out), "0.01", 1, 7), "done"))
    monkeypatch.setattr(sys, "argv", ["analyze_round6.py", str(out), "--no_weights", "--json", str(out / "a.json")])
    an.main()
    rep = json.load(open(out / "a.json"))
    assert rep["decision"].startswith("INCOMPLETE: 19 of 20")


def test_analysis_refuses_other_settings_seeds_and_old_rounds(fake_env, monkeypatch, capfd):
    out = fake_env / "out"
    run_main(monkeypatch, "--out", str(out), "--data_dir", str(fake_env / "data"), "--parallel", "4")
    capfd.readouterr()
    p = os.path.join(rr.run_dir(str(out), "0.01", 3, 6), "results.jsonl")
    recs = [json.loads(x) for x in open(p)]
    for r in recs:
        r["args"]["lr"] = 0.0005
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    monkeypatch.setattr(sys, "argv", ["analyze_round6.py", str(out), "--no_weights"])
    with pytest.raises(SystemExit, match="differ from the pre-registration"):
        an.main()
    for r in recs:
        r["args"]["lr"] = 0.001
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    # a run with an old seed placed in the round-6 tree is refused
    old = rr.run_dir(str(out), "0.0", 0, 1)
    os.makedirs(old)
    for f in ("results.jsonl", "done"):
        open(os.path.join(old, f), "w").write(open(os.path.join(rr.run_dir(str(out), "0.0", 0, 3), f)).read())
    with pytest.raises(SystemExit, match="not one of the pre-registered seeds"):
        an.main()
    # round-4 and round-5 folders are never read as round 6
    assert an.RUN_RE.search("round4/lparam_0.01/test0_seed0/results.jsonl") is None
    assert an.RUN_RE.search("round5/lparam_0.01/test0_seed0/results.jsonl") is None


def test_smoke_reports_an_estimate(fake_env, monkeypatch, capfd):
    run_main(monkeypatch, "--out", str(fake_env / "out"), "--data_dir", str(fake_env / "data"), "--smoke")
    text = capfd.readouterr().out
    assert "Smoke test passed" in text and "for all 40" in text
    assert not os.path.exists(fake_env / "out" / "round6")
