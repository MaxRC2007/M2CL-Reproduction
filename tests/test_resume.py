"""An interrupted training run must end up exactly where an uninterrupted one does."""
import json
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import m2cl.engine as engine  # noqa: E402
from m2cl.engine import RunConfig, train_run  # noqa: E402

DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


class Disconnect(Exception):
    pass


@pytest.fixture(scope="module")
def pacs(tmp_path_factory):
    root = tmp_path_factory.mktemp("pacs")
    rng = np.random.default_rng(0)
    for di, d in enumerate(DOMAINS):
        for ci, c in enumerate(CLASSES):
            os.makedirs(root / d / c)
            for i in range(6):
                img = rng.integers(0, 60, (32, 32, 3)).astype(np.uint8)
                img[..., di % 3] += 120
                img[ci * 4:(ci + 1) * 4] = 255
                Image.fromarray(img).save(root / d / c / f"{i}.jpg")
    return str(root)


def _cfg():
    return RunConfig(target="sketch", model="m2", arch_taps=6, alpha=0.0, epochs=3, batch_size=16, img_size=32,
                     pretrained=False, num_workers=0, seed=3)


def _interrupt_during_epoch(monkeypatch, epoch, n_source_domains=3):
    """Fail at the start of validation of `epoch`, as if the session died mid-epoch."""
    calls = {"n": 0}
    real = engine.evaluate

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == (epoch - 1) * n_source_domains + 1:
            raise Disconnect
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "evaluate", flaky)


def _same_weights(path_a, path_b):
    a = torch.load(path_a, weights_only=False)["model_state"]
    b = torch.load(path_b, weights_only=False)["model_state"]
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def _history(out_dir, name):
    with open(os.path.join(out_dir, name, "result.json")) as f:
        return json.load(f)


def test_interrupted_run_matches_uninterrupted(pacs, tmp_path, monkeypatch):
    cfg = _cfg()
    straight = str(tmp_path / "straight")
    train_run(cfg, pacs, straight)

    resumed = str(tmp_path / "resumed")
    with monkeypatch.context() as m:
        _interrupt_during_epoch(m, epoch=3)
        with pytest.raises(Disconnect):
            train_run(cfg, pacs, resumed)
    run_dir = os.path.join(resumed, cfg.run_name)
    assert os.path.exists(os.path.join(run_dir, "latest.pt"))
    assert not os.path.exists(os.path.join(run_dir, "result.json"))

    res = train_run(cfg, pacs, resumed)
    base = _history(straight, cfg.run_name)
    assert [h["loss"] for h in res["history"]] == [h["loss"] for h in base["history"]]
    assert res["test_acc"] == base["test_acc"]
    assert _same_weights(os.path.join(run_dir, "best.pt"), os.path.join(straight, cfg.run_name, "best.pt"))
    assert sorted(os.listdir(run_dir)) == ["best.pt", "result.json"]  # checkpoints cleaned up


def test_damaged_checkpoint_falls_back_to_previous(pacs, tmp_path, monkeypatch, capsys):
    cfg = _cfg()
    out = str(tmp_path / "damaged")
    with monkeypatch.context() as m:
        _interrupt_during_epoch(m, epoch=3)
        with pytest.raises(Disconnect):
            train_run(cfg, pacs, out)
    run_dir = os.path.join(out, cfg.run_name)
    with open(os.path.join(run_dir, "latest.pt"), "wb") as f:
        f.write(b"cut short")
    train_run(cfg, pacs, out)
    text = capsys.readouterr().out
    assert "cannot read" in text and "from epoch 2" in text and "[done]" in text


def test_finished_run_is_skipped_even_with_stray_checkpoint(pacs, tmp_path, capsys):
    cfg = _cfg()
    out = str(tmp_path / "done")
    train_run(cfg, pacs, out)
    open(os.path.join(out, cfg.run_name, "latest.pt"), "wb").close()
    train_run(cfg, pacs, out)
    assert "[skip]" in capsys.readouterr().out
