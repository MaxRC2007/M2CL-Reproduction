"""VLCS support: detection, indexing, disjoint splits, loaders; PACS seeds unchanged."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from m2cl.splits import (PACS_DOMAINS, VLCS_DOMAINS, detect_domains, domain_seed,  # noqa: E402
                         index_pacs, split_train_val)

VLCS_CLASSES = ["bird", "car", "chair", "dog", "person"]


def fake_vlcs(root, n=6):
    for d in VLCS_DOMAINS:
        for c in VLCS_CLASSES:
            os.makedirs(os.path.join(root, d, c))
            for i in range(n):
                Image.fromarray(np.full((32, 32, 3), i * 10, np.uint8)).save(os.path.join(root, d, c, f"{i}.jpg"))


def test_pacs_domain_seeds_unchanged():
    for s in (0, 1, 2):
        for i, d in enumerate(PACS_DOMAINS):
            assert domain_seed(s, d) == s * 1000 + i


def test_vlcs_index_and_split(tmp_path):
    root = str(tmp_path)
    fake_vlcs(root)
    assert detect_domains(root) == VLCS_DOMAINS
    classes, by_dom = index_pacs(root)
    assert classes == VLCS_CLASSES and list(by_dom) == list(VLCS_DOMAINS)
    assert all(len(v) == 30 for v in by_dom.values())
    assert {s[2] for s in by_dom["SUN09"]} == {2}
    for target in VLCS_DOMAINS:
        sources = [d for d in VLCS_DOMAINS if d != target]
        train, val = split_train_val(by_dom, sources, 0.2, seed=1)
        train_paths = {s[0] for s in train}
        assert all(not ({s[0] for s in v} & train_paths) for v in val.values())
        assert len(train) + sum(map(len, val.values())) == 90


def test_vlcs_loaders(tmp_path):
    from m2cl.data import make_lodo_loaders
    root = str(tmp_path)
    fake_vlcs(root)
    loaders, info = make_lodo_loaders(root, "LabelMe", img_size=32, batch_size=8, num_workers=0)
    assert info["classes"] == VLCS_CLASSES and info["sources"] == ["Caltech101", "SUN09", "VOC2007"]
    x, y, d = next(iter(loaders["train"]))
    assert x.shape[1:] == (3, 32, 32) and int(y.max()) < 5


def test_truncated_jpeg_loads(tmp_path):
    """VLCS has truncated JPEGs (DomainBed's copy); the loader must decode them, as DomainBed does."""
    from m2cl.data import PACSDataset
    from torchvision import transforms
    rng = np.random.default_rng(0)
    p = str(tmp_path / "x.jpg")
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(p, quality=95)
    data = open(p, "rb").read()
    open(p, "wb").write(data[: len(data) * 2 // 3])          # cut the file, as in VLCS
    x, c, d = PACSDataset([(p, 3, 1)], transforms.ToTensor())[0]
    assert tuple(x.shape) == (3, 64, 64) and (c, d) == (3, 1)
