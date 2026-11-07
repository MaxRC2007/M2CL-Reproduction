"""Model / loss / probe tests (need torch; skipped otherwise). Offline: no ImageNet download."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch.nn.functional as F  # noqa: E402

from m2cl.losses import m2cl_contrastive_loss  # noqa: E402
from m2cl.models import M2, build_model  # noqa: E402
from m2cl.probe import train_linear_probe  # noqa: E402


def notebook_loss(reps, labels, tau=1.0):
    """Verbatim copy of the corrected loss in the training notebook (cell 13)."""
    total = torch.tensor(0.0)
    for u in reps:
        u = F.normalize(u, dim=1)
        sim = torch.mm(u, u.t()) / tau
        off = ~torch.eye(u.size(0), dtype=torch.bool)
        denom = torch.exp(sim[off]).sum()
        layer = torch.tensor(0.0)
        for c in labels.unique():
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if len(idx) < 2:
                continue
            ps = sim[idx][:, idx]
            numer = torch.exp(ps[~torch.eye(len(idx), dtype=torch.bool)]).sum()
            layer = layer - torch.log(numer / denom + 1e-12)
        total = total + layer
    return total


def test_loss_matches_notebook_and_bug_factor():
    torch.manual_seed(0)
    labels = torch.arange(7).repeat(19)[:128]
    reps = [torch.randn(128, d) for d in (384, 384, 256, 256)]
    s = m2cl_contrastive_loss(reps, labels, reduction="sum")
    assert torch.allclose(s, notebook_loss(reps, labels), rtol=1e-4)
    m = m2cl_contrastive_loss(reps, labels, reduction="mean")
    assert float(s / m) == pytest.approx(7 * 4, rel=0.05)  # audit: ~C*M = 28x


@pytest.mark.parametrize("taps,dim", [(4, 1792), (6, 2560)])
def test_m2_shapes_and_feature_dims(taps, dim):
    net = M2(7, use_cl=True, arch_taps=taps, pretrained=False).eval()
    assert net.classifier.in_features == dim
    x = torch.randn(2, 3, 224, 224)
    logits, reps = net(x)
    assert logits.shape == (2, 7) and len(reps) == taps
    f = net.forward_features(x)
    assert f["backbone"].shape == (2, 512) and f["eb_concat"].shape == (2, dim - 512)


def test_state_dict_keys_match_notebook_names():
    keys = set(build_model("m2cl", 7, 6, pretrained=False).state_dict())
    for k in ("layer0.0.weight", "eb1a.pipes.0.conv.weight", "eb2a.pipes.2.mlp.1.weight",
              "eb4.pipes.1.bn.running_mean", "classifier.weight"):
        assert k in keys
    assert "features.4.0.conv1.weight" in build_model("erm", pretrained=False).state_dict()


def test_erm_and_m2_backbone_features_agree_when_weights_shared():
    erm = build_model("erm", pretrained=False).eval()
    m2 = build_model("m2", arch_taps=6, pretrained=False).eval()
    m2.layer0.load_state_dict(torch.nn.Sequential(*list(erm.features)[:4]).state_dict())
    for i, n in zip(range(4, 8), ("layer1", "layer2", "layer3", "layer4")):
        getattr(m2, n).load_state_dict(erm.features[i].state_dict())
    x = torch.randn(2, 3, 128, 128)
    assert torch.allclose(erm.forward_features(x)["backbone"], m2.forward_features(x)["backbone"], atol=1e-5)


def test_probe_is_deterministic_and_learns():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    x = (rng.normal(size=(600, 32)) + rng.normal(size=(3, 32))[y]).astype("float32")  # separable, zero-centred classes
    p1 = train_linear_probe(x[:400], y[:400], x[400:], 3, seed=3, epochs=30)
    p2 = train_linear_probe(x[:400], y[:400], x[400:], 3, seed=3, epochs=30)
    assert (p1 == p2).all() and (p1 == y[400:]).mean() > 0.8
