"""ERM, M2 and M2-CL (4 or 6 extraction taps) on ResNet-18.

Module and parameter names are kept identical to the training notebook and the
probe scripts, so every existing checkpoint loads with strict=True.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def _resnet18(pretrained: bool) -> nn.Module:
    return resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)


def _gap(x: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x, 1).flatten(1)


class ERM(nn.Module):
    """Plain ResNet-18 + linear head. Also used (head unused) for the ImageNet-only
    and random-init probe controls."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        bb = _resnet18(pretrained)
        self.features = nn.Sequential(*list(bb.children())[:-1])  # conv1..layer4, avgpool
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1)), None

    @torch.no_grad()
    def forward_features(self, x) -> Dict[str, torch.Tensor]:
        f = self.features
        x = f[3](f[2](f[1](f[0](x))))
        out = {}
        for i, name in zip(range(4, 8), ["layer1", "layer2", "layer3", "backbone"]):
            x = f[i](x)
            out[name] = _gap(x)  # "backbone" = GAP(layer4) = the 512-d feature the head sees
        return out


class ConcentrationPipeline(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, pool_size: int, mlp_dim: int = 128):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.drop = nn.Dropout2d(p=0.5)
        self.pool = nn.AdaptiveMaxPool2d(pool_size)
        self.mlp = nn.Sequential(nn.Flatten(),
                                 nn.Linear(mid_channels * pool_size * pool_size, mlp_dim),
                                 nn.ReLU(inplace=True))
        self.out_dim = mlp_dim

    def forward(self, x):
        return self.mlp(self.pool(self.drop(F.relu(self.bn(self.conv(x)), inplace=True))))


class ExtractionBlock(nn.Module):
    def __init__(self, in_channels: int, r: int = 4, pool_sizes=(4, 2), mlp_dim: int = 128):
        super().__init__()
        mid = max(in_channels // r, 8)
        self.pipes = nn.ModuleList([ConcentrationPipeline(in_channels, mid, ps, mlp_dim)
                                    for ps in pool_sizes])
        self.out_dim = mlp_dim * len(pool_sizes)

    def forward(self, x):
        return torch.cat([p(x) for p in self.pipes], dim=1)


class M2(nn.Module):
    """M2 (use_cl=False) / M2-CL (use_cl=True). arch_taps=4: one block after each
    ResNet stage. arch_taps=6: additionally taps the first residual sub-block of
    layer1 and layer2 (eb1a, eb2a)."""

    def __init__(self, num_classes: int = 7, use_cl: bool = False, mlp_dim: int = 128,
                 arch_taps: int = 4, pretrained: bool = True):
        super().__init__()
        if arch_taps not in (4, 6):
            raise ValueError("arch_taps must be 4 or 6")
        self.use_cl, self.arch_taps = use_cl, arch_taps
        bb = _resnet18(pretrained)
        self.layer0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = bb.layer1, bb.layer2, bb.layer3, bb.layer4
        self.avgpool = bb.avgpool
        self.eb1 = ExtractionBlock(64, 4, (8, 4, 2), mlp_dim)
        self.eb2 = ExtractionBlock(128, 4, (8, 4, 2), mlp_dim)
        self.eb3 = ExtractionBlock(256, 4, (7, 3), mlp_dim)
        self.eb4 = ExtractionBlock(512, 4, (7, 3), mlp_dim)
        total = sum(b.out_dim for b in (self.eb1, self.eb2, self.eb3, self.eb4)) + 512
        if arch_taps == 6:
            self.eb1a = ExtractionBlock(64, 4, (8, 4, 2), mlp_dim)
            self.eb2a = ExtractionBlock(128, 4, (8, 4, 2), mlp_dim)
            total += self.eb1a.out_dim + self.eb2a.out_dim
        self.classifier = nn.Linear(total, num_classes)

    @property
    def tap_names(self) -> List[str]:
        return (["eb1a", "eb1", "eb2a", "eb2", "eb3", "eb4"] if self.arch_taps == 6
                else ["eb1", "eb2", "eb3", "eb4"])

    def _backbone(self, x) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        x = self.layer0(x)
        maps = {}
        if self.arch_taps == 6:
            maps["eb1a"] = self.layer1[0](x)
            maps["layer1"] = self.layer1[1](maps["eb1a"])
            maps["eb2a"] = self.layer2[0](maps["layer1"])
            maps["layer2"] = self.layer2[1](maps["eb2a"])
        else:
            maps["layer1"] = self.layer1(x)
            maps["layer2"] = self.layer2(maps["layer1"])
        maps["layer3"] = self.layer3(maps["layer2"])
        maps["layer4"] = self.layer4(maps["layer3"])
        src = {"eb1": "layer1", "eb2": "layer2", "eb3": "layer3", "eb4": "layer4",
               "eb1a": "eb1a", "eb2a": "eb2a"}
        taps = {n: getattr(self, n)(maps[src[n]]) for n in self.tap_names}
        return maps, taps

    def forward(self, x):
        maps, taps = self._backbone(x)
        feats = [taps[n] for n in self.tap_names]
        gap = self.avgpool(maps["layer4"]).flatten(1)
        logits = self.classifier(torch.cat(feats + [gap], dim=1))
        return logits, (feats if self.use_cl else None)

    @torch.no_grad()
    def forward_features(self, x) -> Dict[str, torch.Tensor]:
        maps, taps = self._backbone(x)
        out = {k: _gap(maps[k]) for k in ("layer1", "layer2", "layer3")}
        out["backbone"] = _gap(maps["layer4"])
        for n in self.tap_names:
            out[n] = taps[n]
        out["eb_concat"] = torch.cat([taps[n] for n in self.tap_names], dim=1)
        return out


def build_model(name: str, num_classes: int = 7, arch_taps: int = 4, pretrained: bool = True) -> nn.Module:
    if name == "erm":
        return ERM(num_classes, pretrained)
    if name == "m2":
        return M2(num_classes, use_cl=False, arch_taps=arch_taps, pretrained=pretrained)
    if name == "m2cl":
        return M2(num_classes, use_cl=True, arch_taps=arch_taps, pretrained=pretrained)
    raise ValueError(f"unknown model {name}")


def load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    """Accepts notebook checkpoints ({'model_state': ...}) and raw state_dicts."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("model_state", raw) if isinstance(raw, dict) else raw
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{path}: checkpoint does not match model.\n"
                           f"  missing={missing[:8]}...\n  unexpected={unexpected[:8]}...\n"
                           "Check --model / --taps for this checkpoint.")
    return model
