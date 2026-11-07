"""Torch datasets, transforms and LODO loaders."""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .splits import Sample, index_pacs, split_train_val

# VLCS (DomainBed's copy) contains truncated JPEGs. Decode them as far as the file goes, exactly as
# DomainBed and the authors' code do (domainbed/datasets.py). No PACS image is truncated, and every PACS
# image decodes to identical pixels with or without this flag (docs/AUDIT.md, F17).
ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_transforms(augmentation: str = "baseline", img_size: int = 224):
    norm = transforms.Normalize(MEAN, STD)
    if augmentation == "baseline":
        train = [transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(0.3, 0.3, 0.3, 0.1), transforms.RandomGrayscale(p=0.1)]
    elif augmentation == "strong_crop":
        train = [transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(0.3, 0.3, 0.3, 0.1), transforms.RandomGrayscale(p=0.1)]
    elif augmentation == "shape_oriented":
        train = [transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(),
                 transforms.Grayscale(num_output_channels=3)]
    else:
        raise ValueError(augmentation)
    test = [transforms.Resize(int(img_size * 256 / 224)), transforms.CenterCrop(img_size)]
    return (transforms.Compose(train + [transforms.ToTensor(), norm]),
            transforms.Compose(test + [transforms.ToTensor(), norm]))


class PACSDataset(Dataset):
    """Returns (image, class_idx, domain_idx)."""

    def __init__(self, samples: Sequence[Sample], transform):
        self.samples, self.transform = list(samples), transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, c, d = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), c, d


def _worker_init(worker_id: int):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(samples, transform, batch_size, shuffle, seed=0, num_workers=2):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(PACSDataset(samples, transform), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=_worker_init, generator=g)


def make_lodo_loaders(root: str, target: str, augmentation="baseline", img_size=224,
                      batch_size=128, val_fraction=0.2, seed=0, num_workers=2):
    classes, by_dom = index_pacs(root)
    if target not in by_dom:
        raise ValueError(f"Target {target!r} is not a domain of {root}: {list(by_dom)}")
    sources = [d for d in by_dom if d != target]
    train, val = split_train_val(by_dom, sources, val_fraction, seed)
    tr_tf, te_tf = get_transforms(augmentation, img_size)
    loaders = {
        "train": make_loader(train, tr_tf, batch_size, True, seed, num_workers),
        "val": {d: make_loader(v, te_tf, batch_size, False, seed, num_workers) for d, v in val.items()},
        "test": make_loader(by_dom[target], te_tf, batch_size, False, seed, num_workers),
    }
    info = {"classes": classes, "sources": sources, "n_train": len(train),
            "n_val": {d: len(v) for d, v in val.items()}, "n_test": len(by_dom[target])}
    return loaders, info
