"""Dataset indexing and train/val/probe splits (NumPy only, no torch).

Everything here is deterministic and OS-independent:
  * directory listings are sorted (os.listdir order differs between Windows
    NTFS and Colab's ext4, which silently changed which images were in each
    split between the two machines the original code ran on);
  * the source train/val split is computed ONCE and reused for both the
    training loader and the per-domain validation loaders.

The original notebook built the training split with one RNG stream across
domains but rebuilt each validation set with a fresh RNG, so ~80% of the
"validation" images of the 2nd and 3rd source domains were training images
(see docs/AUDIT.md, issue A1). `legacy_val_split` reproduces that behaviour
for the audit/test only.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

PACS_DOMAINS: Tuple[str, ...] = ("art_painting", "cartoon", "photo", "sketch")
# VLCS as laid out by DomainBed's download script (folder per domain, then per class)
VLCS_DOMAINS: Tuple[str, ...] = ("Caltech101", "LabelMe", "SUN09", "VOC2007")
DATASETS: Dict[str, Tuple[str, ...]] = {"pacs": PACS_DOMAINS, "vlcs": VLCS_DOMAINS}
ALL_DOMAINS: Tuple[str, ...] = PACS_DOMAINS + VLCS_DOMAINS
IMG_EXTS = (".jpg", ".jpeg", ".png")

Sample = Tuple[str, int, int]  # (path, class_idx, domain_idx within its dataset's domain tuple)


def domains_of(domain: str) -> Tuple[str, ...]:
    """The domain tuple of the dataset that `domain` belongs to."""
    for doms in DATASETS.values():
        if domain in doms:
            return doms
    raise ValueError(f"Unknown domain {domain!r}; expected one of {ALL_DOMAINS}")


def detect_domains(root: str) -> Tuple[str, ...]:
    """Which dataset `root` holds, from its domain folders. PACS is checked first, so a
    PACS root behaves exactly as before VLCS support was added."""
    for doms in DATASETS.values():
        if all(os.path.isdir(os.path.join(root, d)) for d in doms):
            return doms
    raise FileNotFoundError(f"{root} holds neither the PACS domains {PACS_DOMAINS} nor the VLCS domains "
                            f"{VLCS_DOMAINS}")


def list_classes(root: str, domains: Sequence[str] = PACS_DOMAINS) -> List[str]:
    """Union of class folders over all domains, sorted. Fails loudly if domains disagree."""
    per_domain = []
    for d in domains:
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Domain folder not found: {p}")
        per_domain.append(sorted(c for c in os.listdir(p) if os.path.isdir(os.path.join(p, c))))
    if any(cls != per_domain[0] for cls in per_domain):
        raise ValueError(f"Class folders differ across domains: {dict(zip(domains, per_domain))}")
    return per_domain[0]


def list_samples(root: str, domain: str, classes: Sequence[str]) -> List[Sample]:
    d_idx = domains_of(domain).index(domain)
    out: List[Sample] = []
    for c_idx, cls in enumerate(classes):
        folder = os.path.join(root, domain, cls)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(IMG_EXTS):
                out.append((os.path.join(folder, f), c_idx, d_idx))
    return out


def index_pacs(root: str, allow_dct: bool = False) -> Tuple[List[str], Dict[str, List[Sample]]]:
    """Index a PACS root (or a VLCS root; the name is kept for compatibility)."""
    domains = detect_domains(root)
    classes = list_classes(root, domains)
    by_dom = {d: list_samples(root, d, classes) for d in domains}
    if not allow_dct:
        for d, samples in by_dom.items():
            n_dct = sum(os.path.basename(s[0]).lower().startswith("dct_") for s in samples)
            if n_dct:
                raise ValueError(f"{root}: {n_dct}/{len(samples)} '{d}' files are dct_* frequency-domain images, "
                                 "not RGB PACS (docs/AUDIT.md, issue A0). Point data_root at the raw copy, "
                                 "e.g. PACS/pacs_data/pacs_data.")
    return classes, by_dom


def domain_seed(seed: int, domain: str) -> int:
    """Independent, reproducible RNG seed per (run seed, domain)."""
    return seed * 1000 + domains_of(domain).index(domain)


def split_train_val(samples_by_domain: Dict[str, List[Sample]], source_domains: Sequence[str],
                    val_fraction: float = 0.2, seed: int = 0
                    ) -> Tuple[List[Sample], Dict[str, List[Sample]]]:
    """FIXED split: train list + {domain: val list}, disjoint by construction."""
    train: List[Sample] = []
    val: Dict[str, List[Sample]] = {}
    for d in source_domains:
        s = samples_by_domain[d]
        perm = np.random.RandomState(domain_seed(seed, d)).permutation(len(s))
        cut = int(len(s) * (1 - val_fraction))
        train += [s[i] for i in perm[:cut]]
        val[d] = [s[i] for i in perm[cut:]]
    return train, val


def legacy_val_split(sizes: Dict[str, int], source_domains: Sequence[str],
                     val_fraction: float = 0.2, seed: int = 0) -> Dict[str, float]:
    """Reproduces the ORIGINAL notebook logic on index ranges and returns, per source
    domain, the fraction of its 'validation' indices that were also training indices."""
    rng = np.random.RandomState(seed)
    overlap = {}
    for d in source_domains:
        n = sizes[d]
        perm = rng.permutation(n)                       # split_domainwise: shared stream
        train = set(perm[:int(n * (1 - val_fraction))].tolist())
        perm_v = np.random.RandomState(seed).permutation(n)  # make_loaders: fresh RNG
        val = perm_v[int(n * (1 - val_fraction)):]
        overlap[d] = float(np.mean([v in train for v in val]))
    return overlap


def sample_ids_hash(samples: Sequence[Sample]) -> str:
    """Hash of relative paths so features from different conditions can be checked
    to be row-aligned before any paired test."""
    h = hashlib.sha1()
    for p, c, d in samples:
        rel = "/".join(p.replace("\\", "/").split("/")[-3:])
        h.update(f"{rel}|{c}|{d}\n".encode())
    return h.hexdigest()[:16]


def probe_split(domain_labels: np.ndarray, test_size: float = 0.3, seed: int = 42
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified (by domain) train/test split of SOURCE samples for the domain probe."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(domain_labels))
    tr, te = train_test_split(idx, test_size=test_size, stratify=domain_labels, random_state=seed)
    return np.sort(tr), np.sort(te)
