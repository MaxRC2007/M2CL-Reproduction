"""Gradient share of the contrastive term in the authors' M2-CL (ResNet-18) and in ours, on the same batches.

  python -I authors_grad_share.py <authors repo>/domainbed <our repo> <RGB PACS root> <per-domain batch> [n_batches]

The authors' model and loss files are imported by path (they import only torch). Their ImageNet
download is redirected to the local torchvision cache. Batches follow DomainBed: the same number of
images from each source domain (Sketch held out), DomainBed's training augmentation, train mode.
Their update() is replicated exactly, including how it builds the same-class pair lists.
"""
import importlib.util
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

AUTH, OURS, DATA, PER_DOMAIN = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
N_BATCHES = int(sys.argv[5]) if len(sys.argv) > 5 else 2
CACHE = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
torch.manual_seed(0)
np.random.seed(0)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m2cl18 = load("auth_m2cl_18", f"{AUTH}/mymodels/m2cl_18.py")
myloss = load("auth_myloss", f"{AUTH}/lib/myloss.py")
m2cl18.load_state_dict_from_url = lambda url, progress=True: torch.load(CACHE, map_location="cpu", weights_only=True)
sys.path.insert(0, OURS + "/src")
from m2cl.losses import m2cl_contrastive_loss
from m2cl.models import build_model

# DomainBed's training augmentation for PACS
tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), transforms.RandomGrayscale(),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
domains = [datasets.ImageFolder(os.path.join(DATA, d), tf) for d in ("art_painting", "cartoon", "photo")]
assert all(d.classes == domains[0].classes for d in domains)


def batch():
    xs, ys = [], []
    for d in domains:  # with replacement, as DomainBed's InfiniteDataLoader
        for i in np.random.randint(0, len(d), PER_DOMAIN):
            x, y = d[i]
            xs.append(x)
            ys.append(y)
    return torch.stack(xs), torch.tensor(ys)


def same_indexes_like_authors(all_y):
    """Copied from domainbed/algorithms.py, M2CL.update (commit 0b1bce3)."""
    y_np = all_y.numpy()
    y = y_np.tolist()
    tmp = {}
    for i in y:
        tmp[i] = np.where(y_np == i)
    tmp = OrderedDict(sorted(tmp.items()))
    out = []
    for i in range(len(tmp.items())):
        if i in tmp.keys():
            out.append(torch.combinations(torch.tensor(tmp[i][0])))
    return out


def gnorm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(sum((x.float() ** 2).sum() for x in g if x is not None) ** 0.5)


auth = m2cl18.M2CL18(7, pretrained=True).train()
a_bb = [p for n, p in auth.named_parameters() if n.startswith(("conv", "bn", "down"))]
a_eb = [p for n, p in auth.named_parameters() if n.startswith("ext_")]
ours = build_model("m2cl", 7, 6, True).train()
o_bb = [p for n, p in ours.named_parameters() if n.startswith("layer")]
o_eb = [p for n, p in ours.named_parameters() if n.startswith("eb")]
print(f"authors: {len(a_bb)} backbone tensors, {len(a_eb)} extraction tensors | ours: {len(o_bb)} and {len(o_eb)}", flush=True)

for b in range(N_BATCHES):
    x, y = batch()
    t = time.time()
    preds, acts = auth(x)
    assert len(acts) == 21, len(acts)
    same = same_indexes_like_authors(y)
    cl = myloss.my_loss(acts, same, 0.01, 1.0)  # alpha 0.01, tau 1.0: their defaults
    ce = F.cross_entropy(preds, y)
    r_bb = gnorm(cl, a_bb) / gnorm(ce, a_bb)
    r_eb = gnorm(cl, a_eb) / gnorm(ce, a_eb)
    print(f"\nbatch {b}: {len(y)} images, {len(y.unique())} classes, {len(same)} classes used by their loss "
          f"({time.time() - t:.0f}s)")
    print(f"  AUTHORS alpha 0.01 tau 1: CE {ce.item():.3f}, contrastive term {cl.item():.3f} "
          f"(per pipeline {cl.item() / 0.01 / 21:.2f}) | grad ratio backbone {r_bb:.4f}, extraction {r_eb:.4f}", flush=True)
    del preds, acts, cl, ce

    t = time.time()
    logits, reps = ours(x)
    cl_o = m2cl_contrastive_loss(reps, y, 1.0, "sum")
    ce_o = F.cross_entropy(logits, y)
    gb_cl, gb_ce = gnorm(cl_o, o_bb), gnorm(ce_o, o_bb)
    ge_cl, ge_ce = gnorm(cl_o, o_eb), gnorm(ce_o, o_eb)
    print(f"  OURS tau 1: CE {ce_o.item():.3f}, summed loss {cl_o.item():.2f} ({time.time() - t:.0f}s) | grad ratio backbone "
          f"alpha 0.003 {0.003 * gb_cl / gb_ce:.4f}, alpha 0.01 {0.01 * gb_cl / gb_ce:.4f}; extraction alpha 0.003 "
          f"{0.003 * ge_cl / ge_ce:.4f}", flush=True)
    del logits, reps, cl_o, ce_o
