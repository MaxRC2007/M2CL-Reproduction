"""Figure and equivalence test for the 12 seed-matched 6-tap M2 / M2-CL pairs.

  python scripts/make_fig_pairs.py <m2 runs zip> <m2cl runs zip> <output.pdf>

Plots, per held-out fold, M2-CL minus its paired M2 run next to each M2 seed's
deviation from its fold mean, so the effect of the loss is shown on the scale of
seed variation. Also prints a paired t interval and a TOST equivalence test.
"""
import json
import re
import statistics as st
import sys
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

FOLDS = ("art_painting", "cartoon", "photo", "sketch")
NAMES = ("Art", "Cartoon", "Photo", "Sketch")


def results(path, prefix):
    z = zipfile.ZipFile(path)
    out = {}
    for i in z.infolist():
        m = re.search(r"runs/" + prefix + r"_6tap_(\w+?)_seed(\d)/result\.json$", i.filename.replace("\\", "/"))
        if m:
            out[(m[1], int(m[2]))] = json.loads(z.read(i))["test_acc"]
    return out


m2 = results(sys.argv[1], "m2")
cl = results(sys.argv[2], "m2cl")
assert len(m2) == len(cl) == 12

diff = {f: [cl[(f, s)] - m2[(f, s)] for s in range(3)] for f in FOLDS}
seed_dev = {f: [m2[(f, s)] - st.mean(m2[(f, k)] for k in range(3)) for s in range(3)] for f in FOLDS}

d = [x for f in FOLDS for x in diff[f]]
n, mean, sd = len(d), st.mean(d), st.stdev(d)
se = sd / n ** 0.5
for level in (0.90, 0.95):
    t = stats.t.ppf(1 - (1 - level) / 2, n - 1)
    print(f"{int(level * 100)}% CI of mean paired difference: {mean - t * se:+.2f} to {mean + t * se:+.2f} pp")
for margin in (0.5, 0.4):
    p = max(stats.t.sf((mean + margin) / se, n - 1), stats.t.cdf((mean - margin) / se, n - 1))
    print(f"TOST equivalence within +/-{margin} pp: p = {p:.3f}")
print(f"mean {mean:+.3f}, SD {sd:.3f}, n {n}")
print("largest |M2 seed deviation| per fold:", {f: round(max(map(abs, seed_dev[f])), 2) for f in FOLDS})

INK, MUTED, BLUE, GRID = "#0b0b0b", "#52514e", "#2a78d6", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(3.45, 2.2))
ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
ax.yaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
for i, f in enumerate(FOLDS):
    jitter = [-0.05, 0.0, 0.05]  # seeds 0, 1, 2 side by side, so equal values stay visible
    ax.scatter([i - 0.15 + j for j in jitter], seed_dev[f], s=22, facecolors="none", edgecolors=MUTED, lw=1.0, zorder=3,
               label="M2 seed minus fold mean" if i == 0 else None)
    ax.scatter([i + 0.15 + j for j in jitter], diff[f], s=22, color=BLUE, edgecolors="white", lw=0.6, zorder=4,
               label="M2-CL minus paired M2" if i == 0 else None)
ax.set_xticks(range(4), NAMES)
ax.set_xlim(-0.5, 3.5)
ax.set_ylabel("Test accuracy difference (pp)")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, handletextpad=0.3, columnspacing=1.2)
fig.tight_layout()
fig.savefig(sys.argv[3], bbox_inches="tight")
fig.savefig(sys.argv[3].replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("saved", sys.argv[3])
