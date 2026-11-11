"""Figures for the v4 paper, generated only from result files in this repository.

  python scripts/make_paper_figures.py --out paper

Figure 1 (fig_forest.pdf): paired test-accuracy differences, contrastive term on minus off, for the three
  pre-specified/pre-registered tests (PACS ours, PACS authors' code, VLCS ours). Dots: the 12 seed-matched pairs.
  Square and bar: mean and 95% t-interval, as computed by the analysis scripts. Band: the +/-0.71 pp margin.
  Open diamond: the gain the original paper reports for that dataset.
Figure 2 (fig_probe.pdf): domain-probe accuracy, M2-CL minus M2, per layer, PACS and VLCS. Marker: mean over the
  12 pairs. Bar: range over the pairs.
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE, INK, MUTED, BAND = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
MARGIN = 0.71
PAPER_GAIN = {"PACS": 0.71, "VLCS": 1.67}  # Ballas and Diou (2024), ResNet-18, Table I: M2-CL minus M2

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "pdf.fonttype": 42})


def load(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def forest(out):
    r3 = load("results/round3/round3_analysis_all.json")
    r4 = load("results/round4/round4_analysis.json")
    rows = [("PACS, ours\n(round 3)", r3["H1/H2 M2-CL - M2"], "PACS", "inconclusive"),
            ("PACS, authors'\ncode (round 4)", r4["H1/H2 lparam 0.01 - 0"], "PACS", "inconclusive"),
            ("VLCS, ours\n(round 3)", r3["S4 VLCS M2-CL - M2"], "VLCS", "equivalent")]
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    ax.axvspan(-MARGIN, MARGIN, color=BAND, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    for i, (label, res, ds, verdict) in enumerate(rows):
        y = len(rows) - 1 - i
        d = list(res["diffs"].values())
        offs = [(-0.16 + 0.32 * k / (len(d) - 1)) for k in range(len(d))]
        ax.scatter(d, [y + 0.22 + o * 0.25 for o in offs], s=9, color=MUTED, alpha=0.75, lw=0, zorder=2)
        lo, hi = res["ci95"]
        ax.plot([lo, hi], [y, y], color=INK, lw=1.4, solid_capstyle="butt", zorder=3)
        ax.plot([lo, lo], [y - 0.07, y + 0.07], color=INK, lw=1.0, zorder=3)
        ax.plot([hi, hi], [y - 0.07, y + 0.07], color=INK, lw=1.0, zorder=3)
        ax.scatter([res["mean"]], [y], marker="s", s=22, color=INK, zorder=4)
        ax.scatter([PAPER_GAIN[ds]], [y], marker="D", s=26, facecolor="white", edgecolor=ORANGE, lw=1.3, zorder=5)
        ax.text(1.03, y, f"{res['mean']:+.2f}\n[{lo:+.2f}, {hi:+.2f}]\n{verdict}", va="center", ha="left",
                fontsize=6.3, color=INK, linespacing=1.15, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7, color=INK)
    ax.set_xlim(-2.3, 3.9)
    ax.set_ylim(-0.55, len(rows) - 0.25)
    ax.set_xlabel("Test-accuracy difference, term on minus off (pp)")
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    h = [plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=3, label="seed-matched pair"),
         plt.Line2D([], [], marker="s", color=INK, ms=4, lw=1.4, label="mean, 95% CI"),
         plt.Line2D([], [], marker="D", ls="", mfc="white", mec=ORANGE, ms=4.5, label="published gain"),
         plt.Rectangle((0, 0), 1, 1, color=BAND, label=r"$\pm$0.71 pp margin")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6.3,
              handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(pad=0.2)
    fig.subplots_adjust(right=0.80)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_forest.{ext}"), dpi=300)
    plt.close(fig)


def probe_rows(path):
    with open(os.path.join(ROOT, path)) as f:
        return list(csv.DictReader(f))


def probe(out):
    layers = ["eb1a", "eb1", "eb2a", "eb2", "eb3", "eb4", "eb_concat", "backbone"]
    names = ["eb1a", "eb1", "eb2a", "eb2", "eb3", "eb4", "all\nblocks", "back-\nbone"]
    sets = [("PACS", "results/tables/round3_taps/leak_r3_cl_effect_at_taps.csv", BLUE, "o", -0.14),
            ("VLCS", "results/tables/round3_vlcs_taps/leak_r3v_cl_effect_at_taps.csv", ORANGE, "s", 0.14)]
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    for name, path, col, mk, dx in sets:
        rows = probe_rows(path)
        for i, layer in enumerate(layers):
            d = [float(r["Delta"]) for r in rows if r["layer"] == layer]
            assert len(d) == 12, (name, layer, len(d))
            m = sum(d) / len(d)
            ax.plot([i + dx, i + dx], [min(d), max(d)], color=col, lw=1.2, zorder=2)
            ax.scatter([i + dx], [m], marker=mk, s=18, color=col, edgecolor="white", lw=0.6, zorder=3,
                       label=name if i == 0 else None)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(names, fontsize=7, color=INK)
    ax.set_ylabel("Domain-probe accuracy,\nM2-CL minus M2 (pp)")
    ax.set_ylim(-18.5, 2.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=7, handletextpad=0.2)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_probe.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    forest(a.out)
    probe(a.out)
    print("wrote", os.path.join(a.out, "fig_forest.pdf"), "and fig_probe.pdf")
