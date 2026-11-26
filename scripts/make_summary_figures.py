from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import math
import json
from research_common.results import recompute

recorded = recompute()
teaser = json.loads(
    (ROOT / "data/epact-figures/teaser.json").read_text(encoding="utf-8")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)
OUT = str(ROOT / "paper/figures") + "/"


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ---------- EPACT teaser ----------
fig, ax = plt.subplots(figsize=(5.5, 2.25))
conds = [
    "Start view\n(direct QA)",
    "Supplied evidence\n(direct QA)",
    "Handoff after\nreplayed acquisition",
    "Autonomous\ninteraction",
    "Autonomous after\ntrajectory SFT (8B)",
]
models = [
    (r["name"], c, r["pairs"]) for r, c in zip(teaser["models"], (BLUE, ORANGE, AQUA))
]
w = 0.26
x = np.arange(4)
for i, (name, col, vals) in enumerate(models):
    xs = x + (i - 1) * w
    ax.bar(xs, vals, w, color=col, edgecolor="white", lw=0.8, label=name)
    for xx, v in zip(xs, vals):
        ax.text(xx, v + 1.2, str(v), ha="center", va="bottom", fontsize=6.3, color=INK2)
seeds = teaser["sft_seeds"]
m = np.mean(seeds)
ax.bar(4, m, w, color=ORANGE, edgecolor="white", lw=0.8)
ax.scatter([4 - 0.05, 4, 4 + 0.05], seeds, s=10, color=INK, zorder=3, lw=0)
ax.text(
    4.0,
    max(seeds) + 3.0,
    "seeds: 34, 39, 42",
    ha="center",
    va="bottom",
    fontsize=6.3,
    color=INK2,
)
ax.axvline(3.55, color=GRID, lw=0.8, ls="--")
ax.set_xticks(list(x) + [4])
ax.set_xticklabels(conds)
ax.set_ylim(0, 76)
ax.set_yticks([0, 18, 36, 54, 72])
ax.set_ylabel("Correct pairs (of 72)")
ygrid(ax)
ax.legend(
    loc="upper right",
    ncol=3,
    bbox_to_anchor=(1.0, 1.16),
    handlelength=1.0,
    columnspacing=1.0,
)
fig.tight_layout()
fig.savefig(OUT + "epact_fig_teaser.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---------- POST results ----------
fig, (a1, a2) = plt.subplots(
    1, 2, figsize=(5.5, 2.0), gridspec_kw={"width_ratios": [5, 3]}
)
labs = [
    "Stated\nimmediate",
    "Stated\nreasoned",
    "Own check\n12 turns",
    "Own check\n40 turns",
    "Watched\ncheckup",
]
vals = [
    recorded["post"][k]["pair_correct"]
    for k in (
        "told_profile",
        "told_profile_reasoned",
        "checkup_reasoned",
        "checkup_reasoned_k40",
        "checkup_watch_reasoned",
    )
]
cols = [BLUE, BLUE, ORANGE, ORANGE, AQUA]
a1.bar(range(5), vals, 0.62, color=cols, edgecolor="white", lw=0.8)
for i, v in enumerate(vals):
    a1.text(i, v + 0.25, f"{v}/12", ha="center", va="bottom", fontsize=6.5, color=INK2)
a1.set_xticks(range(5))
a1.set_xticklabels(labs, fontsize=6.3)
a1.set_ylim(0, 12)
a1.set_yticks([0, 3, 6, 9, 12])
a1.set_ylabel("Correct pairs (of 12)")
a1.set_title("(a) Paired choice accuracy", loc="left")
ygrid(a1)
labs2 = ["Own check\n12 turns", "Own check\n40 turns", "Watched\ncheckup"]
vals2 = [
    recorded["post"][k]["sufficient_evidence_episodes"]
    for k in ("checkup_reasoned", "checkup_reasoned_k40", "checkup_watch_reasoned")
]
a2.bar(range(3), vals2, 0.62, color=[ORANGE, ORANGE, AQUA], edgecolor="white", lw=0.8)
for i, v in enumerate(vals2):
    a2.text(i, v + 0.5, f"{v}/24", ha="center", va="bottom", fontsize=6.5, color=INK2)
a2.set_xticks(range(3))
a2.set_xticklabels(labs2)
a2.set_ylim(0, 24)
a2.set_yticks([0, 6, 12, 18, 24])
a2.set_ylabel("Episodes with sufficient\nbody evidence (of 24)")
a2.set_title("(b) Evidence yield", loc="left")
ygrid(a2)
from matplotlib.patches import Patch

fig.legend(
    handles=[
        Patch(color=BLUE, label="Body profile stated"),
        Patch(color=ORANGE, label="Planner runs its own checkup"),
        Patch(color=AQUA, label="Scripted checkup, watched"),
    ],
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.07),
    handlelength=1.0,
)
fig.tight_layout()
fig.savefig(
    str(ROOT / "post/paper/figures/post_fig_results.pdf"),
    bbox_inches="tight",
    pad_inches=0.02,
)
plt.close(fig)
