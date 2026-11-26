from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import csv
import json
import textwrap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image

MAT = str(ROOT / "data") + "/"
OUT = str(ROOT / "paper/figures") + "/"
BLUE, ORANGE, AQUA = ("#2a78d6", "#eb6834", "#1baf7a")
GRAY, LIGHT = ("#8a8985", "#c9c8c3")
INK, INK2, GRID = ("#0b0b0b", "#52514e", "#e6e5e1")
PANEL = "#f4f3ef"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.edgecolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    }
)


def text_width_in(fig, s, size, family="DejaVu Sans", weight="normal"):
    r = fig.canvas.get_renderer()
    w, _, _ = r.get_text_width_height_descent(
        s, FontProperties(family=family, size=size, weight=weight), ismath=False
    )
    return w / fig.dpi


def wrap_fit(fig, text, width_in, size, family="DejaVu Sans", weight="normal"):
    """Greedy word wrap so that every line renders narrower than width_in inches."""
    out = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            trial = (line + " " + word).strip()
            if line and text_width_in(fig, trial, size, family, weight) > width_in:
                out.append(line)
                line = word
            else:
                line = trial
        out.append(line)
    return "\n".join(out)


def epact_egg():
    crops = {
        c["name"]: c
        for c in json.load(open(MAT + "epact-figures/crop-coordinates.json"))
    }
    cases_json = json.load(open(MAT + "epact-figures/image-cases.json"))["cases"]

    def answers(task):
        c = cases_json[task]
        return (
            c["reason_first"]["pred"],
            c["class_note"]["pred"],
            c["reason_first"]["answer"],
        )

    rows = [
        ("egg-crop", "val00000-C+D-CHECK-0", "Egg present (ground truth: yes)"),
        (
            "decoy-crop",
            "val00064-C-CHECK-0-twin",
            "Queried egg removed; a non-egg\ndistractor remains (ground truth: no)",
        ),
    ]
    fig = plt.figure(figsize=(5.9, 4.0))
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.15, 0.9, 1.85],
        height_ratios=[1, 1, 0.72],
        wspace=0.08,
        hspace=0.2,
    )
    for r, (cname, task, title) in enumerate(rows):
        c = crops[cname]
        if not c["source"] == task + ".png":
            raise ValueError("Numerical validation failed at source line 77")
        rf, cn, gold = answers(task)
        im = np.asarray(Image.open(MAT + "epact-figures/" + c["source"]).convert("RGB"))
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(im, interpolation="none")
        ax.add_patch(
            Rectangle(
                (c["x"], c["y"]),
                c["width"],
                c["height"],
                fill=False,
                ec="white",
                lw=1.6,
            )
        )
        ax.add_patch(
            Rectangle(
                (c["x"], c["y"]), c["width"], c["height"], fill=False, ec=INK, lw=0.7
            )
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title("Witness image", loc="left", fontsize=7.5)
        ax2 = fig.add_subplot(gs[r, 1])
        ax2.imshow(
            im[c["y"] : c["y"] + c["height"], c["x"] : c["x"] + c["width"]],
            interpolation="none",
        )
        ax2.set_xticks([])
        ax2.set_yticks([])
        if r == 0:
            ax2.set_title("Boxed crop, enlarged", loc="left", fontsize=7.5)
        ax3 = fig.add_subplot(gs[r, 2])
        ax3.axis("off")
        ax3.text(
            0.02,
            0.95,
            "Is there an egg in the fridge?",
            fontsize=7.4,
            color=INK2,
            va="top",
            style="italic",
        )
        ax3.text(
            0.02,
            0.77,
            title,
            fontsize=7.6,
            color=INK,
            va="top",
            weight="bold",
            linespacing=1.2,
        )
        y = 0.4 if "\n" in title else 0.5
        for label, ans in (("Reason-first", rf), ("With class note", cn)):
            ok = ans == gold
            ax3.text(0.02, y, f"{label}:", fontsize=7.6, color=INK2, va="top")
            ax3.text(
                0.62,
                y,
                f"{ans}  ({('correct' if ok else 'wrong')})",
                fontsize=7.6,
                color=AQUA if ok else ORANGE,
                va="top",
                weight="bold",
            )
            y -= 0.17
    axn = fig.add_subplot(gs[2, :])
    axn.axis("off")
    note = open(MAT + "epact-figures/class-note.txt").read().strip()
    if not note.startswith("Note about this environment: eggs are rendered"):
        raise ValueError("Numerical validation failed at source line 101")
    axn.text(
        0.012,
        0.93,
        "Class note added to the question (exact text):",
        fontsize=7.0,
        color=INK2,
        va="top",
        transform=axn.transAxes,
    )
    width_in = axn.get_position().width * fig.get_figwidth() + 0.35
    head, tail = note.split("ovoids; ", 1)
    tn = axn.text(
        0.012,
        0.76,
        "“" + head + "ovoids;\n" + tail + "”",
        fontsize=7.0,
        color=INK,
        va="top",
        transform=axn.transAxes,
        linespacing=1.25,
    )
    fig.canvas.draw()
    bb = tn.get_window_extent(fig.canvas.get_renderer()).transformed(
        axn.transAxes.inverted()
    )
    axn.add_patch(
        FancyBboxPatch(
            (0.0, bb.y0 - 0.04),
            max(1.0, bb.x1 + 0.012),
            0.99 - (bb.y0 - 0.04),
            boxstyle="round,pad=0.01,rounding_size=0.02",
            fc=PANEL,
            ec=GRID,
            lw=0.6,
            transform=axn.transAxes,
            zorder=0,
        )
    )
    from collections import defaultdict

    mem = json.load(open(MAT + "epact-figures/class-note-member-results.json"))

    def solved(rows, key):
        g = defaultdict(list)
        for m in rows:
            g[m["pair_id"]].append(m[key] == m["answer"])
        return (sum((all(v) for v in g.values())), len(g))

    egg = [m for m in mem if m["egg"]]
    other = [m for m in mem if not m["egg"]]
    (e0, en), (e1, _) = (solved(egg, "reasoned_pred"), solved(egg, "pred"))
    (o0, on), (o1, _) = (solved(other, "reasoned_pred"), solved(other, "pred"))
    if not (e0, e1, en, o0, o1, on) == (5, 10, 12, 6, 4, 8):
        raise ValueError("Numerical validation failed at source line 123")
    summary = f"On the {en + on} recorded pairs where the note was tested: egg pairs {e0}/{en} → {e1}/{en}; other pairs {o0}/{on} → {o1}/{on} (two new tomato false positives, three answers become unparseable, one tomato false negative corrected)."
    axn.text(
        0.0,
        bb.y0 - 0.13,
        wrap_fit(fig, summary, width_in, 6.9),
        fontsize=6.9,
        color=INK2,
        va="top",
        transform=axn.transAxes,
        linespacing=1.25,
    )
    fig.savefig(OUT + "epact_fig_egg.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def read_actions(name):
    rows = list(csv.DictReader(open(MAT + f"post-figures/{name}-actions.csv")))
    out = []
    for r in rows:
        a = json.loads(r["action"])
        kind = list(a.keys())[0]
        out.append(
            dict(
                kind=kind,
                ok=r["ok"] == "True",
                change=bool(r["new_feasibility"].strip()),
            )
        )
    return out


def post_reach():
    objs = list(csv.DictReader(open(MAT + "post-figures/object-positions.csv")))
    pair = json.load(open(MAT + "post-figures/scene-and-pair.json"))["pair"]
    dock = pair["dock"]["position"]
    dx, dz = (dock["x"], dock["z"])
    reach = {m["member"]: m["theta"]["rho"] for m in pair["members"]}
    label = {m["member"]: m["label_choice"] for m in pair["members"]}
    if not (
        reach == {"A": 0.9, "B": 2.6} and label == {"A": "butterknife", "B": "apple"}
    ):
        raise ValueError("Numerical validation failed at source line 145")
    cand = {c["name"]: c for c in pair["candidates"]}
    fig = plt.figure(figsize=(5.6, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.22], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect("equal")
    seen = set()
    for o in objs:
        if o["object_id"] in seen or o["candidate"] == "True":
            continue
        seen.add(o["object_id"])
        ax.plot(
            float(o["x"]) - dx, float(o["z"]) - dz, "o", ms=2.2, color=LIGHT, zorder=1
        )
    ax.add_patch(Circle((0, 0), reach["A"], fill=False, ec=INK, lw=1.0, zorder=2))
    ax.add_patch(
        Circle((0, 0), reach["B"], fill=False, ec=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
    )
    ax.plot(0, 0, "s", ms=5.0, color=INK2, zorder=4)
    ax.text(0, -0.2, "dock", ha="center", va="top", fontsize=6.6, color=INK2)
    pos = {
        o["name"]: (float(o["x"]) - dx, float(o["z"]) - dz)
        for o in objs
        if o["candidate"] == "True"
    }
    for nm in ("fork", "butterknife", "apple"):
        ax.plot(*pos[nm], "o", ms=4.0, color=INK, zorder=5)
    lab = {
        "butterknife": (
            (1.55, 0.62),
            f"butterknife\n{cand['butterknife']['mass']:.2f} kg",
        ),
        "fork": ((1.55, -0.72), f"fork\n{cand['fork']['mass']:.2f} kg"),
        "apple": ((-1.35, -1.05), f"apple\n{cand['apple']['mass']:.2f} kg"),
    }
    for nm, ((tx, tz), txt) in lab.items():
        ax.annotate(
            txt,
            xy=pos[nm],
            xytext=(tx, tz),
            fontsize=6.6,
            color=INK,
            ha="center",
            va="center",
            linespacing=1.15,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2, shrinkA=1, shrinkB=2.5),
        )
    ax.set_xlim(-3.75, 2.75)
    ax.set_ylim(-2.75, 2.75)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [
        Line2D([0], [0], color=INK, lw=1.0),
        Line2D([0], [0], color=INK, lw=1.0, ls=(0, (4, 3))),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=LIGHT,
            markeredgecolor=LIGHT,
            ms=3,
        ),
    ]
    ax.legend(
        handles,
        [
            f"A: reach {reach['A']} m; correct: butterknife",
            f"B: reach {reach['B']} m; correct: apple",
            "other catalogued objects",
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 0.0),
        fontsize=6.4,
        frameon=False,
        handlelength=2.0,
        borderaxespad=0.2,
        labelspacing=0.35,
    )
    col = {"rotate": LIGHT, "move": ORANGE, "try_lift": BLUE, "put_down": AQUA}
    rows = [
        (
            "checkup_reasoned_k40_B",
            "Body B, self-run checkup, 40 turns",
            "0 feasibility changes; no valid final choice",
        ),
        (
            "checkup_watch_reasoned_B",
            "Body B, watched script, 15 actions",
            "3 feasibility changes; chooses apple (correct)",
        ),
        (
            "checkup_watch_reasoned_A",
            "Body A, watched script, 14 actions",
            "3 feasibility changes; still chooses apple (wrong)",
        ),
    ]
    sub = gs[0, 1].subgridspec(4, 1, height_ratios=[1.55, 0.95, 0.95, 0.6], hspace=0.62)
    for k, (name, title, outcome) in enumerate(rows):
        axr = fig.add_subplot(sub[k])
        axr.axis("off")
        acts = read_actions(name)
        nrows = (len(acts) - 1) // 20 + 1
        axr.set_xlim(-0.05, 20.5)
        axr.set_ylim(-nrows * 1.05 - 0.1, 0.55)
        axr.set_title(title, loc="left", fontsize=7.0, weight="bold", pad=2)
        for i, a in enumerate(acts):
            rr, cc = divmod(i, 20)
            x0, y0 = (cc * 1.02, -rr * 1.05)
            axr.add_patch(
                Rectangle(
                    (x0, y0 - 0.8), 0.9, 0.8, fc=col[a["kind"]], ec="white", lw=0.4
                )
            )
            if not a["ok"]:
                axr.text(
                    x0 + 0.45,
                    y0 - 0.42,
                    "×",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=INK,
                )
            if a["change"]:
                axr.plot(
                    x0 + 0.45, y0 + 0.22, marker="v", ms=3.6, color=INK, clip_on=False
                )
        axr.text(0, -nrows * 1.05 - 0.05, outcome, fontsize=6.6, color=INK2, va="top")
    axl = fig.add_subplot(sub[3])
    axl.axis("off")
    handles = [
        Rectangle((0, 0), 1, 1, fc=col[k], ec="white")
        for k in ("rotate", "move", "try_lift", "put_down")
    ]
    handles += [
        Line2D(
            [0],
            [0],
            marker="v",
            color="none",
            markerfacecolor=INK,
            markeredgecolor=INK,
            ms=4,
        ),
        Line2D(
            [0],
            [0],
            marker="$×$",
            color="none",
            markerfacecolor=INK,
            markeredgecolor=INK,
            ms=5,
        ),
    ]
    axl.legend(
        handles,
        [
            "rotate",
            "move",
            "try lift",
            "put down",
            "feasibility change",
            "failed action",
        ],
        loc="center left",
        ncol=3,
        fontsize=6.3,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.9,
    )
    fig.canvas.draw()
    top = (
        max((a.get_tightbbox(fig.canvas.get_renderer()).y1 for a in fig.axes))
        / fig.dpi
        / fig.get_figheight()
    )
    fig.text(
        ax.get_position().x0,
        top + 0.03,
        "(a) Plan view, two reach limits",
        fontsize=8,
        va="bottom",
    )
    fig.text(
        gs[0, 1].get_position(fig).x0,
        top + 0.03,
        "(b) Checkups for this pair",
        fontsize=8,
        va="bottom",
    )
    fig.savefig(
        str(ROOT / "post/paper/figures/post_fig_reach.pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


if __name__ == "__main__":
    epact_egg()
    post_reach()
