# -*- coding: utf-8 -*-
"""Fig. 3 - the leakage ladder.

Table VII carries the same numbers, but a table makes the reader do the
subtraction. The point of the decomposition is a shape: three rungs that barely
move and one that drops. The figure states it in one glance, and it states the
honest part too - every rung sits close to the trivial floor throughout, so
nobody can read the drop as a fall from a good score.

Built to the same rules as the other two figures: 8 cm wide, serif type, no
colour carrying meaning on its own.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_paths                                          # noqa: E402

import json
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = io_paths.result_in("RESULTS_ladder_v2.json", required=True)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fig_v2_ladder.png")

INK = "#1a1a1a"
MID = "#767676"
LIGHT = "#c9c9c9"
ACCENT = "#1f365c"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.edgecolor": MID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MID,
    "ytick.color": MID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 400,
})

LABELS = ["R0\nno controls", "R1\n+ exact\ndedup.",
          "R2\n+ near-dup.\nremoval", "R3\n+ thread-\ndisjoint"]


def main():
    d = json.load(io.open(RESULTS, encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)

    for ax, key, title in ((axes[0], "strict", "Strict labels"),
                           (axes[1], "broad", "Broad labels")):
        rungs = d[key]["rungs"]
        f1 = [r["f1"] for r in rungs]
        sd = [r["f1_sd"] for r in rungs]
        floor = rungs[-1]["trivial_floor"]
        x = np.arange(len(f1))

        ax.axhline(floor, color=LIGHT, lw=1.0, zorder=1)
        ax.annotate("trivial all-positive floor", xy=(0.02, floor),
                    xytext=(0.02, floor - 0.018), fontsize=6.5, color=MID)

        ax.errorbar(x, f1, yerr=sd, fmt="none", ecolor=LIGHT, elinewidth=1.0,
                    capsize=2.5, zorder=1)
        ax.plot(x, f1, "-", color=MID, lw=1.0, zorder=2)
        ax.plot(x[:-1], f1[:-1], "o", color="white", mec=MID, mew=1.1,
                ms=5.5, zorder=3)
        ax.plot(x[-1:], f1[-1:], "o", color=ACCENT, mec=ACCENT, ms=5.5,
                zorder=3)

        # The last point's label goes below it: the drop arrow occupies the
        # space above, and the two collided at print size.
        for xi, v in zip(x, f1):
            below = xi == x[-1]
            ax.annotate("%.4f" % v, xy=(xi, v),
                        xytext=(0, -12 if below else 7),
                        textcoords="offset points", ha="center", fontsize=6.8,
                        color=INK)

        drop = f1[-1] - f1[-2]
        ax_x = x[-1] + 0.16
        ax.annotate("", xy=(ax_x, f1[-1]), xytext=(ax_x, f1[-2]),
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
        ax.annotate("%+.4f" % drop, xy=(ax_x, (f1[-1] + f1[-2]) / 2.0),
                    xytext=(5, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=7, color=ACCENT)

        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=6.8)
        ax.set_xlim(-0.35, len(f1) + 0.15)
        lo = min(min(f1), floor) - max(0.045, max(sd) + 0.02)
        hi = max(f1) + max(0.055, max(sd) + 0.03)
        ax.set_ylim(lo, hi)
        ax.set_title(title, fontsize=8, color=INK, pad=6)
        ax.tick_params(length=2)

    axes[0].set_ylabel("Pooled out-of-fold F1")
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("written %s" % OUT)
    for k in ("strict", "broad"):
        print("  %-8s %s" % (k, [r["f1"] for r in d[k]["rungs"]]))


if __name__ == "__main__":
    main()
