# -*- coding: utf-8 -*-
"""Fig. 4: what each model is worth when it is allowed to decline.

Built straight from results/RESULTS_SELECTIVE.json, so the figure cannot drift
from the numbers. Two things the table cannot show as well:

  * the random-abstention band, drawn as a shaded ribbon, so the reader sees
    at a glance that the curves leave it rather than reading z-scores;
  * the annotator reference at 0.744, so the climb is read against what a
    second human reaches rather than against zero.

Coverage on the x axis is the coverage ACTUALLY reached on test, not the
target - the cut-off is an absolute confidence value chosen on validation, so
the two differ.
"""
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import io_paths                                              # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "fig_v2_selective.png")
HUMAN_F1 = 0.744

# One hue per model, assigned in a fixed order and never cycled. The encoder
# is the darkest because it is the one the text singles out.
COLOURS = {"logistic regression": "#8C9196",
           "character n-grams": "#5C6AC4",
           "linear SVM": "#007A5C",
           "fine-tuned encoder": "#1F2430"}
ORDER = ["logistic regression", "character n-grams", "linear SVM",
         "fine-tuned encoder"]


def main():
    R = json.load(io.open(io_paths.result_in("RESULTS_SELECTIVE.json",
                                             required=True), encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    # the random band, from the model with the widest spread at each coverage
    ref = R["fine-tuned encoder"]["rows"]
    xs = [r["actual_coverage"] for r in ref]
    lo = [r["random_mean"] - 2 * r["random_sd"] for r in ref]
    hi = [r["random_mean"] + 2 * r["random_sd"] for r in ref]
    o = np.argsort(xs)
    ax.fill_between(np.array(xs)[o], np.array(lo)[o], np.array(hi)[o],
                    color="#C9CCD1", alpha=0.55, linewidth=0, zorder=1,
                    label="random abstention, ±2 sd")

    for name in ORDER:
        rows = R[name]["rows"]
        x = np.array([r["actual_coverage"] for r in rows])
        y = np.array([r["f1"] for r in rows])
        o = np.argsort(x)
        ax.plot(x[o], y[o], "-o", color=COLOURS[name], linewidth=2.0,
                markersize=4.5, markeredgecolor="white", markeredgewidth=1.2,
                zorder=3, label="%s (AURC %.3f)" % (name, R[name]["aurc"]))

    ax.axhline(HUMAN_F1, color="#B03030", linestyle="--", linewidth=1.4,
               zorder=2)
    # inside the axes and away from the legend, which sits lower left
    ax.text(0.30, HUMAN_F1 + 0.010,
            "a second annotator scored as a classifier, %.3f" % HUMAN_F1,
            ha="right", va="bottom", fontsize=8.5, color="#B03030")

    ax.set_xlabel("coverage  (share of messages the model answers about)")
    ax.set_ylabel("F1 on the messages it answered")
    ax.set_xlim(0.22, 1.03)
    ax.set_ylim(0.30, 0.80)
    ax.invert_xaxis()
    ax.grid(True, color="#E4E6E8", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False,
              handlelength=1.8, borderaxespad=0.6)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print("written %s" % OUT)
    for name in ORDER:
        rows = R[name]["rows"]
        f = rows[0]["f1"]
        l = min(rows, key=lambda r: r["actual_coverage"])
        print("  %-22s %.4f at full coverage -> %.4f at %.3f  (z %+.1f)"
              % (name, f, l["f1"], l["actual_coverage"], l["z"]))


if __name__ == "__main__":
    main()
