# -*- coding: utf-8 -*-
"""The two figures for paper v2.

Both are print figures for a two-column paper, so they are built to survive
being reduced to 8 cm wide and photocopied in grey: serif type to match the
body text, no colour carrying meaning on its own, no chartjunk.

Fig. 1 is the paper's central claim made visible. The text says "no lexical
shortcut exists"; the figure shows the entire vocabulary bunched at the
trivial floor, which is a far harder claim to wave away than a sentence.

Fig. 2 shows why the paper declines to rank the methods: the encoder's
per-fold spread swallows the gap between it and the baselines.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.dirname(HERE)
sys.path.insert(0, SCRATCH)
os.environ.setdefault("DATA_DIR", "enron_with_categories")

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

TRIVIAL = 0.3064


def word_f1s():
    """F1 of every single term used alone as a classifier. Cached to JSON."""
    cache = os.path.join(HERE, "word_f1_cache.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    os.chdir(SCRATCH)
    import colab_v2 as C
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import f1_score
    C.fetch_data()
    df, _ = C.build()
    y = df.label.values
    cv = CountVectorizer(min_df=5, binary=True, max_features=20000)
    B = cv.fit_transform(df.text.tolist()).tocsc()
    vocab = cv.get_feature_names_out()
    out = []
    for j in range(B.shape[1]):
        col = np.asarray(B[:, j].todense()).ravel()
        if col.sum() < 10:
            continue
        out.append([str(vocab[j]), round(float(f1_score(y, col,
                                                        zero_division=0)), 5)])
    json.dump(out, open(cache, "w"))
    return out


def fig_keywords(path):
    data = word_f1s()
    vals = np.array([v for _, v in data])
    best_w, best_v = max(data, key=lambda kv: kv[1])

    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    ax.hist(vals, bins=60, color=LIGHT, edgecolor=MID, linewidth=0.4)
    ax.axvline(TRIVIAL, color=ACCENT, linewidth=1.4, zorder=3)
    ax.annotate("trivial floor\n%.3f" % TRIVIAL,
                xy=(TRIVIAL, ax.get_ylim()[1] * 0.78),
                xytext=(TRIVIAL - 0.105, ax.get_ylim()[1] * 0.60),
                fontsize=7, color=ACCENT, ha="center",
                arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=0.7))
    ax.annotate("strongest word\n'%s'  %.3f" % (best_w, best_v),
                xy=(best_v, 2), xytext=(best_v - 0.02, ax.get_ylim()[1] * 0.34),
                fontsize=7, color=INK, ha="right",
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.7))
    ax.set_xlabel("F1 of a single term used alone as a classifier")
    ax.set_ylabel("terms")
    ax.set_xlim(0, max(0.36, vals.max() * 1.05))
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("%s  (%d terms scored, max %.3f = '%s')"
          % (os.path.basename(path), len(vals), best_v, best_w))


def fig_folds(path):
    """Per-fold spread for both label definitions, NOT a trend.

    Fold order is arbitrary, so points are left unconnected: joining them
    would draw a decline that does not exist. The two panels share a y-axis
    so the reader can see that the encoder's whole broad-label spread sits
    above its whole strict-label spread, and that only in the broad panel
    does every fold clear the logistic-regression line.
    """
    panels = [
        ("STRICT  (trivial 0.306)",
         [0.4474, 0.3804, 0.3717, 0.3902, 0.2892],
         [("LogReg", 0.346), ("LinearSVM", 0.264), ("trivial", 0.3064)]),
        ("BROAD  (trivial 0.468)",
         [0.5741, 0.5556, 0.5746, 0.5398, 0.5189],
         [("LogReg", 0.505), ("LinearSVM", 0.472), ("trivial", 0.4678)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3), sharey=True)
    for ax, (title, folds, refs) in zip(axes, panels):
        m, sd = float(np.mean(folds)), float(np.std(folds))
        ax.axhspan(m - sd, m + sd, color=ACCENT, alpha=0.09, zorder=1,
                   linewidth=0)
        ax.axhline(m, color=ACCENT, linewidth=1.0, zorder=3)
        # Label positions are nudged apart when two reference lines are
        # close together (broad: LinearSVM 0.472 vs trivial 0.468), so the
        # line stays at its true value while its label stays readable.
        GAP = 0.022
        placed = []
        for name, v in sorted(refs, key=lambda r: r[1]):
            col = INK if name == "trivial" else MID
            ax.axhline(v, xmax=0.68, color=col, linewidth=0.9,
                       linestyle=(0, (4, 2)), zorder=2)
            ly = v
            while any(abs(ly - q) < GAP for q in placed):
                ly += GAP / 2
            placed.append(ly)
            ax.text(5.7, ly, name, fontsize=6.5, color=col, va="center",
                    ha="left")
        ax.scatter(np.arange(1, 6), folds, s=24, color=ACCENT, zorder=5,
                   edgecolor="white", linewidth=0.8)
        ax.set_title("%s   mean %.3f $\pm$ %.3f" % (title, m, sd),
                     fontsize=7.5, color=INK, pad=4)
        ax.set_xticks(np.arange(1, 6))
        ax.set_xlabel("fold (order arbitrary)", fontsize=7)
        ax.set_xlim(0.55, 6.6)
        ax.spines["bottom"].set_bounds(1, 5)
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0.22, 0.62)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("%s  (strict %.3f, broad %.3f)"
          % (os.path.basename(path), np.mean(panels[0][1]),
             np.mean(panels[1][1])))


if __name__ == "__main__":
    fig_keywords(os.path.join(HERE, "fig_v2_keywords.png"))
    fig_folds(os.path.join(HERE, "fig_v2_folds.png"))
