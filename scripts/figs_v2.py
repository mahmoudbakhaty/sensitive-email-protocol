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
    """Dot-and-whisker: pooled F1 with its 95% interval, both label sets.

    This replaced a per-fold scatter. The scatter showed dispersion, which is
    worth knowing, but the question the paper actually asks is whether the
    methods are separable - and an interval plot answers that by eye. The
    intervals are the thread-level bootstrap intervals from Tables III and IV,
    so the figure and the tables cannot drift apart.

    Ordering is by point estimate within each panel, and the trivial
    all-positive floor is drawn because a method below it has shown nothing.
    """
    panels = [
        ("STRICT  (250 positives)", 0.3064, 0.744,
         [("LinearSVM", 0.2685, 0.2065, 0.3307),
          ("LogReg",    0.3456, 0.2850, 0.4043),
          ("RoBERTa",   0.3587, 0.3086, 0.4077)]),
        ("BROAD  (422 positives)", 0.4678, None,
         [("LinearSVM", 0.4591, 0.4082, 0.5057),
          ("LogReg",    0.5160, 0.4710, 0.5601),
          ("RoBERTa",   0.5451, 0.5031, 0.5843)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 1.95))
    for ax, (title, triv, ceiling, rows) in zip(axes, panels):
        ys = np.arange(len(rows))
        for y, (name, pt, lo, hi) in zip(ys, rows):
            ax.plot([lo, hi], [y, y], color=MID, linewidth=1.3,
                    solid_capstyle="butt", zorder=2)
            for x in (lo, hi):
                ax.plot([x, x], [y - 0.13, y + 0.13], color=MID,
                        linewidth=1.3, zorder=2)
            ax.scatter([pt], [y], s=30, color=ACCENT, zorder=4,
                       edgecolor="white", linewidth=0.8)
            ax.text(hi + 0.012, y, "%.3f" % pt, fontsize=6.8, color=INK,
                    va="center", ha="left")
        ax.axvline(triv, color=INK, linewidth=1.0, linestyle=(0, (4, 2)),
                   zorder=1)
        if ceiling is not None:
            ax.axvline(ceiling, color=ACCENT, linewidth=1.2, zorder=1)
            ax.text(ceiling, -0.52, " human %.3f" % ceiling,
                    fontsize=6.8, color=ACCENT, ha="left", va="bottom")
        # below the lowest row, where nothing else is drawn
        ax.text(triv, -0.52, " trivial %.3f" % triv, fontsize=6.8,
                color=INK, ha="left", va="bottom")
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
        ax.set_ylim(-0.6, len(rows) - 0.25)
        lo_all = min(r[2] for r in rows)
        hi_all = max(r[3] for r in rows)
        right = max(hi_all, ceiling or 0)
        ax.set_xlim(min(lo_all, triv) - 0.03, right + 0.075)
        ax.set_xlabel("F1 (pooled out-of-fold, 95% CI)", fontsize=7)
        ax.set_title(title, fontsize=7.5, color=INK, pad=4)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("%s  (interval plot, both label sets)" % os.path.basename(path))


if __name__ == "__main__":
    fig_keywords(os.path.join(HERE, "fig_v2_keywords.png"))
    fig_folds(os.path.join(HERE, "fig_v2_folds.png"))
