# -*- coding: utf-8 -*-
"""Is the hybrid's gain real, or an artefact of stacking?

The two-component hybrid gains 0.024 F1 on the broad labels and improves
ROC-AUC on both. Before that is reported as evidence that the components are
complementary, it has to survive a control, because cross-fitted stacking on
out-of-fold scores has a known optimism and could manufacture a gain from
nothing.

The control is a fusion of two components that see almost the same thing:
logistic regression and a linear support vector machine, both over the same
word tf-idf features. If fusing those produces a comparable gain, ours is a
property of the procedure rather than of the signals.

A third arm adds a component that is genuinely different in kind - character
n-grams, which see morphology and formatting where word n-grams see
vocabulary - to test the other direction: if diversity is what pays, a
different view of the same text should pay too.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_paths                                          # noqa: E402

import io
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold

import strengthen as S

D = os.path.dirname(io_paths.result_out(".keep"))
OUT = os.path.join(D, "RESULTS_HYBRID_CONTROL.json")
FOLDS, SEED = 5, 42


def load(name):
    return json.load(io.open(os.path.join(D, name), encoding="utf-8"))


def pick_threshold(y, s):
    grid = np.unique(np.round(np.linspace(0.01, 0.99, 99), 3))
    return float(max(grid, key=lambda t: f1_score(
        y, (s >= t).astype(int), zero_division=0)))


def char_scores(X, y, g):
    """Character n-grams, scored out-of-fold on the same thread-disjoint folds
    as everything else, so its scores are comparable to the released ones."""
    V = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                        sublinear_tf=True, max_features=200000).fit_transform(X)
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=FOLDS).split(X, y, g):
        m = LogisticRegression(max_iter=3000,
                               class_weight="balanced").fit(V[tr], y[tr])
        out[te] = m.predict_proba(V[te])[:, 1]
    return out


def fuse(cols, y, g):
    """Cross-fitted fusion, identical machinery to hybrid_cpu.py."""
    s = np.zeros(len(y))
    p = np.zeros(len(y), dtype=int)
    for tr, te in GroupKFold(n_splits=FOLDS).split(np.zeros(len(y)), y, g):
        Ftr, Fte = [], []
        for c in cols:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(c[tr], y[tr])
            Ftr.append(iso.predict(c[tr]))
            Fte.append(iso.predict(c[te]))
        Ftr, Fte = np.column_stack(Ftr), np.column_stack(Fte)
        fu = LogisticRegression(max_iter=1000,
                                class_weight="balanced").fit(Ftr, y[tr])
        thr = pick_threshold(y[tr], fu.predict_proba(Ftr)[:, 1])
        sc = fu.predict_proba(Fte)[:, 1]
        s[te] = sc
        p[te] = (sc >= thr).astype(int)
    return {"f1": round(float(f1_score(y, p, zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(y, p)), 4),
            "roc_auc": round(float(roc_auc_score(y, s)), 4)}


def main():
    P = load("PREDICTIONS_FINAL.json")
    df = S.build(dedup=True)
    X, g = df.text.tolist(), df.thread_key.values
    out = {"note": "controls for the two-component hybrid",
           "environment": S.ENV}

    for lab in ("strict", "broad"):
        y = (df.label if lab == "strict" else df.label_either).values
        enc = np.array(P["transformer_grouped_%s" % lab]["score"])
        lr = np.array(P["LogReg_grouped_%s" % lab]["score"])
        svm = np.array(P["LinearSVM_grouped_%s" % lab]["score"])
        print("=== %s labels ===" % lab, flush=True)
        print("  scoring character n-grams ...", flush=True)
        ch = char_scores(X, y, g)

        singles = {}
        for nm, col in (("encoder", enc), ("word tf-idf", lr),
                        ("linear SVM", svm), ("char n-grams", ch)):
            thr_s = np.zeros(len(y))
            pp = np.zeros(len(y), dtype=int)
            for tr, te in GroupKFold(n_splits=FOLDS).split(np.zeros(len(y)), y, g):
                t = pick_threshold(y[tr], col[tr])
                thr_s[te] = col[te]
                pp[te] = (col[te] >= t).astype(int)
            singles[nm] = {"f1": round(float(f1_score(y, pp, zero_division=0)), 4),
                           "roc_auc": round(float(roc_auc_score(y, col)), 4)}
            print("    %-14s F1 %.4f  ROC %.4f"
                  % (nm, singles[nm]["f1"], singles[nm]["roc_auc"]), flush=True)

        arms = {
            "encoder + word tf-idf": fuse([enc, lr], y, g),
            "CONTROL: word tf-idf + linear SVM": fuse([lr, svm], y, g),
            "encoder + char n-grams": fuse([enc, ch], y, g),
            "encoder + word + char": fuse([enc, lr, ch], y, g),
        }
        print()
        for nm, r in arms.items():
            print("    %-36s F1 %.4f  MCC %.4f  ROC %.4f"
                  % (nm, r["f1"], r["mcc"], r["roc_auc"]), flush=True)

        ctrl = arms["CONTROL: word tf-idf + linear SVM"]["f1"]
        base_ctrl = max(singles["word tf-idf"]["f1"], singles["linear SVM"]["f1"])
        real = arms["encoder + word tf-idf"]["f1"]
        base_real = max(singles["encoder"]["f1"], singles["word tf-idf"]["f1"])
        print()
        print("    control gain (redundant pair):   %+.4f" % (ctrl - base_ctrl))
        print("    hybrid  gain (encoder + tf-idf): %+.4f" % (real - base_real))
        print(flush=True)
        out[lab] = {"singles": singles, "fusions": arms,
                    "control_gain": round(ctrl - base_ctrl, 4),
                    "hybrid_gain": round(real - base_real, 4)}

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=2)
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
