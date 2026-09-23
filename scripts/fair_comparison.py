# -*- coding: utf-8 -*-
"""Table III compares models that were not treated the same way.

Checking whether a character n-gram baseline could join Table III turned up an
asymmetry in the table itself. The classical rows call predict(), which is a
fixed 0.5 threshold, and are fitted on the whole training fold. The encoder row
selects its threshold on a validation split of whole threads and is therefore
fitted on 70% of the fold.

Neither is leakage - a fixed threshold is not tuned on anything - so the
protocol was not violated. But the rows are not comparable to each other, and
the two differences push in opposite directions: the encoder gets to move its
operating point, which helps, and trains on less data, which does not. Their
net effect is unknown, and a table whose rows are read against each other
should not leave it unknown.

This runs every non-GPU model under the encoder's treatment exactly: fitted on
the training fold minus 30% of its threads, threshold chosen on those held-out
threads over the same grid, pooled out-of-fold, thread-level intervals. The
encoder's published numbers can then be read against these.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

import strengthen as S

OUT = io_paths.result_out("RESULTS_FAIR_COMPARISON.json")
FOLDS, SEED, VAL_FRAC, N_BOOT = 5, 42, 0.30, 2000
GRID = np.arange(0.05, 0.96, 0.01)


def boot(y, pred, g, n=N_BOOT):
    rng = np.random.RandomState(SEED)
    uniq = np.array(sorted(set(g)))
    by = {t: np.where(g == t)[0] for t in uniq}
    f1s, mccs = [], []
    for _ in range(n):
        idx = np.concatenate([by[t] for t in rng.choice(uniq, len(uniq), True)])
        if len(set(y[idx])) < 2:
            continue
        f1s.append(f1_score(y[idx], pred[idx], zero_division=0))
        mccs.append(matthews_corrcoef(y[idx], pred[idx]))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4),
                   round(float(np.percentile(a, 97.5)), 4)]
    return {"f1_ci95": q(f1s), "mcc_ci95": q(mccs)}


def run_model(make, feats, y, g, scorer, treatment):
    """treatment: 'encoder' mirrors the encoder row; 'published' mirrors the
    classical rows, so the difference between them is measurable."""
    score = np.zeros(len(y))
    pred = np.zeros(len(y), dtype=int)
    thresholds = []
    for tr, te in GroupKFold(n_splits=FOLDS).split(np.zeros(len(y)), y, g):
        assert not (set(g[tr]) & set(g[te])), "thread leaked"
        if treatment == "published":
            m = make().fit(feats[tr], y[tr])
            s = scorer(m, feats[te])
            score[te] = s
            pred[te] = m.predict(feats[te])
            thresholds.append(0.5)
            continue
        rng = np.random.RandomState(SEED)
        th = np.array(sorted(set(g[tr])))
        rng.shuffle(th)
        va_th = set(th[:max(1, int(VAL_FRAC * len(th)))])
        va = np.array([i for i in tr if g[i] in va_th])
        tr2 = np.array([i for i in tr if g[i] not in va_th])
        m = make().fit(feats[tr2], y[tr2])
        pv = scorer(m, feats[va])
        t = float(max(GRID, key=lambda x: f1_score(
            y[va], (pv >= x).astype(int), zero_division=0)))
        pt = scorer(m, feats[te])
        score[te] = pt
        pred[te] = (pt >= t).astype(int)
        thresholds.append(round(t, 3))

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    r = {"f1": round(float(f1_score(y, pred, zero_division=0)), 4),
         "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
         "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
         "mcc": round(float(matthews_corrcoef(y, pred)), 4),
         "roc_auc": round(float(roc_auc_score(y, score)), 4),
         "pr_auc": round(float(average_precision_score(y, score)), 4),
         "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
         "thresholds": thresholds}
    r.update(boot(y, pred, g))
    return r


def main():
    df = S.build(dedup=True)
    X, g = df.text.tolist(), df.thread_key.values
    word = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                           max_features=50000).fit_transform(X)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True,
                           max_features=200000).fit_transform(X)
    print("messages %d | threads %d | word feats %d | char feats %d"
          % (len(df), df.thread_key.nunique(), word.shape[1], char.shape[1]),
          flush=True)

    models = [
        ("LogReg, word",
         lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
         word, lambda m, f: m.predict_proba(f)[:, 1]),
        ("LinearSVM, word",
         lambda: LinearSVC(C=0.5, class_weight="balanced"),
         word, lambda m, f: m.decision_function(f)),
        ("LogReg, char",
         lambda: LogisticRegression(max_iter=3000, class_weight="balanced"),
         char, lambda m, f: m.predict_proba(f)[:, 1]),
    ]

    out = {"environment": S.ENV,
           "note": "every model under both treatments, so the gap between "
                   "them is measured rather than assumed"}

    for lab in ("strict", "broad"):
        y = (df.label if lab == "strict" else df.label_either).values
        floor = round(float(2 * y.mean() / (1 + y.mean())), 4)
        print()
        print("=== %s labels (floor %.4f) ===" % (lab, floor), flush=True)
        print("  %-18s %-22s %-22s" % ("", "published treatment",
                                       "encoder treatment"))
        rec = {"trivial_floor": floor}
        for name, make, feats, scorer in models:
            # LinearSVC has no probability, so its decision function is not on
            # a 0-1 scale; the grid would be meaningless. Rank it by ROC only.
            if "SVM" in name:
                pub = run_model(make, feats, y, g, scorer, "published")
                rec[name] = {"published": pub}
                print("  %-18s F1 %.4f  ROC %.4f   %-22s"
                      % (name, pub["f1"], pub["roc_auc"],
                         "(no calibrated score)"), flush=True)
                continue
            pub = run_model(make, feats, y, g, scorer, "published")
            enc = run_model(make, feats, y, g, scorer, "encoder")
            rec[name] = {"published": pub, "encoder_treatment": enc,
                         "delta_f1": round(enc["f1"] - pub["f1"], 4)}
            print("  %-18s F1 %.4f  ROC %.4f   F1 %.4f  ROC %.4f   %+.4f"
                  % (name, pub["f1"], pub["roc_auc"], enc["f1"],
                     enc["roc_auc"], enc["f1"] - pub["f1"]), flush=True)
        out[lab] = rec

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=2)
    print()
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
