# -*- coding: utf-8 -*-
"""Character n-grams under the paper's exact protocol.

The control experiment turned up, incidentally, that character n-grams beat
the fine-tuned encoder on both label sets. That comparison was made inside its
own experiment, with thresholds chosen the way the fusion chooses them, so it
cannot be quoted against Table III.

This runs the same feature set under the protocol the paper actually uses:
thread-disjoint GroupKFold, a validation split of whole threads carved out of
each training fold, the threshold selected there and nowhere else, pooled
out-of-fold scoring, thread-level bootstrap intervals, and the trivial floor
reported beside it. The output is directly comparable to Tables III and IV.

Whether it survives that is the question. A cheap character-level baseline
beating a fine-tuned transformer is the kind of claim that usually evaporates
when the thresholds are chosen honestly.
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

import strengthen as S

OUT = io_paths.result_out("RESULTS_CHAR_NGRAM.json")
FOLDS, SEED, VAL_FRAC, N_BOOT = 5, 42, 0.30, 2000


def metrics(y, pred, score, floor):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(y, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y, score)), 4),
            "pr_auc": round(float(average_precision_score(y, score)), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "trivial_floor": floor}


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


def main():
    df = S.build(dedup=True)
    X, g = df.text.tolist(), df.thread_key.values
    print("messages %d | threads %d" % (len(df), df.thread_key.nunique()))
    V = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                        sublinear_tf=True, max_features=200000).fit_transform(X)
    print("character features: %d" % V.shape[1], flush=True)

    out = {"environment": S.ENV,
           "features": {"analyzer": "char_wb", "ngram_range": [3, 5],
                        "min_df": 3, "max_features": 200000,
                        "vocabulary": int(V.shape[1])},
           "note": "the paper's protocol exactly: thread-disjoint folds, "
                   "threshold on a validation split of whole training threads, "
                   "pooled out-of-fold, thread-level bootstrap"}

    for lab in ("strict", "broad"):
        y = (df.label if lab == "strict" else df.label_either).values
        floor = round(float(2 * y.mean() / (1 + y.mean())), 4)
        score = np.zeros(len(y))
        pred = np.zeros(len(y), dtype=int)
        thresholds, folds_f1 = [], []

        for tr, te in GroupKFold(n_splits=FOLDS).split(X, y, g):
            assert not (set(g[tr]) & set(g[te])), "thread leaked"
            rng = np.random.RandomState(SEED)
            th = np.array(sorted(set(g[tr])))
            rng.shuffle(th)
            va_th = set(th[:max(1, int(VAL_FRAC * len(th)))])
            va = np.array([i for i in tr if g[i] in va_th])
            tr2 = np.array([i for i in tr if g[i] not in va_th])

            m = LogisticRegression(max_iter=3000,
                                   class_weight="balanced").fit(V[tr2], y[tr2])
            pv = m.predict_proba(V[va])[:, 1]
            t = float(max(np.arange(0.05, 0.96, 0.01),
                          key=lambda x: f1_score(y[va], (pv >= x).astype(int),
                                                 zero_division=0)))
            pt = m.predict_proba(V[te])[:, 1]
            score[te] = pt
            pred[te] = (pt >= t).astype(int)
            thresholds.append(round(t, 3))
            folds_f1.append(round(float(f1_score(y[te], pred[te],
                                                 zero_division=0)), 4))

        r = metrics(y, pred, score, floor)
        r.update(boot(y, pred, g))
        r.update(thresholds=thresholds, folds_f1=folds_f1,
                 fold_mean_f1=round(float(np.mean(folds_f1)), 4),
                 fold_sd_f1=round(float(np.std(folds_f1)), 4))
        out["char_grouped_%s" % lab] = r
        print("  %-7s F1 %.4f  MCC %.4f  ROC %.4f  PR %.4f  CI [%.3f, %.3f]  "
              "floor %.4f" % (lab, r["f1"], r["mcc"], r["roc_auc"],
                              r["pr_auc"], r["f1_ci95"][0], r["f1_ci95"][1],
                              floor), flush=True)
        print("          folds %s  thresholds %s" % (folds_f1, thresholds))

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=2)
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
