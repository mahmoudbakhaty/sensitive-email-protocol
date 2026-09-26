# -*- coding: utf-8 -*-
"""Every model under one treatment, so Tables III and IV compare like with like.

The published tables give the classical models a fixed 0.5 threshold and the
whole training fold, and give the encoder a validation-selected threshold and
70% of the fold. Measuring that gap showed it is worth up to 0.034 F1, which is
as large as the differences the tables are read for, and that correcting it
reverses the encoder's reported lead on the broad labels.

This puts every non-GPU model under the encoder's treatment exactly: fitted on
the training fold minus 30% of its threads, threshold chosen on those threads,
pooled out-of-fold, thread-level intervals. The encoder's published record
already uses it, so the rows become comparable.

LinearSVC has no probability output, so the fixed 0.05-0.95 grid is meaningless
for it. Its threshold is selected over the quantiles of its own validation
decision values instead, which is the same procedure expressed on the scale the
model actually produces.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_paths                                          # noqa: E402

import hashlib
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

OUT = io_paths.result_out("RESULTS_UNIFORM.json")
FOLDS, SEED, VAL_FRAC, N_BOOT = 5, 42, 0.30, 2000


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


def fingerprint(n, y, g):
    """The fold assignment, hashed. Identical definition to final_run's, so
    the two runs' partitions can be compared as strings rather than taken on
    trust - which is what a reader was being asked to do."""
    folds = list(GroupKFold(n_splits=FOLDS).split(np.zeros(n), y, g))
    return hashlib.md5("|".join(",".join(map(str, te)) for _, te in folds)
                       .encode()).hexdigest()[:16]


def grid_for(pv, calibrated):
    if calibrated:
        return np.arange(0.05, 0.96, 0.01)
    # the decision function lives on its own scale; use its own quantiles
    return np.unique(np.percentile(pv, np.arange(2, 99, 1)))


def run(make, feats, scorer, calibrated, y, g):
    score = np.zeros(len(y))
    pred = np.zeros(len(y), dtype=int)
    thresholds, folds_f1 = [], []
    for tr, te in GroupKFold(n_splits=FOLDS).split(np.zeros(len(y)), y, g):
        assert not (set(g[tr]) & set(g[te])), "thread leaked"
        rng = np.random.RandomState(SEED)
        th = np.array(sorted(set(g[tr])))
        rng.shuffle(th)
        va_th = set(th[:max(1, int(VAL_FRAC * len(th)))])
        va = np.array([i for i in tr if g[i] in va_th])
        tr2 = np.array([i for i in tr if g[i] not in va_th])
        m = make().fit(feats[tr2], y[tr2])
        pv = scorer(m, feats[va])
        grid = grid_for(pv, calibrated)
        t = float(max(grid, key=lambda x: f1_score(
            y[va], (pv >= x).astype(int), zero_division=0)))
        pt = scorer(m, feats[te])
        score[te] = pt
        pred[te] = (pt >= t).astype(int)
        thresholds.append(round(t, 4))
        folds_f1.append(round(float(f1_score(y[te], pred[te],
                                             zero_division=0)), 4))
    per_message = {"pred": [int(v) for v in pred],
                   "score": [round(float(v), 6) for v in score]}
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    r = {"per_message": per_message,
         "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
         "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
         "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
         "mcc": round(float(matthews_corrcoef(y, pred)), 4),
         "roc_auc": round(float(roc_auc_score(y, score)), 4),
         "pr_auc": round(float(average_precision_score(y, score)), 4),
         "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
         "thresholds": thresholds, "folds_f1": folds_f1,
         "fold_mean_f1": round(float(np.mean(folds_f1)), 4),
         "fold_sd_f1": round(float(np.std(folds_f1)), 4)}
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
    print("messages %d | threads %d" % (len(df), df.thread_key.nunique()),
          flush=True)

    models = [
        ("LinearSVM", lambda: LinearSVC(C=0.5, class_weight="balanced"),
         word, lambda m, f: m.decision_function(f), False),
        ("LogReg", lambda: LogisticRegression(max_iter=2000,
                                              class_weight="balanced"),
         word, lambda m, f: m.predict_proba(f)[:, 1], True),
        ("CharNgram", lambda: LogisticRegression(max_iter=3000,
                                                 class_weight="balanced"),
         char, lambda m, f: m.predict_proba(f)[:, 1], True),
    ]

    # The fingerprint a reader needs to confirm these rows share the
    # encoder's partition. It was absent, so Section VII-B's claim that every
    # method sees the identical partition could only be taken on the code.
    fp_strict = fingerprint(len(df), df.label.values, g)
    PUBLISHED = "63e3aea5c3d37629"
    assert fp_strict == PUBLISHED, (
        "this run's partition is %s, the released one is %s - the classical "
        "rows would not share the encoder's folds" % (fp_strict, PUBLISHED))
    print("fold fingerprint %s (matches the released partition)" % fp_strict,
          flush=True)

    out = {"environment": S.ENV,
           "fold_fingerprint": fp_strict,
           "dataset": {"messages": int(len(df)),
                       "threads": int(df.thread_key.nunique())},
           "note": "every model under the encoder's treatment: fitted on the "
                   "training fold minus 30%% of its threads, threshold chosen "
                   "on those threads"}

    for lab in ("strict", "broad"):
        y = (df.label if lab == "strict" else df.label_either).values
        floor = round(float(2 * y.mean() / (1 + y.mean())), 4)
        out["trivial_floor_%s" % lab] = floor
        print()
        print("=== %s labels (floor %.4f) ===" % (lab, floor), flush=True)
        for name, make, feats, scorer, cal in models:
            r = run(make, feats, scorer, cal, y, g)
            r["trivial_floor"] = floor
            out["%s_uniform_%s" % (name, lab)] = r
            print("  %-10s F1 %.4f  P %.4f  R %.4f  MCC %.4f  ROC %.4f  "
                  "CI [%.3f, %.3f]"
                  % (name, r["f1"], r["precision"], r["recall"], r["mcc"],
                     r["roc_auc"], r["f1_ci95"][0], r["f1_ci95"][1]),
                  flush=True)

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=2)
    print()
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
