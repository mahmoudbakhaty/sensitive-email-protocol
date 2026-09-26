# -*- coding: utf-8 -*-
"""Addendum to run 2: the three things the paper claims or should claim.

Reviewing v2 against the supervisor's review turned up two defects and one
regression:

  1. The artifact section promises "per-fold result records including raw
     probabilities". The main run computes out-of-fold probabilities and then
     throws them away, so the promise was not kept. This saves them.
  2. v1 reported bootstrap confidence intervals and the review named them as
     one of the things lifting the paper above typical MSc work. v2 dropped
     them entirely. This restores them.
  3. The paper's own reporting rule says pooled and fold-mean +/- SD figures
     are both given; in practice v2 gives a fold-mean once. This computes
     per-fold scores for every classical baseline.

It also runs the paired comparison the paper currently declines to make. The
folds are identical across methods by construction (GroupKFold is
deterministic on the same groups), so transformer and baseline scores on
fold i are paired observations and a Wilcoxon signed-rank test is licensed.

BOOTSTRAP RESAMPLES THREADS, NOT MESSAGES. Messages in a thread are not
independent - that is the paper's whole argument for grouped splitting - so an
item-level bootstrap would understate the interval for exactly the reason the
paper spends a section on. Resampling whole threads keeps the dependence
structure intact.

Run this in the SAME session as the main notebook, so the library versions
that produced Tables III-V also produce these. It takes about two minutes:
nothing here trains a transformer.
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, matthews_corrcoef

SEED = 42
FOLDS = 5
N_BOOT = 2000
MAIN_JSON = "RESULTS_v2.json"
OUT_JSON = "RESULTS_v2_addendum.json"
PRED_JSON = "predictions_v2.json"


def thread_bootstrap(y, pred, groups, n=N_BOOT, seed=SEED):
    """Percentile CI over threads resampled with replacement.

    Returns (lo, hi) for F1 and MCC. Threads rather than messages: a reply and
    its parent are not independent draws, and pretending otherwise would give
    an interval narrower than the data supports.
    """
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(groups)))
    idx_by_thread = {t: np.where(groups == t)[0] for t in uniq}
    f1s, mccs = [], []
    for _ in range(n):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_thread[t] for t in picked])
        yy, pp = y[idx], pred[idx]
        if len(set(yy)) < 2:
            continue
        f1s.append(f1_score(yy, pp, zero_division=0))
        mccs.append(matthews_corrcoef(yy, pp))
    q = lambda a: (round(float(np.percentile(a, 2.5)), 4),
                   round(float(np.percentile(a, 97.5)), 4))
    return {"f1_ci95": q(f1s), "mcc_ci95": q(mccs), "resamples": len(f1s)}


def per_fold_and_ci(X, y, g, label_name, OUT, PRED):
    """Per-fold scores, pooled scores, bootstrap CIs, and saved predictions."""
    Xv = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                         max_features=50000).fit_transform(X)
    for mdl_fn, name, meth in (
            (lambda: LogisticRegression(max_iter=2000,
                                        class_weight="balanced"),
             "LogReg", "predict_proba"),
            (lambda: LinearSVC(C=0.5, class_weight="balanced",
                                      random_state=42),
             "LinearSVM", "decision_function")):
        oof_pred = np.zeros(len(y), dtype=int)
        oof_score = np.zeros(len(y), dtype=float)
        fold_f1, fold_mcc = [], []
        for tr, te in GroupKFold(n_splits=FOLDS).split(X, y, g):
            assert not (set(g[tr]) & set(g[te])), "thread leaked train/test"
            m = mdl_fn().fit(Xv[tr], y[tr])
            p = m.predict(Xv[te])
            s = getattr(m, meth)(Xv[te])
            if np.ndim(s) == 2:
                s = s[:, 1]
            oof_pred[te], oof_score[te] = p, s
            fold_f1.append(round(float(f1_score(y[te], p, zero_division=0)), 4))
            fold_mcc.append(round(float(matthews_corrcoef(y[te], p)), 4))

        key = "%s_%s" % (name, label_name)
        rec = {"pooled_f1": round(float(f1_score(y, oof_pred,
                                                 zero_division=0)), 4),
               "pooled_mcc": round(float(matthews_corrcoef(y, oof_pred)), 4),
               "folds_f1": fold_f1, "folds_mcc": fold_mcc,
               "fold_mean_f1": round(float(np.mean(fold_f1)), 4),
               "fold_std_f1": round(float(np.std(fold_f1)), 4),
               "fold_mean_mcc": round(float(np.mean(fold_mcc)), 4),
               "fold_std_mcc": round(float(np.std(fold_mcc)), 4)}
        rec.update(thread_bootstrap(y, oof_pred, g))
        OUT[key] = rec
        PRED[key] = {"pred": oof_pred.tolist(),
                     "score": [round(float(v), 6) for v in oof_score]}
        print("%-10s %-7s pooled F1 %.3f | fold-mean %.3f +- %.3f | "
              "F1 95%% CI [%.3f, %.3f]"
              % (name, label_name, rec["pooled_f1"], rec["fold_mean_f1"],
                 rec["fold_std_f1"], rec["f1_ci95"][0], rec["f1_ci95"][1]),
              flush=True)


# Per-fold transformer F1 from run 2, so this script is standalone: a fresh
# Kaggle session has no RESULTS_v2.json, and the numbers are published in the
# paper anyway. Verified reproducible - the Arabic run re-trained the same
# English model in a later session and returned 0.447 0.380 0.372 0.390 for
# folds 1-4, matching these to three decimals.
TRANSFORMER_FOLDS = {
    "strict": [0.4474, 0.3804, 0.3717, 0.3902, 0.2892],
    "broad": [0.5741, 0.5556, 0.5746, 0.5398, 0.5189],
}


def paired_tests(OUT):
    """Transformer vs each baseline, paired on identical folds."""
    if os.path.exists(MAIN_JSON):
        main = json.load(open(MAIN_JSON, encoding="utf-8"))
        folds = {k: main["transformer_%s" % k]["folds_f1"]
                 for k in ("strict", "broad") if "transformer_%s" % k in main}
        print("transformer folds read from", MAIN_JSON)
    else:
        folds = TRANSFORMER_FOLDS
        print("RESULTS_v2.json absent - using the recorded run-2 fold scores")
    for label_name in ("strict", "broad"):
        if label_name not in folds:
            continue
        tf = folds[label_name]
        for base in ("LogReg", "LinearSVM"):
            bkey = "%s_%s" % (base, label_name)
            if bkey not in OUT:
                continue
            bf = OUT[bkey]["folds_f1"]
            diff = [a - b for a, b in zip(tf, bf)]
            # n = 5 folds: Wilcoxon cannot go below p = 0.0625 here, so the
            # test can never reach 0.05 no matter how consistent the win is.
            # Report it with that ceiling stated rather than implying the
            # result failed a threshold it could not have reached.
            try:
                stat, pv = wilcoxon(tf, bf)
                pv = round(float(pv), 4)
            except ValueError:
                stat, pv = None, None
            k = "paired_%s_vs_%s_%s" % ("transformer", base, label_name)
            OUT[k] = {"transformer_folds": tf, "baseline_folds": bf,
                      "differences": [round(d, 4) for d in diff],
                      "wins": int(sum(1 for d in diff if d > 0)),
                      "n_folds": len(diff),
                      "mean_difference": round(float(np.mean(diff)), 4),
                      "wilcoxon_p": pv,
                      "min_attainable_p_at_n5": 0.0625}
            print("  transformer vs %-10s %-7s wins %d/%d  mean diff %+.4f  "
                  "p=%s (floor 0.0625)"
                  % (base, label_name, OUT[k]["wins"], len(diff),
                     OUT[k]["mean_difference"], pv), flush=True)


def main():
    import colab_v2 as C
    C.fetch_data()
    df, _ = C.build()
    X = df.text.tolist()
    g = df.thread_key.values
    OUT = {"note": "addendum to RESULTS_v2.json - same environment, same "
                   "folds; bootstrap resamples THREADS not messages"}
    PRED = {}
    for label_name, y in (("strict", df.label.values),
                          ("broad", df.label_either.values)):
        print("--- %s labels ---" % label_name, flush=True)
        per_fold_and_ci(X, y, g, label_name, OUT, PRED)
    print("--- paired comparisons ---", flush=True)
    paired_tests(OUT)

    json.dump(OUT, open(OUT_JSON, "w"), indent=2)
    json.dump(PRED, open(PRED_JSON, "w"))
    print("\nsaved -> %s and %s (%d predictions per model)"
          % (OUT_JSON, PRED_JSON, len(df)))


if __name__ == "__main__":
    main()
