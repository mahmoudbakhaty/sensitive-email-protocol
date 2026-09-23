# -*- coding: utf-8 -*-
"""A two-component hybrid from the released scores, while the GPU quota is out.

The full framework combines three signals and has to be run on a GPU. Two of
the three already have their per-message out-of-fold scores in
PREDICTIONS_FINAL.json, so the encoder-plus-classical half of the question can
be answered now.

Fitting the combination needs care. Those scores are out-of-fold predictions on
the test folds, so fitting a fusion on them directly would be fitting on the
test set - the thing this paper exists to complain about. Instead the fusion is
cross-fitted over the same thread groups: for each outer fold the calibrators,
the fusion weights and the threshold are fitted on the other folds' messages
and applied to this one, so the fusion never sees a score for a message it is
about to predict.

One caveat has to be stated rather than buried, and it is the reason this is
preliminary. The base scores for the other folds came from models that were
trained on data including this fold, which is the standard optimism of stacking
on out-of-fold predictions. The framework run avoids it properly by fitting
the fusion inside each fold on held-out validation threads. Expect this number
to be slightly optimistic against that one.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_paths                                          # noqa: E402

import io
import json
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import GroupKFold

import strengthen as S


OUT = io_paths.result_out("RESULTS_HYBRID_CPU.json")
FOLDS, SEED, N_BOOT = 5, 42, 2000


def load(name):
    """Read through io_paths: the repository's results/ counts as
    a source, which RESULTS_DIR alone does not."""
    return json.load(io.open(io_paths.result_in(name, required=True),
                             encoding="utf-8"))


def pick_threshold(y, s):
    grid = np.unique(np.round(np.linspace(0.01, 0.99, 99), 3))
    return float(max(grid, key=lambda t: f1_score(
        y, (s >= t).astype(int), zero_division=0)))


def boot(y, pred, g, n=N_BOOT):
    rng = np.random.RandomState(SEED)
    uniq = np.array(sorted(set(g)))
    by = {t: np.where(g == t)[0] for t in uniq}
    f1s = []
    for _ in range(n):
        idx = np.concatenate([by[t] for t in rng.choice(uniq, len(uniq), True)])
        if len(set(y[idx])) < 2:
            continue
        f1s.append(f1_score(y[idx], pred[idx], zero_division=0))
    return [round(float(np.percentile(f1s, 2.5)), 4),
            round(float(np.percentile(f1s, 97.5)), 4)]


def report(name, y, pred, score, g, floor):
    return {"f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(y, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y, score)), 4),
            "pr_auc": round(float(average_precision_score(y, score)), 4),
            "f1_ci95": boot(y, pred, g), "trivial_floor": floor}


def main():
    P = load("PREDICTIONS_FINAL.json")
    R = load("RESULTS_FINAL.json")
    df = S.build(dedup=True)
    g = df.thread_key.values
    assert len(df) == R["dataset"]["messages"], "corpus does not match the run"

    out = {"note": "cross-fitted fusion of the encoder and the classical "
                   "model, from released out-of-fold scores; preliminary, "
                   "see the caveat in the script",
           "components": ["encoder", "classical"],
           "environment": S.ENV}

    for lab in ("strict", "broad"):
        y = (df.label if lab == "strict" else df.label_either).values
        floor = round(float(2 * y.mean() / (1 + y.mean())), 4)
        enc = np.array(P["transformer_grouped_%s" % lab]["score"])
        cls = np.array(P["LogReg_grouped_%s" % lab]["score"])
        print("=== %s labels (floor %.4f) ===" % (lab, floor))

        hyb_s = np.zeros(len(y))
        hyb_p = np.zeros(len(y), dtype=int)
        weights = []
        for tr, te in GroupKFold(n_splits=FOLDS).split(np.zeros(len(y)), y, g):
            assert not (set(g[tr]) & set(g[te])), "thread leaked"
            cal = {}
            for nm, col in (("encoder", enc), ("classical", cls)):
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0,
                                         y_max=1.0)
                iso.fit(col[tr], y[tr])
                cal[nm] = iso
            Ftr = np.column_stack([cal["encoder"].predict(enc[tr]),
                                   cal["classical"].predict(cls[tr])])
            Fte = np.column_stack([cal["encoder"].predict(enc[te]),
                                   cal["classical"].predict(cls[te])])
            fu = LogisticRegression(max_iter=1000,
                                    class_weight="balanced").fit(Ftr, y[tr])
            thr = pick_threshold(y[tr], fu.predict_proba(Ftr)[:, 1])
            s = fu.predict_proba(Fte)[:, 1]
            hyb_s[te] = s
            hyb_p[te] = (s >= thr).astype(int)
            weights.append({"encoder": round(float(fu.coef_[0][0]), 4),
                            "classical": round(float(fu.coef_[0][1]), 4)})

        rec = {}
        for nm, key in (("encoder", "transformer_grouped_%s" % lab),
                        ("classical", "LogReg_grouped_%s" % lab)):
            r = R[key]
            rec[nm] = {"f1": r["f1"], "mcc": r["mcc"],
                       "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
                       "f1_ci95": r["f1_ci95"], "trivial_floor": floor}
            print("  %-10s F1 %.4f  MCC %.4f  ROC %.4f"
                  % (nm, r["f1"], r["mcc"], r["roc_auc"]))
        rec["hybrid"] = report("hybrid", y, hyb_p, hyb_s, g, floor)
        h = rec["hybrid"]
        print("  %-10s F1 %.4f  MCC %.4f  ROC %.4f  CI [%.3f, %.3f]"
              % ("hybrid", h["f1"], h["mcc"], h["roc_auc"],
                 h["f1_ci95"][0], h["f1_ci95"][1]))
        best = max(rec[n]["f1"] for n in ("encoder", "classical"))
        print("  gain over the better component: %+.4f" % (h["f1"] - best))
        print("  fusion weights per fold: %s" % weights)
        print()
        rec["weights"] = weights
        rec["gain_over_best_component"] = round(h["f1"] - best, 4)
        out[lab] = rec

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=2)
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
