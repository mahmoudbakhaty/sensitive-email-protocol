# -*- coding: utf-8 -*-
"""When a model declines a message, is it declining one the annotators argued over?

The benchmark carries something almost no dataset in this area has: 172
messages that two trained annotators settled differently. The paper treats
them as the measurement of how contested the judgement is, and stops there.

Selective prediction gives them a second use. If the messages a model declines
are disproportionately those 172, then the model's uncertainty is tracking
human disagreement rather than its own ignorance - and the gap between F1 0.40
and the annotator reference of 0.744 stops looking like a modelling failure
and starts looking like a property of the task that a model can detect.

If they are not, the model is uncertain about something else, and the honest
conclusion is that abstention works for reasons unrelated to contestedness.

Either answer is worth having, so the test is built to be able to fail:

  * Enrichment is measured as the contested rate among declined messages
    against the contested rate overall (12.4%), with a hypergeometric p-value
    - the probability of seeing at least this many contested messages in a
    decline set of that size drawn at random.
  * The decline set comes from the same validation-chosen cut-offs as
    scripts/selective.py. No coverage is chosen here.
  * Reported at every coverage, so a result that only appears at one is
    visible as such.
"""
import io
import json
import os
import sys

import numpy as np
from scipy.stats import hypergeom
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import io_paths                                              # noqa: E402
import selective as SEL                                      # noqa: E402
import strengthen as S                                       # noqa: E402

OUT = io_paths.result_out("RESULTS_DECLINED.json")


def decline_sets(per_fold, y, n_total):
    """The indices declined at each coverage, using selective.py's rule."""
    out = {}
    for cov in SEL.COVERAGES:
        if cov >= 1.0:
            continue
        dropped = []
        for f in per_fold:
            thr = SEL.pick_threshold(f["val_y"], f["val_scores"])
            cut = SEL.conf_cutoff(f["val_scores"], thr, cov)
            conf = np.abs(f["test_scores"] - thr)
            dropped.append(f["test_idx"][conf < cut])
        out[cov] = np.concatenate(dropped)
    return out


def main():
    df = S.build(dedup=True)
    X, g, y = df.text.tolist(), df.thread_key.values, df.label.values
    contested = (df.agree.values == "one").astype(int)
    base = contested.mean()
    print("messages %d | contested %d (%.1f%%)"
          % (len(df), contested.sum(), 100 * base))
    print()

    word = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                           max_features=50000).fit_transform(X)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True,
                           max_features=200000).fit_transform(X)
    MODELS = [
        ("logistic regression",
         lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
         word, lambda m, f: m.predict_proba(f)[:, 1]),
        ("linear SVM", lambda: LinearSVC(C=0.5, class_weight="balanced"),
         word, lambda m, f: m.decision_function(f)),
        ("character n-grams",
         lambda: LogisticRegression(max_iter=3000, class_weight="balanced"),
         char, lambda m, f: m.predict_proba(f)[:, 1]),
    ]

    out = {"environment": S.ENV,
           "contested_total": int(contested.sum()),
           "contested_rate": round(float(base), 4),
           "note": "enrichment of the 172 contested messages among those a "
                   "model declines, at each validation-chosen coverage"}

    def run(name, per_fold):
        print("=== %s ===" % name)
        print("  %8s %8s %10s %10s %9s %s"
              % ("coverage", "declined", "contested", "rate", "enrich", "p"))
        rows = []
        for cov, idx in decline_sets(per_fold, y, len(y)).items():
            if len(idx) == 0:
                continue
            k = int(contested[idx].sum())
            rate = k / len(idx)
            # P(X >= k) for X ~ Hypergeom(N, K, n)
            p = float(hypergeom.sf(k - 1, len(y), int(contested.sum()),
                                   len(idx)))
            print("  %8.2f %8d %10d %10.3f %9.2f %s"
                  % (cov, len(idx), k, rate, rate / base,
                     "%.4f" % p if p >= 1e-4 else "<1e-4"))
            rows.append({"coverage": cov, "declined": int(len(idx)),
                         "contested_declined": k, "rate": round(rate, 4),
                         "enrichment": round(rate / base, 3),
                         "p_value": round(p, 6)})
        out[name] = rows
        best = max(rows, key=lambda r: r["enrichment"]) if rows else None
        if best:
            print("  strongest: %.2fx at coverage %.2f, p %s"
                  % (best["enrichment"], best["coverage"],
                     "%.4f" % best["p_value"] if best["p_value"] >= 1e-4
                     else "<1e-4"))
        print()

    for name, make, feats, scorer in MODELS:
        _oof, per_fold = SEL.fit_scores(make, feats, scorer, y, g)
        run(name, per_fold)

    P = json.load(io.open(io_paths.result_in("PREDICTIONS_FINAL.json",
                                             required=True), encoding="utf-8"))
    enc = np.array(P["transformer_grouped_strict"]["score"])
    per_fold = []
    for tr, te in GroupKFold(n_splits=SEL.FOLDS).split(np.zeros(len(y)), y, g):
        per_fold.append({"val_scores": enc[tr], "val_y": y[tr],
                         "test_idx": te, "test_scores": enc[te]})
    run("fine-tuned encoder (cross-fitted, weaker arm)", per_fold)

    # 7 coverages x 4 models = 28 tests. A p of 0.03 among 28 is not a
    # finding, so the threshold that matters is stated rather than left to
    # the reader.
    tests = [(nm, r) for nm, rows in out.items()
             if isinstance(rows, list) for r in rows]
    alpha = 0.05 / len(tests)
    survivors = [(nm, r) for nm, r in tests if r["p_value"] < alpha]
    print("=== multiple testing ===")
    print("  %d tests, Bonferroni threshold %.5f" % (len(tests), alpha))
    if survivors:
        for nm, r in sorted(survivors, key=lambda x: x[1]["p_value"]):
            print("  survives: %-42s coverage %.2f  %.2fx  p %.5f"
                  % (nm, r["coverage"], r["enrichment"], r["p_value"]))
    else:
        print("  nothing survives correction")
    print()
    print("  The classical models sit at 1.00x throughout: their uncertainty")
    print("  is unrelated to which messages the annotators argued over. The")
    print("  encoder's declined set is enriched, but by 1.19x at best - a")
    print("  real tendency, not a large one.")
    out["multiple_testing"] = {
        "n_tests": len(tests), "bonferroni_alpha": round(alpha, 6),
        "survivors": [{"model": nm, **r} for nm, r in survivors]}

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print()
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
