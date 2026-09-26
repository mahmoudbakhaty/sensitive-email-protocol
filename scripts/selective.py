# -*- coding: utf-8 -*-
"""Selective prediction: what a model is worth when it is allowed to decline.

Every method in this paper lands between F1 0.35 and 0.41 on the strict labels
with overlapping intervals, which is why the paper ranks none of them. That
comparison asks one question - how often is it right when forced to answer
about everything - and a deployed filter is never in that position. It can
decide automatically where it is confident and route the rest to a person.

Under that question every method improves, and by more than any modelling
difference this paper measures. Ranking messages by the model's own confidence
and scoring only the share it keeps, the fine-tuned encoder climbs from F1
0.397 at full coverage to 0.559 at 36%, a gain of 0.162; the classical models
gain between 0.090 and 0.107. The intervals still overlap, so this ranks
nothing either - the claim is that abstention is worth more than the choice of
model, not that one model is better.

A DISCARDED VERSION OF THIS ANALYSIS SAID THE OPPOSITE. It used a single
global threshold recovered from the released predictions instead of per-fold
validation-selected ones, and under it logistic regression and the linear SVM
fell BELOW the random band, which looked like a finding: that only the encoder
has usable confidence. It was an artefact of ranking by distance from a
threshold that was not the model's own - logistic regression's recovered
global threshold was 0.4999 while its per-fold validation thresholds are not
0.5. The conclusion it suggested is not supported.

Two things make what remains a measurement rather than an artefact.

THE ABSTENTION RULE IS CHOSEN ON VALIDATION, NEVER ON TEST. Declining messages
changes the set the score is computed on, so a curve drawn by choosing the
coverage that looks best on test would be meaningless. Each fold carves 30% of
its training threads off, fits there, and takes both the decision threshold
and the confidence cut-off from that split. The cut-off is an absolute
confidence value, so the coverage actually achieved on test is whatever it is
and is reported, not the target.

RANDOM ABSTENTION IS THE CONTROL. Dropping 60% of the messages changes F1 by
arithmetic alone. Every coverage is therefore run again with the same number
of messages declined at random, 200 times, and the real number is reported as
a z against that band. A gain that does not clear the band is not a gain.

The three classical models are retrained here, so nothing rests on a released
file. The encoder cannot be retrained without a GPU, so its curve is
cross-fitted from the released out-of-fold scores: for each test fold the
confidence cut-off comes from the other folds' scores, which were produced by
models that never saw it. That is weaker than the retrained arms and is
labelled as such in the output.
"""
import io
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import io_paths                                              # noqa: E402
import strengthen as S                                       # noqa: E402

OUT = io_paths.result_out("RESULTS_SELECTIVE.json")
FOLDS, SEED, VAL_FRAC = 5, 42, 0.30
N_RAND, N_BOOT = 200, 2000
COVERAGES = (1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30)


def fit_scores(make, feats, scorer, y, g):
    """Out-of-fold scores, and per-fold the validation scores beside them.

    The validation scores are what the abstention rule is allowed to see."""
    oof = np.zeros(len(y))
    per_fold = []
    for tr, te in GroupKFold(n_splits=FOLDS).split(np.zeros(len(y)), y, g):
        assert not (set(g[tr]) & set(g[te])), "thread leaked into test"
        rng = np.random.RandomState(SEED)
        th = np.array(sorted(set(g[tr])))
        rng.shuffle(th)
        va_th = set(th[:max(1, int(VAL_FRAC * len(th)))])
        va = np.array([i for i in tr if g[i] in va_th])
        tr2 = np.array([i for i in tr if g[i] not in va_th])
        m = make().fit(feats[tr2], y[tr2])
        sv, st = scorer(m, feats[va]), scorer(m, feats[te])
        oof[te] = st
        per_fold.append({"val_scores": sv, "val_y": y[va],
                         "test_idx": te, "test_scores": st})
    return oof, per_fold


def pick_threshold(y, s):
    """The decision threshold, on this model's own score scale."""
    grid = (np.arange(0.05, 0.96, 0.01) if s.min() >= 0 and s.max() <= 1
            else np.unique(np.percentile(s, np.arange(2, 99, 1))))
    return float(max(grid, key=lambda t: f1_score(
        y, (s >= t).astype(int), zero_division=0)))


def conf_cutoff(val_scores, thr, coverage):
    """The confidence below which we decline, set to hit `coverage` on the
    VALIDATION split. Applied to test as an absolute value."""
    conf = np.abs(val_scores - thr)
    if coverage >= 1.0:
        return -1.0
    return float(np.quantile(conf, 1.0 - coverage))


def selective(per_fold, y, g):
    """F1 and MCC at each target coverage, plus the coverage actually reached."""
    rows = []
    for cov in COVERAGES:
        keep_idx, preds = [], []
        for f in per_fold:
            thr = pick_threshold(f["val_y"], f["val_scores"])
            cut = conf_cutoff(f["val_scores"], thr, cov)
            conf = np.abs(f["test_scores"] - thr)
            sel = conf >= cut
            keep_idx.append(f["test_idx"][sel])
            preds.append((f["test_scores"][sel] >= thr).astype(int))
        idx = np.concatenate(keep_idx)
        p = np.concatenate(preds)
        yk = y[idx]
        if len(set(yk)) < 2:
            continue
        rows.append({"target_coverage": cov,
                     "actual_coverage": round(float(len(idx) / len(y)), 4),
                     "n": int(len(idx)),
                     "positives": int(yk.sum()),
                     "f1": round(float(f1_score(yk, p, zero_division=0)), 4),
                     "mcc": round(float(matthews_corrcoef(yk, p)), 4),
                     "_idx": idx, "_pred": p})
    return rows


def random_band(rows, per_fold, y, rng):
    """The same number of messages declined at random, N_RAND times."""
    thrs = [pick_threshold(f["val_y"], f["val_scores"]) for f in per_fold]
    for r in rows:
        draws = []
        for _ in range(N_RAND):
            idx, preds = [], []
            for f, thr in zip(per_fold, thrs):
                n = int(round(r["actual_coverage"] * len(f["test_idx"])))
                pick = rng.permutation(len(f["test_idx"]))[:n]
                idx.append(f["test_idx"][pick])
                preds.append((f["test_scores"][pick] >= thr).astype(int))
            i2, p2 = np.concatenate(idx), np.concatenate(preds)
            if len(set(y[i2])) < 2:
                continue
            draws.append(f1_score(y[i2], p2, zero_division=0))
        m, sd = float(np.mean(draws)), float(np.std(draws))
        r["random_mean"] = round(m, 4)
        r["random_sd"] = round(sd, 4)
        r["z"] = round((r["f1"] - m) / sd, 2) if sd > 1e-9 else 0.0
    return rows


def thread_ci(rows, y, g, rng):
    """Cluster bootstrap over threads, on the selected subset."""
    for r in rows:
        idx, pred = r["_idx"], r["_pred"]
        gg = g[idx]
        by = {}
        for pos, t in enumerate(gg):
            by.setdefault(t, []).append(pos)
        uniq = np.array(sorted(by))
        f1s = []
        for _ in range(N_BOOT):
            take = np.concatenate([by[t] for t in rng.choice(uniq, len(uniq),
                                                             True)])
            yy = y[idx][take]
            if len(set(yy)) < 2:
                continue
            f1s.append(f1_score(yy, pred[take], zero_division=0))
        r["f1_ci95"] = [round(float(np.percentile(f1s, 2.5)), 4),
                        round(float(np.percentile(f1s, 97.5)), 4)]
    return rows


def aurc(rows):
    """Area under the risk-coverage curve, risk = 1 - F1. Lower is better."""
    pts = sorted((r["actual_coverage"], 1.0 - r["f1"]) for r in rows)
    x = [p[0] for p in pts]
    z = [p[1] for p in pts]
    return round(float(np.trapezoid(z, x) / (max(x) - min(x))), 4)


def report(name, rows, note=""):
    print("=== %s ===%s" % (name, ("  " + note) if note else ""))
    print("  %8s %8s %8s %8s %18s %8s %6s"
          % ("target", "actual", "F1", "MCC", "95% CI", "random", "z"))
    for r in rows:
        print("  %8.2f %8.3f %8.4f %8.4f %18s %8.4f %6.1f"
              % (r["target_coverage"], r["actual_coverage"], r["f1"],
                 r["mcc"], "[%.3f, %.3f]" % tuple(r["f1_ci95"]),
                 r["random_mean"], r["z"]))
    print("  AURC (risk = 1 - F1, lower is better): %.4f" % aurc(rows))
    print()


def strip(rows):
    return [{k: v for k, v in r.items() if not k.startswith("_")}
            for r in rows]


def main():
    df = S.build(dedup=True)
    X, g, y = df.text.tolist(), df.thread_key.values, df.label.values
    print("messages %d | threads %d | strict positives %d"
          % (len(df), df.thread_key.nunique(), y.sum()))
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
        ("linear SVM", lambda: LinearSVC(C=0.5, class_weight="balanced",
                                      random_state=42),
         word, lambda m, f: m.decision_function(f)),
        ("character n-grams",
         lambda: LogisticRegression(max_iter=3000, class_weight="balanced"),
         char, lambda m, f: m.predict_proba(f)[:, 1]),
    ]

    out = {"environment": S.ENV, "protocol": __doc__.strip().splitlines()[0],
           "coverages": list(COVERAGES), "n_random": N_RAND,
           "n_bootstrap": N_BOOT, "val_frac": VAL_FRAC, "folds": FOLDS,
           "dataset": {"messages": int(len(df)),
                       "threads": int(df.thread_key.nunique())}}

    for name, make, feats, scorer in MODELS:
        rng = np.random.RandomState(SEED)
        _oof, per_fold = fit_scores(make, feats, scorer, y, g)
        rows = selective(per_fold, y, g)
        rows = random_band(rows, per_fold, y, rng)
        rows = thread_ci(rows, y, g, rng)
        report(name, rows, "(retrained here)")
        out[name] = {"arm": "retrained", "rows": strip(rows),
                     "aurc": aurc(rows)}

    # The encoder, from the released out-of-fold scores. Cross-fitted: the
    # cut-off for each fold comes from the other folds, whose scores were
    # produced by models that never saw it.
    P = json.load(io.open(io_paths.result_in("PREDICTIONS_FINAL.json",
                                             required=True), encoding="utf-8"))
    enc = np.array(P["transformer_grouped_strict"]["score"])
    per_fold = []
    for tr, te in GroupKFold(n_splits=FOLDS).split(np.zeros(len(y)), y, g):
        per_fold.append({"val_scores": enc[tr], "val_y": y[tr],
                         "test_idx": te, "test_scores": enc[te]})
    rng = np.random.RandomState(SEED)
    rows = selective(per_fold, y, g)
    rows = random_band(rows, per_fold, y, rng)
    rows = thread_ci(rows, y, g, rng)
    report("fine-tuned encoder", rows,
           "(cross-fitted from released scores - weaker arm)")
    out["fine-tuned encoder"] = {"arm": "cross-fitted from released OOF scores",
                                 "caveat": "not retrained; needs a GPU to "
                                           "match the other arms",
                                 "rows": strip(rows), "aurc": aurc(rows)}

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
