# -*- coding: utf-8 -*-
"""Four additions that make the paper's central claim quantitative.

The paper currently argues that leakage inflates reported accuracy and shows
one contrast: a random message split against a thread-disjoint one. That is a
single number and it bundles every contamination route together.

This script separates the routes and measures each one, adds an empirical null
so the headline figure has a p-value with real power behind it, repeats the
cross-validation over many thread-to-fold assignments so the variance estimate
does not rest on five numbers, and puts an interval on the annotator agreement
reference that the paper leans on.

Everything here runs on CPU from the corpus already on disk. The classical
pipeline is copied from final_run.py unchanged so the numbers stay comparable
to the ones already published.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_paths                                          # noqa: E402

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_path                                       # noqa: E402

import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import (GroupKFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.preprocessing import normalize

# Resolved, not hardcoded: $ENRON_DIR, then the usual places, then the
# author's original path. scripts/verify_corpus.py puts it where this
# finds it. See corpus_path.py.
BASE = corpus_path.resolve()
OUT_JSON = io_paths.result_out("RESULTS_strengthen.json")
SENS = {(4, 10), (3, 10), (2, 8), (3, 5), (3, 4), (1, 5), (1, 2)}
EMPTY = {(1, 7), (1, 8)}
WS = re.compile(r"\s+")
SUBJ = re.compile(r"^\s*((re|fw|fwd|aw)\s*:\s*)+", re.I)

FOLDS, SEED = 5, 42
N_PERM = 500          # permutations for the empirical null
N_REPEAT = 20         # distinct thread-to-fold assignments
N_BOOT = 2000         # bootstrap resamples for the agreement interval
NEAR_DUP = 0.90       # cosine similarity at which two messages are near-copies

ENV = {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
       "numpy": np.__version__}


def build(dedup=True):
    """The dataset builder from final_run.py, with exact dedup made optional.

    Keeping the switch here is the whole point: the cost of a control can only
    be measured against a build that does not apply it.
    """
    rows, seen = [], set()
    for cf in sorted(glob.glob(os.path.join(BASE, "*", "*.cats"))):
        cats = [tuple(map(int, l.strip().split(",")))
                for l in open(cf) if l.strip().count(",") == 2]
        if any((t, s) in EMPTY for t, s, _ in cats):
            continue
        raw = open(cf.replace(".cats", ".txt"), encoding="utf-8",
                   errors="replace").read()
        i = raw.find("X-FileName:")
        body = (raw[raw.find("\n", i) + 1:] if i >= 0 else raw).strip()
        if len(body.split()) < 30:
            continue
        h = hashlib.md5(WS.sub(" ", body.lower()).strip().encode()).hexdigest()
        if dedup:
            if h in seen:
                continue
            seen.add(h)
        agreed = any((t, s) in SENS and f >= 2 for t, s, f in cats)
        either = any((t, s) in SENS for t, s, f in cats)
        # A tab or space only: \s* crosses the newline, so on a message
        # whose Subject header is empty it captures the next header
        # instead. 43 messages here, 34 of which then shared one false
        # thread key.
        m = re.search(r"^Subject:[ \t]*(.*)$", raw, re.M | re.I)
        subj = m.group(1).strip() if m else ""
        rows.append({"label": int(agreed), "label_either": int(either),
                     "agree": ("both" if agreed
                               else "one" if either else "neither"),
                     "thread_key": WS.sub(" ", SUBJ.sub("", subj).strip().lower())
                                   or ("m:" + h[:10]),
                     "text": body})
    return pd.DataFrame(rows)


def vectorise(X):
    return TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                           max_features=50000).fit_transform(X)


def near_dup_keep(X):
    """Keep one message from every near-duplicate cluster.

    Cosine on L2-normalised tf-idf, greedy single pass; the first message of a
    cluster survives. Exact duplicates are a special case, so a build that has
    already been deduplicated still loses messages here.
    """
    V = normalize(vectorise(X))
    S = (V @ V.T).toarray()
    np.fill_diagonal(S, 0.0)
    keep = np.ones(len(X), dtype=bool)
    dropped = 0
    for i in range(len(X)):
        if not keep[i]:
            continue
        for j in np.where(S[i] >= NEAR_DUP)[0]:
            if j > i and keep[j]:
                keep[j] = False
                dropped += 1
    return keep, dropped


def run_cv(X, y, g, split, seed=None, Xv=None):
    """One cross-validated LogReg run. Returns pooled F1/MCC and fold scores.

    A grouped run with seed=None uses the same deterministic GroupKFold the
    published results used, so the ladder and the null stay comparable to them.
    A seed selects a different thread-to-fold assignment instead, which is what
    the repeated cross-validation varies.
    """
    if Xv is None:
        Xv = vectorise(X)
    if split == "grouped":
        cv = (GroupKFold(n_splits=FOLDS) if seed is None else
              StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed))
        splits = list(cv.split(X, y, g))
    else:
        seed = SEED if seed is None else seed
        cv = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
        splits = list(cv.split(X, y))
    pred = np.zeros(len(y), dtype=int)
    ff = []
    for tr, te in splits:
        if split == "grouped":
            assert not (set(g[tr]) & set(g[te])), "thread leaked"
        m = LogisticRegression(max_iter=2000,
                               class_weight="balanced").fit(Xv[tr], y[tr])
        p = m.predict(Xv[te])
        pred[te] = p
        ff.append(float(f1_score(y[te], p, zero_division=0)))
    return {"f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(y, pred)), 4),
            "fold_mean_f1": round(float(np.mean(ff)), 4),
            "fold_std_f1": round(float(np.std(ff)), 4),
            "n": int(len(y)),
            "positives": int(y.sum())}, pred


def trivial_f1(y):
    b = float(np.mean(y))
    return round(2 * b / (1 + b), 4)


# ---------------------------------------------------------------- 1. ladder
def ladder(lab):
    """Add one contamination control at a time and record what each costs."""
    out = []

    raw = build(dedup=False)
    ded = build(dedup=True)

    def col(d):
        return (d.text.tolist(),
                (d.label if lab == "strict" else d.label_either).values,
                d.thread_key.values)

    # R0  no controls at all: duplicates in, random message split
    X, y, g = col(raw)
    r, _ = run_cv(X, y, g, "stratified")
    r.update(rung="R0", control="none (duplicates kept, random split)")
    out.append(r)

    # R1  exact duplicates removed, still a random message split
    X, y, g = col(ded)
    r, _ = run_cv(X, y, g, "stratified")
    r.update(rung="R1", control="+ exact deduplication")
    out.append(r)

    # R2  near-duplicates removed as well
    keep, dropped = near_dup_keep(X)
    nd = ded[keep].reset_index(drop=True)
    X2, y2, g2 = col(nd)
    r, _ = run_cv(X2, y2, g2, "stratified")
    r.update(rung="R2", control="+ near-duplicate removal",
             near_duplicates_dropped=int(dropped))
    out.append(r)

    # R3  and finally the split is made thread-disjoint: the full protocol
    r, pred = run_cv(X2, y2, g2, "grouped")
    r.update(rung="R3", control="+ thread-disjoint split (full protocol)")
    out.append(r)

    base = out[0]["f1"]
    for r in out:
        r["drop_from_R0"] = round(base - r["f1"], 4)
        r["trivial_floor"] = trivial_f1(np.array(
            [1] * r["positives"] + [0] * (r["n"] - r["positives"])))
    return out, (X2, y2, g2)


# ------------------------------------------------------- 2. permutation null
def permutation_null(X, y, g, observed):
    """What F1 does this pipeline reach when the labels carry no information?

    Ojala and Garriga's label-permutation test: the class proportion is held
    fixed and the association between text and label is destroyed. It answers a
    question five fold scores cannot - the smallest p a Wilcoxon test over five
    folds can return is 0.0625, which is above every conventional threshold.
    """
    rng = np.random.RandomState(SEED)
    Xv = vectorise(X)
    null = []
    for i in range(N_PERM):
        yp = rng.permutation(y)
        r, _ = run_cv(X, yp, g, "grouped", Xv=Xv)
        null.append(r["f1"])
        if (i + 1) % 100 == 0:
            print("    permutation %d/%d" % (i + 1, N_PERM), flush=True)
    null = np.array(null)
    p = (1.0 + float((null >= observed).sum())) / (N_PERM + 1.0)
    return {"observed_f1": observed,
            "null_mean_f1": round(float(null.mean()), 4),
            "null_std_f1": round(float(null.std()), 4),
            "null_p95_f1": round(float(np.percentile(null, 95)), 4),
            "null_max_f1": round(float(null.max()), 4),
            "permutations": N_PERM,
            "p_value": round(p, 5),
            "note": "one-sided; p = (1 + #{null >= observed}) / (N + 1)"}


# --------------------------------------------------------- 3. repeated CV
def repeated_cv(X, y, g):
    """Re-partition the threads many times instead of trusting one partition.

    GroupKFold is deterministic, so the published five fold scores describe a
    single assignment of threads to folds. Varying the assignment separates the
    variance of the method from the luck of one partition.
    """
    fold_scores, run_scores = [], []
    Xv = vectorise(X)
    for s in range(N_REPEAT):
        r, _ = run_cv(X, y, g, "grouped", seed=SEED + s, Xv=Xv)
        run_scores.append(r["fold_mean_f1"])
        fold_scores.append(r["f1"])
    a = np.array(run_scores)
    b = np.array(fold_scores)
    return {"repeats": N_REPEAT,
            "fold_mean_f1_mean": round(float(a.mean()), 4),
            "fold_mean_f1_std": round(float(a.std()), 4),
            "fold_mean_f1_range": [round(float(a.min()), 4),
                                   round(float(a.max()), 4)],
            "pooled_f1_mean": round(float(b.mean()), 4),
            "pooled_f1_std": round(float(b.std()), 4),
            "pooled_f1_range": [round(float(b.min()), 4),
                                round(float(b.max()), 4)],
            "note": "each repeat is a different thread-to-fold assignment"}


# ------------------------------------------ 4. interval on the agreement point
def agreement_interval(df):
    """A confidence interval for the annotator agreement reference.

    The point estimate treats one annotator as a classifier scored against the
    other. The counts are symmetric, so the messages only one annotator marked
    split evenly into false positives and false negatives. Resampling is over
    threads, for the same reason the model intervals are.
    """
    def score(sub):
        both = int((sub == "both").sum())
        one = int((sub == "one").sum())
        if both == 0:
            return None, None
        f1 = 2.0 * both / (2.0 * both + one)
        neither = int((sub == "neither").sum())
        n = both + one + neither
        po = (both + neither) / float(n)
        pa = (both + one / 2.0) / float(n)
        pe = pa * pa + (1 - pa) * (1 - pa)
        kap = (po - pe) / (1 - pe) if pe < 1 else None
        return f1, kap

    a = df.agree.values
    g = df.thread_key.values
    f1_hat, k_hat = score(pd.Series(a))

    rng = np.random.RandomState(SEED)
    uniq = np.array(sorted(set(g)))
    by = {t: np.where(g == t)[0] for t in uniq}
    f1s, ks = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([by[t] for t in rng.choice(uniq, len(uniq), True)])
        f, k = score(pd.Series(a[idx]))
        if f is not None:
            f1s.append(f)
        if k is not None:
            ks.append(k)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4),
                   round(float(np.percentile(v, 97.5)), 4)]
    return {"agreement_f1": round(float(f1_hat), 4),
            "agreement_f1_ci95": q(f1s),
            "cohens_kappa": round(float(k_hat), 4),
            "cohens_kappa_ci95": q(ks),
            "resamples": len(f1s),
            "note": "thread-level bootstrap; one annotator scored against the other"}


def main():
    print("=== environment ===")
    for k, v in ENV.items():
        print("  %-14s %s" % (k, v))
    print()

    OUT = {"environment": ENV,
           "purpose": "quantitative strengthening of the leakage claim"}

    for lab in ("strict", "broad"):
        print("=== leakage ladder, %s labels ===" % lab, flush=True)
        rungs, full = ladder(lab)
        for r in rungs:
            print("  %-3s %-45s n=%4d  F1 %.4f  (floor %.4f, drop %+0.4f)"
                  % (r["rung"], r["control"], r["n"], r["f1"],
                     r["trivial_floor"], -r["drop_from_R0"]), flush=True)
        OUT["ladder_" + lab] = rungs

        X, y, g = full
        obs = rungs[-1]["f1"]

        print("  repeated cross-validation (%d partitions)..." % N_REPEAT,
              flush=True)
        OUT["repeated_cv_" + lab] = repeated_cv(X, y, g)
        rc = OUT["repeated_cv_" + lab]
        print("    pooled F1 %.4f +/- %.4f  range [%.4f, %.4f]"
              % (rc["pooled_f1_mean"], rc["pooled_f1_std"],
                 rc["pooled_f1_range"][0], rc["pooled_f1_range"][1]),
              flush=True)

        print("  permutation null (%d permutations)..." % N_PERM, flush=True)
        OUT["permutation_" + lab] = permutation_null(X, y, g, obs)
        pm = OUT["permutation_" + lab]
        print("    observed %.4f | null %.4f +/- %.4f | max %.4f | p = %.5f"
              % (pm["observed_f1"], pm["null_mean_f1"], pm["null_std_f1"],
                 pm["null_max_f1"], pm["p_value"]), flush=True)
        print()

    print("=== annotator agreement interval ===", flush=True)
    OUT["agreement"] = agreement_interval(build(dedup=True))
    ag = OUT["agreement"]
    print("  F1 %.4f  CI95 [%.4f, %.4f]"
          % (ag["agreement_f1"], ag["agreement_f1_ci95"][0],
             ag["agreement_f1_ci95"][1]))
    print("  kappa %.4f  CI95 [%.4f, %.4f]"
          % (ag["cohens_kappa"], ag["cohens_kappa_ci95"][0],
             ag["cohens_kappa_ci95"][1]))
    print()

    json.dump(OUT, open(OUT_JSON, "w"), indent=2)
    print("written %s" % OUT_JSON)


if __name__ == "__main__":
    main()
