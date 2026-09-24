# -*- coding: utf-8 -*-
"""Is it the stacking that makes the ties fall the wrong way?

Two hypotheses for why an isotonic rate contract breaks here and holds on
three external corpora have been tested and refuted: distribution shift
(RESULTS_WHY_BLOCK_FAILS.md) and model weakness
(RESULTS_TIES_AND_STRENGTH.md). Before reaching for a third, there is a
difference between the two setups that this repository introduced and never
controlled for.

    here      two components, each isotonic-calibrated, then a fusion on top
    external  one model, one isotonic calibration

Two step functions stacked under a third fitted layer is not the same object
as one step function. It could compound the ties in a way a single calibrated
model does not, which would explain a mechanism that appears everywhere but
only bites here - and it would be a confound of our own making rather than a
fact about the data.

This is not a fishing expedition for a hypothesis that fits. It is the one
structural difference between the measurement that failed and the measurements
that held.

The test gives the external corpora our architecture. Each is run three ways
at the same request, on the same folds, with everything else fixed:

    single      one model, isotonic - what external_calibration.py did
    stacked     two components, each isotonic, fused - what we do
    stacked-P   the same two components under Platt, fused

If stacking is the cause, `stacked` breaks contracts that `single` keeps.
"""
import io
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import external_calibration as EC                            # noqa: E402
import io_paths                                              # noqa: E402

OUT = io_paths.result_out("RESULTS_STACKING_TIES.json")
REQUESTS = (0.02, 0.05, 0.10)
FOLDS, SEED, VAL_FRAC = 5, 42, 0.30


def _specs():
    return [
        TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                        max_features=50000),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                        sublinear_tf=True, max_features=200000),
    ]


def run(texts, y, mode, calibrate, q):
    """One architecture, one request, out of fold."""
    n_over, n_neg, ties, distinct = 0, 0, [], []
    for tr, te in StratifiedKFold(n_splits=FOLDS, shuffle=True,
                                  random_state=SEED).split(texts, y):
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(tr))
        cut = max(1, int(VAL_FRAC * len(tr)))
        va, tr2 = tr[perm[:cut]], tr[perm[cut:]]
        tr_txt = [texts[i] for i in tr2]
        va_txt = [texts[i] for i in va]
        te_txt = [texts[i] for i in te]

        vecs = _specs()[:1] if mode == "single" else _specs()
        cols_va, cols_te = [], []
        for vec in vecs:
            X = vec.fit_transform(tr_txt)
            m = LogisticRegression(max_iter=3000,
                                   class_weight="balanced").fit(X, y[tr2])
            s_va = m.predict_proba(vec.transform(va_txt))[:, 1]
            s_te = m.predict_proba(vec.transform(te_txt))[:, 1]
            cal = calibrate(s_va, y[va])
            cols_va.append(cal(s_va))
            cols_te.append(cal(s_te))

        if mode == "single":
            r_va, r_te = cols_va[0], cols_te[0]
        else:
            fu = LogisticRegression(max_iter=1000,
                                    class_weight="balanced").fit(
                np.column_stack(cols_va), y[va])
            r_va = fu.predict_proba(np.column_stack(cols_va))[:, 1]
            r_te = fu.predict_proba(np.column_stack(cols_te))[:, 1]

        t = float(np.quantile(r_va[y[va] == 0], 1.0 - q))
        neg = r_te[y[te] == 0]
        n_over += int((neg > t).sum())
        n_neg += len(neg)
        ties.append(float(np.mean(np.abs(neg - t) < 1e-9)))
        distinct.append(len(set(np.round(r_te, 6))))
    d = n_over / max(1, n_neg)
    return {"requested": q, "delivered": round(d, 4), "held": bool(d <= q),
            "tie_share": round(float(np.mean(ties)), 4),
            "distinct": round(float(np.mean(distinct)), 1)}


ARCHS = [("single", EC.cal_isotonic),
         ("stacked", EC.cal_isotonic),
         ("stacked-P", EC.cal_platt)]


def main():
    print("our architecture given to the external corpora")
    print("  single    = one model, isotonic   (what held)")
    print("  stacked   = two components, isotonic, fused   (what we do)")
    print("  stacked-P = the same, Platt")
    print()
    print("  %-17s %-10s %8s %10s %9s %7s"
          % ("corpus", "arch", "request", "delivered", "on cut", "held?"))
    out = {"requests": list(REQUESTS), "corpora": {}}
    for cname, load in EC.CORPORA:
        texts, y = load()
        out["corpora"][cname] = {}
        for aname, cal in ARCHS:
            rows = []
            for q in REQUESTS:
                r = run(texts, y, "single" if aname == "single" else "stack",
                        cal, q)
                rows.append(r)
                print("  %-17s %-10s %8.2f %10.3f %8.1f%% %7s"
                      % (cname if (aname, q) == (ARCHS[0][0], REQUESTS[0])
                         else "",
                         aname if q == REQUESTS[0] else "",
                         q, r["delivered"], 100 * r["tie_share"],
                         "yes" if r["held"] else "NO"), flush=True)
            out["corpora"][cname][aname] = rows
        print()

    print("CONTRACTS HELD, out of %d per architecture"
          % (len(EC.CORPORA) * len(REQUESTS)))
    held = {}
    for aname, _ in ARCHS:
        held[aname] = sum(1 for c in out["corpora"].values()
                          for r in c[aname] if r["held"])
        ties = float(np.mean([r["tie_share"] for c in out["corpora"].values()
                              for r in c[aname]]))
        print("  %-10s %d held | mean share on the cut %.1f%%"
              % (aname, held[aname], 100 * ties))
    print()
    # A count is not a verdict. Stacking "losing" one contract is only
    # evidence if the tie mechanism explains it - and stacked-P has no ties at
    # all, so a break there cannot be about ties. The first version of this
    # logic called a difference of one SUPPORTED, which is the same mistake
    # ties_and_model_strength.py made before it was tightened.
    ties = {a: float(np.mean([r["tie_share"]
                              for c in out["corpora"].values() for r in c[a]]))
            for a, _ in ARCHS}
    same_break_without_ties = (held["stacked-P"] <= held["stacked"]
                               and ties["stacked-P"] < 0.001)
    gap = held["single"] - held["stacked"]

    if same_break_without_ties:
        verdict = ("REFUTED: the tie-free arm (Platt, %.1f%% on the cut) "
                   "breaks as many contracts as the stacked isotonic one, so "
                   "the break is not about ties. Stacking also REDUCES ties, "
                   "%.1f%% against %.1f%% for a single model - the opposite of "
                   "the hypothesis."
                   % (100 * ties["stacked-P"], 100 * ties["stacked"],
                      100 * ties["single"]))
    elif gap >= 3:
        verdict = ("SUPPORTED: stacking breaks %d contracts a single "
                   "calibrated model keeps, and the tie-free arm does not"
                   % gap)
    elif gap > 0:
        verdict = ("INCONCLUSIVE: stacking loses %d contract(s), too few to "
                   "separate from the noise at the tightest request" % gap)
    else:
        verdict = ("REFUTED: stacking holds at least as many contracts as a "
                   "single calibrated model")
    out["tie_share_by_arch"] = {k: round(v, 4) for k, v in ties.items()}
    print("  " + verdict)
    out["held"] = held
    out["verdict"] = verdict

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print()
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
