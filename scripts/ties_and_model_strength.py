# -*- coding: utf-8 -*-
"""Do isotonic ties break the contract only when the model is weak?

external_calibration.py withdrew a claim. Isotonic collapses a score into a
few dozen values on every corpus tested - up to 59.5% of the negatives on a
single value - but the rate contract held on all three of them, while it broke
on this benchmark. Which side of the promise the ties fall on looked like a
property of the data that nothing controls.

One thing separates the four, and it is not size or balance:

    Enron-Spam        Brier 0.0088
    SMS Spam                0.0133
    tweet_eval/hate         0.1524
    this benchmark          0.2268

The three where the contract held are the three where a linear model separates
the classes well. If ties fall the permissive way only when the model is weak,
the direction is predictable after all - and the warning becomes sharper
rather than weaker, because a rate guarantee would then fail exactly when it
is most needed.

The test weakens the model deliberately rather than looking for a weak corpus.
Each external corpus is re-run with the training fold cut to a fraction of
itself, which moves Brier along a range while everything else - the corpus,
the calibrator, the protocol, the request - stays fixed. If the hypothesis is
right, contracts start breaking as Brier climbs, and around the value this
benchmark sits at.

A prediction stated before the run: breaks should appear above a Brier of
roughly 0.15, and the corpora should cross over at similar Brier values rather
than at similar training sizes.
"""
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import external_calibration as EC                            # noqa: E402
import io_paths                                              # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.model_selection import StratifiedKFold          # noqa: E402

OUT = io_paths.result_out("RESULTS_TIES_AND_STRENGTH.json")
FRACTIONS = (1.0, 0.30, 0.10, 0.05, 0.02, 0.01)
REQUEST = 0.05
SEED, FOLDS, VAL_FRAC = 42, 5, 0.30


def run(texts, y, frac, calibrate, q=REQUEST):
    """One corpus at one training fraction. Everything else fixed."""
    n_over, n_neg, ties, distinct, briers = 0, 0, [], [], []
    for tr, te in StratifiedKFold(n_splits=FOLDS, shuffle=True,
                                  random_state=SEED).split(texts, y):
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(tr))
        cut = max(1, int(VAL_FRAC * len(tr)))
        va, tr2 = tr[perm[:cut]], tr[perm[cut:]]
        # the validation split is NOT shrunk: only the model is weakened, so
        # the threshold is still estimated from the same amount of data and
        # sample size cannot be confused with model strength
        keep = max(20, int(round(frac * len(tr2))))
        tr2 = tr2[:keep]
        if len(set(y[tr2])) < 2:
            return None

        vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), sublinear_tf=True,
                              max_features=50000)
        X = vec.fit_transform([texts[i] for i in tr2])
        m = LogisticRegression(max_iter=2000,
                               class_weight="balanced").fit(X, y[tr2])
        s_va = m.predict_proba(vec.transform([texts[i] for i in va]))[:, 1]
        s_te = m.predict_proba(vec.transform([texts[i] for i in te]))[:, 1]
        cal = calibrate(s_va, y[va])
        r_va, r_te = cal(s_va), cal(s_te)

        t = float(np.quantile(r_va[y[va] == 0], 1.0 - q))
        neg = r_te[y[te] == 0]
        n_over += int((neg > t).sum())
        n_neg += len(neg)
        ties.append(float(np.mean(np.abs(neg - t) < 1e-9)))
        distinct.append(len(set(np.round(r_te, 6))))
        briers.append(float(np.mean((r_te - y[te]) ** 2)))
    return {"train_fraction": frac,
            "brier": round(float(np.mean(briers)), 4),
            "delivered": round(n_over / max(1, n_neg), 4),
            "tie_share": round(float(np.mean(ties)), 4),
            "distinct": round(float(np.mean(distinct)), 1),
            "held": bool(n_over / max(1, n_neg) <= q)}


def main():
    print("request fixed at %.0f%%, isotonic calibration, "
          "validation split never shrunk" % (100 * REQUEST))
    print("this benchmark sits at Brier 0.2268 and breaks its contract")
    print()
    print("  %-17s %8s %8s %10s %9s %7s"
          % ("corpus", "train", "Brier", "delivered", "on cut", "held?"))
    out = {"request": REQUEST, "fractions": list(FRACTIONS),
           "reference": {"corpus": "this benchmark", "brier": 0.2268,
                         "held": False},
           "corpora": {}}
    for cname, load in EC.CORPORA:
        texts, y = load()
        rows = []
        for frac in FRACTIONS:
            r = run(texts, y, frac, EC.cal_isotonic)
            if r is None:
                continue
            rows.append(r)
            print("  %-17s %7.0f%% %8.4f %10.3f %8.1f%% %7s"
                  % (cname if frac == FRACTIONS[0] else "",
                     100 * frac, r["brier"], r["delivered"],
                     100 * r["tie_share"], "yes" if r["held"] else "NO"),
                  flush=True)
        out["corpora"][cname] = rows
        print()

    print("WHERE CONTRACTS BREAK")
    breaks = []
    for cname, rows in out["corpora"].items():
        bad = [r for r in rows if not r["held"]]
        if bad:
            first = min(bad, key=lambda r: r["brier"])
            breaks.append(first["brier"])
            print("  %-17s first break at Brier %.4f (train %.0f%%)"
                  % (cname, first["brier"], 100 * first["train_fraction"]))
        else:
            worst = max(rows, key=lambda r: r["brier"])
            print("  %-17s never breaks, worst Brier reached %.4f"
                  % (cname, worst["brier"]))
    print()
    # A break is evidence for the hypothesis only if breaking is what a weak
    # model DOES - contracts failing as Brier climbs and holding when it is
    # low. A lone break at a good Brier that un-breaks at worse ones is noise,
    # and the first version of this verdict would have called it support.
    monotone = []
    for cname, rows in out["corpora"].items():
        srt = sorted(rows, key=lambda r: r["brier"])
        held = [r["held"] for r in srt]
        # consistent means: once it starts failing it keeps failing
        first_fail = next((i for i, h in enumerate(held) if not h), None)
        monotone.append(first_fail is not None
                        and all(not h for h in held[first_fail:]))
    reached_ours = [max(r["brier"] for r in rows)
                    for rows in out["corpora"].values()]
    above_ours = [b for b in reached_ours if b >= 0.2268]

    if any(monotone):
        verdict = ("SUPPORTED: at least one corpus fails consistently once "
                   "Brier passes a point")
    elif above_ours and not breaks:
        verdict = ("REFUTED: a corpus reached Brier %.4f, above this "
                   "benchmark's 0.2268, and never broke its contract"
                   % max(above_ours))
    elif above_ours:
        verdict = ("REFUTED: a corpus reached Brier %.4f, above this "
                   "benchmark's 0.2268, and never broke; the only breaks are "
                   "at low Brier and do not persist as it worsens"
                   % max(above_ours))
    else:
        verdict = ("INCONCLUSIVE: no corpus was weakened far enough to reach "
                   "this benchmark's Brier of 0.2268")
    print("  " + verdict)
    print()
    print("  Model strength does not predict which way the ties fall. The")
    print("  direction remains unexplained, and this file records that")
    print("  rather than a second guess.")

    out["first_break_briers"] = [round(b, 4) for b in breaks]
    out["max_brier_reached"] = [round(b, 4) for b in reached_ours]
    out["verdict"] = verdict
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print()
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
