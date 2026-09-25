# -*- coding: utf-8 -*-
"""Does the calibration finding hold outside this corpus?

RESULTS_WHY_BLOCK_FAILS.md reports that isotonic calibration makes a rate
guarantee impossible to honour: the calibrated score takes few distinct
values, a tenth of the negatives land on the threshold's exact value, and
"at most q above this cut" then has no solution. Platt scaling, strictly
monotone, holds the same contract.

That was measured on one corpus of 1,382 messages, which is not enough to call
it a property of isotonic calibration rather than of this data. It is a claim
about a calibrator and an order statistic, so it should hold anywhere - and if
it does not, the finding is about Enron and should be stated that way.

Three public datasets, chosen to vary on the things that could explain it away:

  * SMS Spam (UCI), 5,574 messages, 13.4% positive - closest in shape to the
    benchmark, four times the size.
  * tweet_eval/hate (CardiffNLP), 9,000 tweets, 42% positive - contested human
    judgement like ours, but nearly balanced.
  * Enron-Spam, 33,716 messages, 50.9% positive - balanced and twenty-four
    times the size.

None has thread structure, so plain stratified folds are used rather than the
grouped ones the benchmark needs. Everything else is the protocol: fit on the
training fold minus a 30% validation split, calibrate there, set the threshold
there, measure out of fold.

Reported per dataset and calibrator: distinct values, the share of negatives
sitting on the threshold's exact value, Brier score, and whether a 2%, 5% and
10% rate contract is delivered.
"""
import io
import json
import os
import sys
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import io_paths                                              # noqa: E402

OUT = io_paths.result_out("RESULTS_EXTERNAL_CALIBRATION.json")
DATA = os.environ.get("EXTERNAL_DATA", os.path.join(HERE, "_external"))
FOLDS, SEED, VAL_FRAC, ALPHA = 5, 42, 0.30, 0.10
REQUESTS = (0.02, 0.05, 0.10)


# ---- the three corpora, each returning (texts, labels) --------------------

# RESULTS_EXTERNAL_CALIBRATION.md told readers "the three corpora download from
# UCI, HuggingFace and a GitHub mirror". They did not: the loaders read local
# files and a clean clone died on FileNotFoundError before printing anything.
# In a release whose whole claim is that a reader can re-run it, a Reproduce
# block that cannot run is the worst kind of defect. They download now.
SOURCES = {
    "sms.zip":
        "https://archive.ics.uci.edu/static/public/228/"
        "sms+spam+collection.zip",
    "hate.parquet":
        "https://huggingface.co/datasets/cardiffnlp/tweet_eval/resolve/main/"
        "hate/train-00000-of-00001.parquet",
    "enron_spam.zip":
        "https://github.com/MWiechmann/enron_spam_data/raw/master/"
        "enron_spam_data.zip",
}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}


def ensure(name):
    """Local path for `name`, fetched on first use and fingerprinted.

    Not redistributed here - fetched from the source the write-up names, and
    the digest is printed so two readers can tell whether they have the same
    bytes."""
    import hashlib
    import urllib.request as UR
    if not os.path.isdir(DATA):
        os.makedirs(DATA)
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        url = SOURCES[name]
        sys.stderr.write("fetching %s ...\n" % name)
        io.open(p, "wb").write(
            UR.urlopen(UR.Request(url, headers=UA), timeout=600).read())
    sha = hashlib.sha256(io.open(p, "rb").read()).hexdigest()[:16]
    FETCHED[name] = sha
    return p


FETCHED = {}


def load_sms():
    z = zipfile.ZipFile(ensure("sms.zip"))
    name = [n for n in z.namelist() if "SMSSpam" in n][0]
    raw = z.read(name).decode("utf-8", "replace")
    rows = [l.split("\t", 1) for l in raw.splitlines() if "\t" in l]
    return ([r[1] for r in rows],
            np.array([1 if r[0] == "spam" else 0 for r in rows]))


def load_hate():
    import pandas as pd
    d = pd.read_parquet(ensure("hate.parquet"))
    return d.text.tolist(), d.label.values.astype(int)


def load_enron_spam():
    import pandas as pd
    z = zipfile.ZipFile(ensure("enron_spam.zip"))
    d = pd.read_csv(z.open(z.namelist()[0]))
    d = d.dropna(subset=["Message"])
    txt = (d["Subject"].fillna("") + " " + d["Message"]).tolist()
    return txt, (d["Spam/Ham"] == "spam").values.astype(int)


CORPORA = [("SMS Spam", load_sms),
           ("tweet_eval hate", load_hate),
           ("Enron-Spam", load_enron_spam)]


# ---- the two calibrators --------------------------------------------------

def cal_isotonic(s_va, y_va):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0,
                             y_max=1.0).fit(s_va, y_va)
    return iso.predict


def cal_platt(s_va, y_va):
    lr = LogisticRegression(max_iter=1000).fit(
        np.asarray(s_va).reshape(-1, 1), y_va)
    return lambda s: lr.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]


CALIBRATORS = [("isotonic", cal_isotonic), ("Platt", cal_platt)]


def evaluate(texts, y, calibrate, q):
    """Out-of-fold: the score's resolution, and whether the contract holds.

    The contract is the block side of filter_system.py stated plainly: set a
    cut so that at most q of the negatives sit above it, using the validation
    split, then see what share of negatives actually do out of fold."""
    n_over, n_neg, ties, distinct, briers = 0, 0, [], [], []
    for tr, te in StratifiedKFold(n_splits=FOLDS, shuffle=True,
                                  random_state=SEED).split(texts, y):
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(tr))
        cut = max(1, int(VAL_FRAC * len(tr)))
        va, tr2 = tr[perm[:cut]], tr[perm[cut:]]

        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                              max_features=50000)
        X = vec.fit_transform([texts[i] for i in tr2])
        m = LogisticRegression(max_iter=2000,
                               class_weight="balanced").fit(X, y[tr2])
        s_va = m.predict_proba(vec.transform([texts[i] for i in va]))[:, 1]
        s_te = m.predict_proba(vec.transform([texts[i] for i in te]))[:, 1]
        cal = calibrate(s_va, y[va])
        r_va, r_te = cal(s_va), cal(s_te)

        t = float(np.quantile(r_va[y[va] == 0], 1.0 - q))
        neg_te = r_te[y[te] == 0]
        # >= , not > . The system this replicates blocks at `r >= t_block`
        # (filter_system.actions), so a message sitting exactly ON the cut is
        # acted on. Counting it as compliant excluded precisely the cases this
        # script exists to measure: the tie share on the line below is large
        # under isotonic and ~0 under Platt, so the old comparison flattered
        # isotonic - the calibrator the published conclusion is against.
        n_over += int((neg_te >= t).sum())
        n_neg += len(neg_te)
        ties.append(float(np.mean(np.abs(neg_te - t) < 1e-9)))
        distinct.append(len(set(np.round(r_te, 6))))
        briers.append(float(np.mean((r_te - y[te]) ** 2)))
    return {"delivered": round(n_over / max(1, n_neg), 4),
            "tie_share": round(float(np.mean(ties)), 4),
            "distinct": round(float(np.mean(distinct)), 1),
            "brier": round(float(np.mean(briers)), 4)}


def main():
    out = {"note": "the isotonic tie finding, tested on three public corpora",
           "requests": list(REQUESTS), "alpha": ALPHA, "corpora": {}}
    print("%-17s %-9s %8s %9s %8s %8s %9s %9s"
          % ("corpus", "calib.", "distinct", "on cut", "Brier",
             "request", "delivered", "held?"))
    for cname, load in CORPORA:
        texts, y = load()
        out["corpora"][cname] = {"n": int(len(y)),
                                 "positive_rate": round(float(y.mean()), 4),
                                 "results": {}}
        for calname, cal in CALIBRATORS:
            rows = []
            for q in REQUESTS:
                r = evaluate(texts, y, cal, q)
                r["requested"] = q
                r["held"] = bool(r["delivered"] <= q)
                rows.append(r)
                print("%-17s %-9s %8.1f %8.1f%% %8.4f %9.2f %9.3f %9s"
                      % (cname if q == REQUESTS[0] and
                         calname == CALIBRATORS[0][0] else "",
                         calname if q == REQUESTS[0] else "",
                         r["distinct"], 100 * r["tie_share"], r["brier"],
                         q, r["delivered"], "yes" if r["held"] else "NO"),
                      flush=True)
            out["corpora"][cname]["results"][calname] = rows
        print()

    print("SUMMARY: contracts held, out of %d per calibrator" % (
        len(CORPORA) * len(REQUESTS)))
    for calname, _ in CALIBRATORS:
        held = sum(1 for c in out["corpora"].values()
                   for r in c["results"][calname] if r["held"])
        ties = float(np.mean([r["tie_share"] for c in out["corpora"].values()
                              for r in c["results"][calname]]))
        print("  %-9s %d held | mean share of negatives on the cut %.1f%%"
              % (calname, held, 100 * ties))

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print()
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
