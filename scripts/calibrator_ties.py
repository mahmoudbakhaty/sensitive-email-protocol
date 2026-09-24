# -*- coding: utf-8 -*-
"""The block promise is not hard to estimate. It is impossible to express.

why_block_fails.py ruled out the obvious explanation. The block threshold is
FOUR TIMES more stable between folds than the leak threshold - 0.063 against
0.252 standard deviations - so distribution shift is not the cause. What
separates them is density, and counting the ties says why.

Across one test fold the risk score takes 62 distinct values over 277
messages. At the leak threshold no message sits exactly on the cut. At the
block threshold 28 of 225 harmless messages do - 12.4% of them, on one value.

So a contract of "at most 5% of harmless messages above this cut" has no
solution. Move the cut a hair down and 12% go over; a hair up and 0% do.
Nothing in between exists. The bound is not failing to estimate a quantile;
there is no quantile there to estimate.

The cause is the calibrator. Isotonic regression is a step function: it maps
whole intervals of input to one output, and with a few hundred validation
points those steps are wide. It is chosen for calibration quality and it is
good at that, but it destroys exactly the resolution a rate guarantee needs.

This measures the alternative. Platt scaling is a fitted sigmoid - strictly
monotone, so it preserves every distinction the underlying model made. It
usually calibrates slightly worse. The question is whether the trade buys a
contract that holds:

  * how many distinct values each calibrator leaves, and how many messages
    pile up on the threshold;
  * whether the delivered rates then match the requested ones;
  * what it costs in calibration quality, measured as Brier score, so the
    trade is visible rather than assumed.
"""
import io
import json
import os
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import filter_system as FS                                   # noqa: E402
import io_paths                                              # noqa: E402
import strengthen as S                                       # noqa: E402

OUT = io_paths.result_out("RESULTS_CALIBRATOR_TIES.json")
REQUESTS = (0.02, 0.05, 0.10)


class IsotonicFilter(FS.SensitivityFilter):
    """The same system with the step-function calibrator.

    The base class is Platt now, because that is the calibrator measured to
    keep the contract. This subclass is what the comparison needs: isotonic,
    which calibrates a little better and cannot keep it."""

    def _calibrate(self, s_va, y_va):
        return _iso_calibrate(s_va, y_va)


# Kept as a name because two other scripts import it, and it is now simply
# the default system. A subclass that overrides nothing is a copy waiting to
# drift out of step with what it copies.
PlattFilter = FS.SensitivityFilter


def _iso_calibrate(s_va, y_va):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0,
                             y_max=1.0).fit(s_va, y_va)
    return iso.predict


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def main():
    df = S.build(dedup=True)
    texts, y, g = df.text.tolist(), df.label.values, df.thread_key.values
    print("messages %d | sensitive %d" % (len(df), y.sum()))
    print()

    # ---- resolution and ties, per calibrator, on the same folds ----------
    print("RESOLUTION OF THE RISK SCORE")
    print("  %-10s %10s %12s %14s %10s"
          % ("calibrator", "distinct", "on the cut", "of harmless", "Brier"))
    res = {}
    for name, cls in (("isotonic", IsotonicFilter),
                      ("Platt", FS.SensitivityFilter)):
        distinct, on_cut, share, briers = [], [], [], []
        for tr, te in GroupKFold(n_splits=FS.FOLDS).split(texts, y, g):
            f = cls()
            f.fit([texts[i] for i in tr], y[tr], g[tr])
            r = f.risk([texts[i] for i in te])
            yt = y[te]
            neg = r[yt == 0]
            t = float(np.quantile(neg, 0.95))
            k = int((np.abs(neg - t) < 1e-9).sum())
            distinct.append(len(set(np.round(r, 6))))
            on_cut.append(k)
            share.append(k / max(1, len(neg)))
            briers.append(brier(r, yt))
        res[name] = {"distinct": round(float(np.mean(distinct)), 1),
                     "on_cut": round(float(np.mean(on_cut)), 1),
                     "share_on_cut": round(float(np.mean(share)), 4),
                     "brier": round(float(np.mean(briers)), 4)}
        s = res[name]
        print("  %-10s %10.1f %12.1f %13.1f%% %10.4f"
              % (name, s["distinct"], s["on_cut"], 100 * s["share_on_cut"],
                 s["brier"]))
    print()

    # ---- does the contract hold with each? -------------------------------
    print("REQUESTED against DELIVERED")
    print("  %-10s %8s %10s %10s %12s"
          % ("calibrator", "request", "leak", "false block", "automated"))
    rows = []
    for name, cls in (("isotonic", IsotonicFilter),
                      ("Platt", FS.SensitivityFilter)):
        for q in REQUESTS:
            actions = np.full(len(y), FS.ESCALATE, dtype=object)
            for tr, te in GroupKFold(n_splits=FS.FOLDS).split(texts, y, g):
                f = cls(max_leak=q, max_false_block=q)
                f.fit([texts[i] for i in tr], y[tr], g[tr])
                a, _r = f.actions([texts[i] for i in te])
                actions[te] = a
            oc = FS.operating_characteristics(actions, y)
            rows.append({"calibrator": name, "requested": q,
                         "delivered_leak": oc["leak_rate_of_sensitive"],
                         "delivered_false_block":
                             oc["false_block_rate_of_harmless"],
                         "auto_share": oc["auto_share"]})
            print("  %-10s %8.2f %9.3f%s %9.3f%s %11.1f%%"
                  % (name, q, oc["leak_rate_of_sensitive"],
                     " " if oc["leak_rate_of_sensitive"] <= q else "*",
                     oc["false_block_rate_of_harmless"],
                     " " if oc["false_block_rate_of_harmless"] <= q else "*",
                     100 * oc["auto_share"]))
    print()
    print("  * marks a broken promise")

    json.dump({"environment": S.ENV, "resolution": res, "contract": rows,
               "note": "isotonic against Platt: ties, calibration quality, "
                       "and whether the rate contract can be honoured"},
              io.open(OUT, "w", encoding="utf-8"), indent=1)
    print()
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
