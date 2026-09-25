# -*- coding: utf-8 -*-
"""The strict-vs-inclusive count on Enron-Spam, measured, not remembered.

RESULTS_EXTERNAL_CALIBRATION.md carries a box contrasting `neg > t` with
`neg >= t` at a 10% request on Enron-Spam, and README.md repeats it. Both were
typed from a run that is no longer the released one: they say 0.8367 delivered
and 76.2% of negatives on the cut, while RESULTS_EXTERNAL_CALIBRATION.json -
the record those documents describe - says 0.6676 and 59.5% for the same cell.
The two cannot both be true of one release.

Rather than retype the record's numbers into the prose, this measures the
contrast through external_calibration.evaluate itself, in ONE pass: the rule
hook counts the strict variant as a side effect and returns the inclusive one,
so both come off the same folds, the same split and the same fitted models by
construction. A second call could not promise that.

The inclusive figure it produces is checked against the released record, so
this script cannot quietly disagree with the run the paper cites.

    python scripts/tie_diagnostic.py
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

OUT = io_paths.result_out("RESULTS_TIE_DIAGNOSTIC.json")
CORPUS, REQUEST = "Enron-Spam", 0.10


def counting_rule(box):
    """Return the inclusive count, accumulate the strict one and the ties.

    Tie equality is `abs(x - t) < 1e-9`, the same tolerance evaluate() uses for
    its tie_share, so the two cannot disagree about what sits on the cut."""
    def rule(neg_va, neg_te, q, _rng):
        t = float(np.quantile(neg_va, 1.0 - q))
        box["strict"] += int((neg_te > t).sum())
        box["inclusive"] += int((neg_te >= t).sum())
        box["ties"] += int((np.abs(neg_te - t) < 1e-9).sum())
        box["negatives"] += int(len(neg_te))
        return int((neg_te >= t).sum()), t
    return rule


def main():
    load = dict(EC.CORPORA)[CORPUS]
    print("strict vs inclusive on %s at a %.0f%% request, one pass"
          % (CORPUS, REQUEST * 100), flush=True)
    texts, y = load()
    box = {"strict": 0, "inclusive": 0, "ties": 0, "negatives": 0}
    r = EC.evaluate(texts, y, EC.cal_isotonic, REQUEST,
                    rule=counting_rule(box))

    neg = box["negatives"]
    strict = box["strict"] / float(neg)
    incl = box["inclusive"] / float(neg)
    tie_share = box["ties"] / float(neg)

    # The inclusive rate is what the release already published for this cell.
    # If this run disagrees with it, the diagnostic is describing a different
    # run than the paper cites and must not be used to correct the prose.
    rec = json.load(io.open(io_paths.result_in(
        "RESULTS_EXTERNAL_CALIBRATION.json", required=True),
        encoding="utf-8"))
    published = [row for row in rec["corpora"][CORPUS]["results"]["isotonic"]
                 if abs(row["requested"] - REQUEST) < 1e-9][0]
    agrees = abs(published["delivered"] - r["delivered"]) < 5e-4

    out = {"corpus": CORPUS, "request": REQUEST, "calibrator": "isotonic",
           "negatives": neg, "on_cut": box["ties"],
           "on_cut_share": round(tie_share, 4),
           "delivered_strict": round(strict, 4),
           "delivered_inclusive": round(incl, 4),
           "factor_over_request": round(incl / REQUEST, 1),
           "published_inclusive": published["delivered"],
           "agrees_with_release": bool(agrees)}

    print(flush=True)
    print("  negatives, pooled over folds   %d" % neg, flush=True)
    print("  sitting exactly on the cut     %d (%.1f%%)"
          % (box["ties"], tie_share * 100), flush=True)
    print("  delivered, counting neg >  t   %.4f" % strict, flush=True)
    print("  delivered, counting neg >= t   %.4f  (%.1f x request)"
          % (incl, incl / REQUEST), flush=True)
    print(flush=True)
    print("  released record for this cell  %.4f -> %s"
          % (published["delivered"],
             "agrees" if agrees else "*** DISAGREES ***"), flush=True)
    if not agrees:
        print("  the prose must not be corrected from this run", flush=True)
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print(flush=True)
    print("written %s" % OUT, flush=True)
    return 0 if agrees else 1


if __name__ == "__main__":
    sys.exit(main())
