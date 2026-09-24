# -*- coding: utf-8 -*-
"""Does a policy set on validation hold on traffic it has not seen?

filter_system.py sets its thresholds to bound two rates: at most 5% of
sensitive messages auto-allowed, at most 1% of harmless ones auto-blocked. On
the validation split those bounds hold by construction. Out of fold they do
not: the delivered rates were 7.2% and 6.3%.

That gap is the finding, and it is the same failure this whole project is
about - a number that holds where it was chosen and not where it is used. A
threshold picked at the 5th percentile of ~75 validation positives is an
estimate with a wide interval, and using it as though it were the true
quantile is optimism of exactly the kind Section IV-D warns about for decision
thresholds.

Two questions, both answerable here:

  1. HOW BIG IS THE GAP, across the range? Sweep the requested bound from 1%
     to 20% and record what is actually delivered out of fold. If delivered
     tracks requested with a constant multiplier, the policy is usable once
     inverted. If it wanders, it is not.

  2. DOES ASKING FOR LESS DELIVER WHAT YOU WANTED? For each target, find the
     request that delivers it, and report the price in coverage. A filter
     that honours its contract while automating less is worth more than one
     that automates more and breaks it.

Both are measured out of fold under the same thread-disjoint protocol.
"""
import io
import json
import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import filter_system as FS                                   # noqa: E402
import io_paths                                              # noqa: E402
import strengthen as S                                       # noqa: E402

OUT = io_paths.result_out("RESULTS_POLICY_TRANSFER.json")
REQUESTS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20)


def run_policy(texts, y, g, max_leak, max_false_block):
    """Out-of-fold actions under one requested policy."""
    actions = np.full(len(y), FS.ESCALATE, dtype=object)
    for tr, te in GroupKFold(n_splits=FS.FOLDS).split(texts, y, g):
        f = FS.SensitivityFilter(max_leak=max_leak,
                                 max_false_block=max_false_block)
        f.fit([texts[i] for i in tr], y[tr], g[tr])
        a, _r = f.actions([texts[i] for i in te])
        actions[te] = a
    return actions


def main():
    df = S.build(dedup=True)
    texts, y, g = df.text.tolist(), df.label.values, df.thread_key.values
    print("messages %d | sensitive %d (%.1f%%)"
          % (len(df), y.sum(), 100 * y.mean()), flush=True)
    print()
    print("REQUESTED against DELIVERED, out of fold", flush=True)
    print("  %9s %9s %9s %9s %9s %9s"
          % ("req leak", "got leak", "ratio", "req block", "got block",
             "ratio"), flush=True)

    rows = []
    for q in REQUESTS:
        a = run_policy(texts, y, g, q, q)
        oc = FS.operating_characteristics(a, y)
        gl = oc["leak_rate_of_sensitive"]
        gb = oc["false_block_rate_of_harmless"]
        rows.append({"requested": q, "delivered_leak": gl,
                     "delivered_false_block": gb,
                     "leak_ratio": round(gl / q, 2),
                     "block_ratio": round(gb / q, 2),
                     "auto_share": oc["auto_share"],
                     "error_rate_among_auto": oc["error_rate_among_auto"],
                     "escalated_share": oc["escalated_share"]})
        print("  %9.3f %9.3f %9.2f %9.3f %9.3f %9.2f"
              % (q, gl, gl / q, q, gb, gb / q), flush=True)

    print()
    print("  the price of honouring the contract", flush=True)
    print("  %9s %11s %11s %11s"
          % ("target", "request", "delivered", "automated"), flush=True)
    honoured = []
    for target in (0.01, 0.02, 0.05, 0.10):
        ok = [r for r in rows if r["delivered_leak"] <= target
              and r["delivered_false_block"] <= target]
        if not ok:
            print("  %9.3f %11s" % (target, "not reachable in this range"), flush=True)
            honoured.append({"target": target, "reachable": False})
            continue
        best = max(ok, key=lambda r: r["auto_share"])
        print("  %9.3f %11.3f %11.3f %10.1f%%"
              % (target, best["requested"],
                 max(best["delivered_leak"], best["delivered_false_block"]),
                 100 * best["auto_share"]), flush=True)
        honoured.append({"target": target, "reachable": True,
                         "request": best["requested"],
                         "delivered_leak": best["delivered_leak"],
                         "delivered_false_block": best["delivered_false_block"],
                         "auto_share": best["auto_share"]})

    out = {"environment": S.ENV,
           "note": "a policy set on validation threads, measured on the "
                   "thread-disjoint test folds it was not set on",
           "requests": list(REQUESTS), "rows": rows, "honoured": honoured}
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print()
    print("written %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
