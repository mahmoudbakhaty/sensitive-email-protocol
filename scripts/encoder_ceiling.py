# -*- coding: utf-8 -*-
"""What could the encoder add to the system? An optimistic upper bound.

filter_system.py runs on the two components a CPU can fit. The fine-tuned
encoder is a slot, and hybrid_framework.py fills it when a GPU is available -
which it has not been. The question worth answering first is whether it is
worth waiting for.

THIS MEASUREMENT LEAKS, ON PURPOSE, AND THE RESULT IS AN UPPER BOUND.

The encoder's released scores are out of fold: the score for message j came
from a model trained on every fold except j's. When those scores are used as a
component column, the training messages of MY fold k carry scores from models
that were trained on data including fold k. The encoder column therefore knows
something about the test fold, and the fusion fitted on it inherits that.

The leak is second-order - it reaches the fusion weights, not the labels - but
it is a leak, and this paper is about not pretending otherwise. So the number
below is what the encoder could add AT BEST. If the honest gain were larger
than this, something would be wrong.

The comparison is against the same system with the same folds, same policy and
same thresholds, differing only in whether the encoder column is present. Read
it as: is there enough here to spend the GPU quota on.
"""
import io
import json
import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrator_ties as CT                                 # noqa: E402
import filter_system as FS                                   # noqa: E402
import io_paths                                              # noqa: E402
import strengthen as S                                       # noqa: E402

OUT = io_paths.result_out("RESULTS_ENCODER_CEILING.json")
REQUESTS = (0.02, 0.05, 0.10)


def run(cls, texts, y, g, q, enc_by_text=None):
    actions = np.full(len(y), FS.ESCALATE, dtype=object)
    for tr, te in GroupKFold(n_splits=FS.FOLDS).split(texts, y, g):
        f = cls(max_leak=q, max_false_block=q)
        if enc_by_text is not None:
            f.add_component("fine-tuned encoder",
                            lambda ts: np.array([enc_by_text[t] for t in ts]))
        f.fit([texts[i] for i in tr], y[tr], g[tr])
        a, _r = f.actions([texts[i] for i in te])
        actions[te] = a
    return FS.operating_characteristics(actions, y)


def main():
    df = S.build(dedup=True)
    texts, y, g = df.text.tolist(), df.label.values, df.thread_key.values
    P = json.load(io.open(io_paths.result_in("PREDICTIONS_FINAL.json",
                                             required=True), encoding="utf-8"))
    enc = np.array(P["transformer_grouped_strict"]["score"])
    assert len(enc) == len(texts), (len(enc), len(texts))
    enc_by_text = {}
    for t, v in zip(texts, enc):
        enc_by_text.setdefault(t, float(v))
    print("messages %d | encoder scores %d | distinct texts %d"
          % (len(texts), len(enc), len(enc_by_text)))
    print("Platt calibration throughout - plain isotonic holds no external "
          "block contract, and randomising the cut recovers the rate in "
          "expectation but not per deployment "
          "(RESULTS_WHY_BLOCK_FAILS.md, RESULTS_CALIBRATOR_REMEDIES.md)")
    print()
    print("  %-26s %8s %9s %13s %11s"
          % ("system", "request", "leak", "false block", "automated"))

    rows = []
    for label, use_enc in (("two components", False),
                           ("+ encoder (UPPER BOUND)", True)):
        for q in REQUESTS:
            oc = run(CT.PlattFilter, texts, y, g, q,
                     enc_by_text if use_enc else None)
            rows.append({"system": label, "requested": q,
                         "delivered_leak": oc["leak_rate_of_sensitive"],
                         "delivered_false_block":
                             oc["false_block_rate_of_harmless"],
                         "auto_share": oc["auto_share"],
                         "error_rate_among_auto":
                             oc["error_rate_among_auto"]})
            print("  %-26s %8.2f %8.3f%s %12.3f%s %10.1f%%"
                  % (label, q, oc["leak_rate_of_sensitive"],
                     " " if oc["leak_rate_of_sensitive"] <= q else "*",
                     oc["false_block_rate_of_harmless"],
                     " " if oc["false_block_rate_of_harmless"] <= q else "*",
                     100 * oc["auto_share"]))
    print()
    print("  * marks a broken promise")
    print()

    base = {r["requested"]: r for r in rows if r["system"] == "two components"}
    gains = []
    for r in rows:
        if r["system"] == "two components":
            continue
        b = base[r["requested"]]
        d = r["auto_share"] - b["auto_share"]
        gains.append(d)
        print("  at a %.0f%% request the encoder column adds %+.1f points of "
              "automation, AT BEST (%.1f%% -> %.1f%%)"
              % (100 * r["requested"], 100 * d,
                 100 * b["auto_share"], 100 * r["auto_share"]))
    print()
    mean_gain = float(np.mean(gains))
    print("  mean upper-bound gain: %+.1f points of automation" % (100 * mean_gain))
    print("  the honest gain from a GPU run is smaller than this, by an")
    print("  unknown amount. Read it as a ceiling on what the quota buys.")

    json.dump({"environment": S.ENV, "rows": rows,
               "mean_upper_bound_gain_in_auto_share": round(mean_gain, 4),
               "caveat": "the encoder column is built from released "
                         "out-of-fold scores, so it carries information about "
                         "the test fold through the models that produced the "
                         "training messages' scores. Deliberately optimistic; "
                         "an upper bound, not a result."},
              io.open(OUT, "w", encoding="utf-8"), indent=1)
    print()
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
