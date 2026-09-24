# -*- coding: utf-8 -*-
"""Why the leak promise transfers and the block promise does not.

filter_system.py can keep one half of its contract. "No more than X% of
sensitive messages pass automatically" holds at every request measured; "no
more than Y% of harmless messages are blocked automatically" overshoots by
1.2x to 3.1x. Both thresholds are set the same way, on the same validation
threads, with the same bound. Only one holds.

An observation is not a result until the asymmetry has a cause, so three
candidate explanations are measured rather than argued.

  SHIFT. The threshold is a quantile of a validation distribution applied to a
  test distribution. If the upper tail of the harmless scores moves between
  folds more than the lower tail of the sensitive scores does, the block side
  is asking more of the same machinery.

  DENSITY. A threshold in a crowded region sweeps in many messages when it
  moves a little. The two sides sit in different parts of the score range and
  need not be equally crowded.

  COUNT. A 1% promise over 1,132 harmless messages is eleven messages; a 5%
  promise over 250 sensitive ones is twelve. Similar in absolute terms, so if
  this were the whole story the two sides would fail together.

Each is measured per fold, on the same splits the system uses, with the
validation and test values side by side.
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

OUT = io_paths.result_out("RESULTS_WHY_BLOCK_FAILS.json")
Q = 0.05          # the request both sides are measured at


def local_density(vals, t, width):
    """Share of `vals` within +/- width of t."""
    if len(vals) == 0:
        return 0.0
    return float(np.mean(np.abs(np.asarray(vals) - t) <= width))


def main():
    df = S.build(dedup=True)
    texts, y, g = df.text.tolist(), df.label.values, df.thread_key.values
    print("messages %d | sensitive %d | harmless %d"
          % (len(df), y.sum(), (1 - y).sum()))
    print("both sides measured at a %.0f%% request" % (100 * Q))
    print()

    rows = []
    for k, (tr, te) in enumerate(
            GroupKFold(n_splits=FS.FOLDS).split(texts, y, g), 1):
        f = FS.SensitivityFilter(max_leak=Q, max_false_block=Q)
        f.fit([texts[i] for i in tr], y[tr], g[tr])

        # the validation split the filter used, recovered the same way
        rng = np.random.RandomState(FS.SEED)
        th = np.array(sorted(set(g[tr])))
        rng.shuffle(th)
        va_th = set(th[:max(1, int(FS.VAL_FRAC * len(th)))])
        va = np.array([i for i in tr if g[i] in va_th])

        r_va = f.risk([texts[i] for i in va])
        r_te = f.risk([texts[i] for i in te])
        y_va, y_te = y[va], y[te]

        # where the two thresholds would fall if set on each side's own data
        q_va_leak = float(np.quantile(r_va[y_va == 1], Q))
        q_te_leak = float(np.quantile(r_te[y_te == 1], Q))
        q_va_blk = float(np.quantile(r_va[y_va == 0], 1 - Q))
        q_te_blk = float(np.quantile(r_te[y_te == 0], 1 - Q))

        spread = float(np.std(r_va))
        # how crowded each threshold is, measured in the same units on both
        # sides: a window of a tenth of the score spread
        w = 0.1 * spread
        d_leak = local_density(r_te[y_te == 1], q_va_leak, w)
        d_blk = local_density(r_te[y_te == 0], q_va_blk, w)

        row = {
            "fold": k,
            "score_sd": round(spread, 4),
            "leak_side": {
                "val_quantile": round(q_va_leak, 4),
                "test_quantile": round(q_te_leak, 4),
                "shift": round(q_te_leak - q_va_leak, 4),
                "shift_in_sd": round((q_te_leak - q_va_leak) / spread, 3),
                "local_density": round(d_leak, 4),
                "n_val": int((y_va == 1).sum()),
                "n_test": int((y_te == 1).sum()),
            },
            "block_side": {
                "val_quantile": round(q_va_blk, 4),
                "test_quantile": round(q_te_blk, 4),
                "shift": round(q_te_blk - q_va_blk, 4),
                "shift_in_sd": round((q_te_blk - q_va_blk) / spread, 3),
                "local_density": round(d_blk, 4),
                "n_val": int((y_va == 0).sum()),
                "n_test": int((y_te == 0).sum()),
            },
        }
        rows.append(row)
        print("fold %d  score sd %.3f" % (k, spread))
        for side in ("leak_side", "block_side"):
            r = row[side]
            print("   %-11s val q %.3f -> test q %.3f  shift %+.3f "
                  "(%+.2f sd)  density %.3f  n_val %d"
                  % (side.replace("_side", ""), r["val_quantile"],
                     r["test_quantile"], r["shift"], r["shift_in_sd"],
                     r["local_density"], r["n_val"]))
    print()

    def agg(side, field):
        return [r[side][field] for r in rows]

    summary = {}
    for side in ("leak_side", "block_side"):
        sh = np.abs(agg(side, "shift_in_sd"))
        summary[side] = {
            "mean_abs_shift_in_sd": round(float(np.mean(sh)), 3),
            "max_abs_shift_in_sd": round(float(np.max(sh)), 3),
            "mean_local_density": round(float(np.mean(agg(side,
                                                          "local_density"))),
                                        4),
            "mean_n_val": round(float(np.mean(agg(side, "n_val"))), 1),
        }

    print("ACROSS FOLDS")
    print("  %-12s %14s %14s %14s %10s"
          % ("side", "mean |shift|", "max |shift|", "density", "n val"))
    for side in ("leak_side", "block_side"):
        s = summary[side]
        print("  %-12s %13.3f %14.3f %14.4f %10.1f"
              % (side.replace("_side", ""), s["mean_abs_shift_in_sd"],
                 s["max_abs_shift_in_sd"], s["mean_local_density"],
                 s["mean_n_val"]))
    print()
    print("  shift is in units of the score's own standard deviation, so the")
    print("  two sides are comparable; density is the share of that side's")
    print("  test messages within a tenth of an sd of the threshold.")
    print()

    ls, bs = summary["leak_side"], summary["block_side"]
    verdicts = []
    if bs["mean_abs_shift_in_sd"] > 1.5 * ls["mean_abs_shift_in_sd"]:
        verdicts.append("SHIFT: the block threshold moves between folds "
                        "%.1f times as much as the leak one"
                        % (bs["mean_abs_shift_in_sd"]
                           / max(1e-9, ls["mean_abs_shift_in_sd"])))
    if bs["mean_local_density"] > 1.5 * ls["mean_local_density"]:
        verdicts.append("DENSITY: the block threshold sits %.1f times as "
                        "crowded as the leak one, so the same error in the "
                        "threshold sweeps in more messages"
                        % (bs["mean_local_density"]
                           / max(1e-9, ls["mean_local_density"])))
    if not verdicts:
        verdicts.append("neither shift nor density separates the two sides; "
                        "the asymmetry is not explained by these measurements")
    for v in verdicts:
        print("  -> " + v)

    json.dump({"environment": S.ENV, "request": Q, "folds": rows,
               "summary": summary, "verdicts": verdicts},
              io.open(OUT, "w", encoding="utf-8"), indent=1)
    print()
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
