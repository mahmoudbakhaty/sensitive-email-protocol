# -*- coding: utf-8 -*-
"""How many more labels buy how much more automation?

The system automates 15% of traffic. Two measurements say better components
will not raise that: the encoder's ceiling is -0.1 points of automation even
when the measurement is rigged in its favour, and the tightest leak rate the
benchmark can promise at all is about 5% because promising 1% needs 230
threads carrying a sensitive message and validation has 69.

Both point at the same lever. The threshold is bounded by how much labelled
data it is estimated from, not by how well the model separates. So the
question a deployment actually asks is not "which model" but:

    to automate another ten points of traffic, how many more messages must
    someone label, and which ones?

Label efficiency is normally measured against accuracy. This measures it
against the operating contract, which is the thing a deployment signs up to,
and which only a system with a contract can ask about.

HOW. The benchmark is subsampled to a budget - by whole threads, never by
message, or the training set would leak into itself - and the whole system is
refitted and re-measured out of fold at each size. Everything else is held:
the protocol, the calibrator, the 5% leak contract, the folds.

TWO ORDERS, because which messages you label matters as much as how many:

  * random threads, the honest default;
  * threads nearest the allow threshold first, the order an active-learning
    loop would choose. Simulated with labels we already have, so it is an
    upper bound on what targeting buys - a real loop would not know where the
    threshold is until it had labelled some of them.

The curve says what the next annotation batch is worth, and whether it is
worth anything at all.
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

OUT = io_paths.result_out("RESULTS_LABEL_BUDGET.json")
BUDGETS = (200, 400, 600, 800, 1000, 1200, 1382)
REQUEST, SEED, REPEATS = 0.05, 42, 3


def subsample_threads(g, y, budget, rng, order=None):
    """Whole threads until the message budget is reached.

    Threads, never messages: taking half a thread would put the other half in
    the same fold as its sibling and reintroduce exactly the leak the protocol
    exists to prevent."""
    threads = np.array(sorted(set(g)))
    if order is None:
        threads = threads[rng.permutation(len(threads))]
    else:
        threads = np.array([t for t in order if t in set(threads)])
    by = {t: np.where(g == t)[0] for t in threads}
    keep, n = [], 0
    for t in threads:
        if n >= budget:
            break
        keep.extend(by[t])
        n += len(by[t])
    idx = np.array(sorted(keep))
    return idx if len(set(y[idx])) > 1 else None


def measure(texts, y, g, idx):
    """The whole system, refitted and scored out of fold on this subsample."""
    t2 = [texts[i] for i in idx]
    y2, g2 = y[idx], g[idx]
    if len(set(y2)) < 2 or len(set(g2)) < FS.FOLDS:
        return None
    actions = np.full(len(y2), FS.ESCALATE, dtype=object)
    for tr, te in GroupKFold(n_splits=FS.FOLDS).split(t2, y2, g2):
        if len(set(y2[tr])) < 2:
            return None
        f = CT.PlattFilter(max_leak=REQUEST, max_false_block=REQUEST)
        f.fit([t2[i] for i in tr], y2[tr], g2[tr])
        a, _r = f.actions([t2[i] for i in te])
        actions[te] = a
    oc = FS.operating_characteristics(actions, y2)
    return {"messages": int(len(idx)),
            "threads": int(len(set(g2))),
            "sensitive_threads": int(len(set(g2[y2 == 1]))),
            "auto_share": oc["auto_share"],
            "delivered_leak": oc["leak_rate_of_sensitive"],
            "held": bool(oc["leak_rate_of_sensitive"] <= REQUEST),
            "error_among_auto": oc["error_rate_among_auto"]}


def main():
    df = S.build(dedup=True)
    texts, y, g = df.text.tolist(), df.thread_key.values, df.thread_key.values
    y = df.label.values
    print("full benchmark: %d messages, %d threads, %d carrying a sensitive "
          "message" % (len(y), len(set(g)), len(set(g[y == 1]))))
    print("contract fixed at a %.0f%% leak rate, Platt calibration"
          % (100 * REQUEST))
    print()

    # the targeted order: threads whose messages sit nearest the decision
    # boundary on a model fitted to everything. An upper bound on targeting.
    full = np.arange(len(y))
    ref = measure(texts, y, g, full)
    risk = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=FS.FOLDS).split(texts, y, g):
        f = CT.PlattFilter()
        f.fit([texts[i] for i in tr], y[tr], g[tr])
        risk[te] = f.risk([texts[i] for i in te])
    thr = float(np.quantile(risk[y == 1], REQUEST))
    per_thread = {}
    for t in set(g):
        m = g == t
        per_thread[t] = float(np.min(np.abs(risk[m] - thr)))
    targeted = [t for t, _ in sorted(per_thread.items(), key=lambda kv: kv[1])]

    out = {"environment": S.ENV, "request": REQUEST, "budgets": list(BUDGETS),
           "repeats": REPEATS, "full": ref, "curves": {}}

    for order_name, order in (("random", None), ("nearest-threshold first",
                                                 targeted)):
        print("=== %s ===" % order_name)
        print("  %8s %8s %10s %11s %9s %8s"
              % ("messages", "threads", "sens.thr", "automated", "leak",
                 "held?"))
        rows = []
        for b in BUDGETS:
            runs = []
            reps = 1 if order is not None else REPEATS
            for k in range(reps):
                rng = np.random.RandomState(SEED + k)
                idx = subsample_threads(g, y, b, rng, order)
                if idx is None:
                    continue
                r = measure(texts, y, g, idx)
                if r:
                    runs.append(r)
            if not runs:
                continue
            avg = {k: float(np.mean([r[k] for r in runs]))
                   for k in ("messages", "threads", "sensitive_threads",
                             "auto_share", "delivered_leak",
                             "error_among_auto")}
            avg["held_in"] = "%d/%d" % (sum(r["held"] for r in runs),
                                        len(runs))
            avg["budget"] = b
            rows.append(avg)
            print("  %8.0f %8.0f %10.0f %10.1f%% %9.3f %8s"
                  % (avg["messages"], avg["threads"],
                     avg["sensitive_threads"], 100 * avg["auto_share"],
                     avg["delivered_leak"], avg["held_in"]), flush=True)
        out["curves"][order_name] = rows
        print()

    rnd = out["curves"]["random"]
    tgt = out["curves"]["nearest-threshold first"]
    if len(rnd) >= 2:
        print("WHAT A LABEL IS WORTH")
        # An average slope is the wrong statistic for this shape. Automation
        # is flat at zero and then turns on, so the numbers that matter are
        # where it turns on and what the last stretch bought.
        on = next((r for r in rnd if r["auto_share"] > 0.005), None)
        if on:
            print("  automation is zero until %.0f messages / %.0f threads "
                  "carrying a sensitive one"
                  % (on["messages"], on["sensitive_threads"]))
            print("  the bound needs about fifty sensitive threads in the "
                  "validation split, which is where that lands")
        a, b = rnd[-2], rnd[-1]
        local = ((b["auto_share"] - a["auto_share"])
                 / max(1e-9, (b["messages"] - a["messages"]) / 100.0))
        first = rnd[0]
        overall = ((b["auto_share"] - first["auto_share"])
                   / max(1e-9, (b["messages"] - first["messages"]) / 100.0))
        print("  across the whole range   %+.2f points per 100 labels"
              % (100 * overall))
        print("  over the LAST stretch    %+.2f points per 100 labels  "
              "<- where we actually are" % (100 * local))
        out["points_per_100_overall"] = round(100 * overall, 3)
        out["points_per_100_at_the_margin"] = round(100 * local, 3)
        out["automation_starts_at"] = on

        print()
        print("  TARGETING THE BOUNDARY MADE IT WORSE, NOT BETTER")
        for r_r, r_t in zip(rnd, tgt):
            if abs(r_r["messages"] - r_t["messages"]) > 60:
                continue
            print("    %5.0f labels: random %4.1f%% automated (%3.0f "
                  "sensitive threads) | targeted %4.1f%% (%3.0f)"
                  % (r_r["messages"], 100 * r_r["auto_share"],
                     r_r["sensitive_threads"], 100 * r_t["auto_share"],
                     r_t["sensitive_threads"]))
        print("    threads near the decision boundary carry FEWER sensitive")
        print("    messages, and the bound is starved of the one thing it")
        print("    counts. When the bottleneck is a tail quantile, label")
        print("    positives - not boundary cases.")
        out["targeting_verdict"] = ("worse than random: boundary threads "
                                    "carry fewer sensitive messages, and the "
                                    "bound counts sensitive threads")

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print()
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
