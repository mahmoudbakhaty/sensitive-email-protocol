# -*- coding: utf-8 -*-
"""Does the filter design work anywhere else, or only here?

The calibration mechanism has been tested on three public corpora
(`RESULTS_EXTERNAL_CALIBRATION.md`). The filter has not. Those are different
questions: one asks whether isotonic ties up a score everywhere, the other
asks whether a three-outcome policy with a leak contract is a design that
works at all, or an arrangement that only makes sense on one benchmark.

It matters because the largest weakness in this work is a single corpus, and
no second corpus for context-dependent sensitivity exists to test the protocol
on. The design can still be tested, even if the task cannot.

On this benchmark the system automates 5.5% of traffic, delivers a 1.2% leak
rate against a 5% promise, and blocks nothing. Two readings, and they point
opposite ways:

  * the design is sound and this task is simply hard, in which case the same
    system on an easier corpus should automate most of the traffic;
  * the design is too conservative to be useful anywhere, in which case it
    will automate almost nothing on corpora where a linear model is nearly
    perfect.

SMS Spam, tweet_eval/hate and Enron-Spam span Brier 0.009 to 0.152 against
0.227 here, so they separate those readings cleanly. Nothing about the system
is retuned for them: the same contract, the same confidence, the same
precision requirement for blocking, the same calibrator. Only the folds change
- stratified rather than thread-grouped, because none of them has threads.
"""
import io
import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import external_calibration as EC                            # noqa: E402
import filter_system as FS                                   # noqa: E402
import io_paths                                              # noqa: E402

OUT = io_paths.result_out("RESULTS_SYSTEM_ELSEWHERE.json")
REQUESTS = (0.02, 0.05, 0.10)
SEED = 42

# this benchmark, from its own record, for the row that matters
OURS = {"corpus": "this benchmark (thread-grouped)", "n": 1382,
        "positive_rate": 0.1809, "brier": 0.2449}


def run(texts, y, q):
    """The system, unchanged, on stratified folds."""
    actions = np.full(len(y), FS.ESCALATE, dtype=object)
    groups = np.arange(len(y))            # no threads: each message its own
    for tr, te in StratifiedKFold(n_splits=FS.FOLDS, shuffle=True,
                                  random_state=SEED).split(texts, y):
        f = FS.SensitivityFilter(max_leak=q, max_false_block=q)
        f.fit([texts[i] for i in tr], y[tr], groups[tr])
        a, _r = f.actions([texts[i] for i in te])
        actions[te] = a
    return FS.operating_characteristics(actions, y)


def main():
    print("the system as it stands, unchanged, on corpora it was not designed "
          "for", flush=True)
    print("this benchmark automates 5.5%% at a 5%% request, leak 1.2%%, "
          "blocks nothing", flush=True)
    print()
    print("  %-17s %8s %10s %9s %11s %9s %7s"
          % ("corpus", "request", "automated", "of which", "leak", "err|auto",
             "held?"), flush=True)
    out = {"ours": OURS, "requests": list(REQUESTS), "corpora": {}}
    for cname, load in EC.CORPORA:
        texts, y = load()
        rows = []
        for q in REQUESTS:
            oc = run(texts, y, q)
            held = oc["leak_rate_of_sensitive"] <= q
            rows.append({"requested": q, "auto_share": oc["auto_share"],
                         "auto_allowed": oc["auto_allowed"],
                         "auto_blocked": oc["auto_blocked"],
                         "leak": oc["leak_rate_of_sensitive"],
                         "error_among_auto": oc["error_rate_among_auto"],
                         "escalated_share": oc["escalated_share"],
                         "held": bool(held)})
            print("  %-17s %8.2f %9.1f%% %10s %9.3f %8.1f%% %7s"
                  % (cname if q == REQUESTS[0] else "", q,
                     100 * oc["auto_share"],
                     "%d/%d" % (oc["auto_allowed"], oc["auto_blocked"]),
                     oc["leak_rate_of_sensitive"],
                     100 * oc["error_rate_among_auto"],
                     "yes" if held else "NO"), flush=True)
        out["corpora"][cname] = {"n": int(len(y)),
                                 "positive_rate": round(float(y.mean()), 4),
                                 "rows": rows}
        print()

    print("  'of which' is auto-allowed / auto-blocked", flush=True)
    print()
    print("WHAT THE DESIGN DOES ELSEWHERE", flush=True)
    at5 = {c: [r for r in v["rows"] if r["requested"] == 0.05][0]
           for c, v in out["corpora"].items()}
    best = max(at5.values(), key=lambda r: r["auto_share"])
    held_all = all(r["held"] for v in out["corpora"].values()
                   for r in v["rows"])
    blocks = sum(r["auto_blocked"] for v in out["corpora"].values()
                 for r in v["rows"])
    print("  at a 5%% request, automation ranges %.1f%% to %.1f%% "
          "(this benchmark: 5.5%%)"
          % (100 * min(r["auto_share"] for r in at5.values()),
             100 * max(r["auto_share"] for r in at5.values())))
    print("  every contract held: %s" % ("yes" if held_all else "NO"), flush=True)
    print("  messages auto-blocked anywhere: %d" % blocks, flush=True)
    print()
    if best["auto_share"] > 0.40:
        verdict = ("the design works: on an easier corpus the same system "
                   "automates %.0f%% of traffic under the same contract, so "
                   "5.5%% here is the task and not the policy"
                   % (100 * best["auto_share"]))
    elif best["auto_share"] > 0.15:
        verdict = ("the design works but modestly: the best external corpus "
                   "reaches %.0f%% automation, several times this benchmark "
                   "but far from clearing the traffic"
                   % (100 * best["auto_share"]))
    else:
        verdict = ("the design is too conservative to be useful: even where a "
                   "linear model is nearly perfect it automates only %.0f%%, "
                   "so the policy and not the task is the limit"
                   % (100 * best["auto_share"]))
    print("  " + verdict, flush=True)
    out["verdict"] = verdict
    out["all_contracts_held"] = bool(held_all)
    out["total_auto_blocked"] = int(blocks)

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print()
    print("written %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
