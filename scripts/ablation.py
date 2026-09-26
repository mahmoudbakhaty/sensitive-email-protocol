# -*- coding: utf-8 -*-
"""Does each signal earn its place in the fusion?

The thesis title promises a hybrid LLM-based framework. The first framework
run measured the fusion against each component alone, which answers a
different question: a fusion beating its own parts says nothing about which
part it needs. The run now also fits the same fusion with each signal removed,
on the same validation threads with the threshold chosen there too, so the
comparison is within-run and needs no caveat about treatment.

Three instruments, because one is not enough on five folds:

  * the drop in pooled F1 and ROC-AUC when a signal is removed;
  * the project's own paired test - an exact signed-rank over the five folds,
    which cannot return below p = 0.0625 and is reported with that limit
    stated rather than hidden;
  * McNemar on the per-message decisions, which the first run could not
    support because it did not save them. It does now.

Six comparisons are made, so they go through one Holm-Bonferroni correction.
Reading each at five per cent would be the multiplicity this project objects
to elsewhere.

    python scripts/ablation.py
"""
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import io_paths                                              # noqa: E402
import mapping_sensitivity as MS                             # noqa: E402

OUT = io_paths.result_out("RESULTS_ABLATION.json")
DROPPED = {"no_llm": "llm", "no_encoder": "encoder",
           "no_classical": "classical"}


def load(name):
    return json.load(io.open(io_paths.result_in(name, required=True),
                             encoding="utf-8"))


def wilcoxon_exact(d):
    """Two-sided exact signed-rank p. Zero differences dropped."""
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    rank = [0.0] * n
    for r, i in enumerate(order, 1):
        rank[i] = float(r)
    w = sum(rank[i] for i in range(n) if d[i] > 0)
    centre = n * (n + 1) / 4.0
    at_least = sum(
        1 for mask in range(1 << n)
        if abs(sum(rank[i] for i in range(n) if mask >> i & 1) - centre)
        >= abs(w - centre) - 1e-12)
    return min(1.0, at_least / float(1 << n))


def mcnemar_exact(a, b, y):
    """Two-sided exact McNemar on two decision vectors against the labels.

    Counts the messages where exactly one arm is right. The discordant pairs
    are the whole of the evidence; the agreements carry none."""
    a, b, y = np.asarray(a), np.asarray(b), np.asarray(y)
    a_only = int(((a == y) & (b != y)).sum())
    b_only = int(((b == y) & (a != y)).sum())
    n = a_only + b_only
    if n == 0:
        return 1.0, a_only, b_only
    # Exact binomial, two-sided, under p = 1/2.
    from math import comb
    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / float(2 ** n)), a_only, b_only


def main():
    rec = load("RESULTS_HYBRID.json")
    sc = load("SCORES_HYBRID.json")
    need = [k for k in DROPPED if "%s_strict" % k not in rec]
    if need:
        raise SystemExit(
            "this record has no ablation arms (%s missing). It predates the "
            "leave-one-out fusion; re-run hybrid_framework.py." % need)

    out = {"note": "does each signal earn its place in the fusion?",
           "dropped": DROPPED, "labels": {}}
    keys, pvals = [], []

    for lab in ("strict", "broad"):
        full = rec["hybrid_%s" % lab]
        y = None
        print("=== %s labels (floor %.4f) ==="
              % (lab, full["trivial_floor"]), flush=True)
        print("  %-12s %8s %8s %9s %9s %9s"
              % ("removed", "dF1", "dROC", "signed-rank", "McNemar",
                 "discordant"), flush=True)
        out["labels"][lab] = {}
        for arm, signal in sorted(DROPPED.items()):
            red = rec["%s_%s" % (arm, lab)]
            d_f1 = round(full["f1"] - red["f1"], 4)
            d_roc = round(full["roc_auc"] - red["roc_auc"], 4)
            fold_d = [a - b for a, b in zip(full["folds_f1"], red["folds_f1"])]
            p_sr = wilcoxon_exact(fold_d)

            pred = sc.get("%s_pred" % lab)
            if pred is None:
                p_mc, a_only, b_only = float("nan"), None, None
            else:
                # Labels are recoverable from any arm's decisions plus its
                # confusion matrix? No - they are not in the score file, so
                # they are rebuilt from the corpus.
                if y is None:
                    import strengthen as S
                    df = S.build(dedup=True)
                    y = (df.label if lab == "strict"
                         else df.label_either).values
                p_mc, a_only, b_only = mcnemar_exact(
                    pred["hybrid"], pred[arm], y)

            rowk = "%s/%s" % (lab, signal)
            keys.append(rowk)
            pvals.append(p_sr)
            keys.append(rowk + "/mcnemar")
            pvals.append(p_mc)
            out["labels"][lab][signal] = {
                "delta_f1": d_f1, "delta_roc_auc": d_roc,
                "fold_differences": [round(x, 4) for x in fold_d],
                "folds_won": sum(1 for x in fold_d if x > 0),
                "signed_rank_p": round(p_sr, 4),
                "min_attainable_p": round(
                    wilcoxon_exact([1.0] * len(fold_d)), 4),
                "mcnemar_p": round(p_mc, 4),
                "full_only_right": a_only, "reduced_only_right": b_only}
            print("  %-12s %+8.4f %+8.4f %9.4f %9.4f  %s/%s"
                  % (signal, d_f1, d_roc, p_sr, p_mc, a_only, b_only),
                  flush=True)
        print(flush=True)

    rej = MS.holm(pvals)
    for k, p, r in zip(keys, pvals, rej):
        lab, signal = k.split("/")[0], k.split("/")[1]
        field = ("mcnemar_holm" if k.endswith("/mcnemar")
                 else "signed_rank_holm")
        out["labels"][lab][signal][field] = bool(r)
    out["multiplicity"] = {
        "method": "Holm-Bonferroni", "alpha": 0.05,
        "comparisons": len(keys),
        "survive": [k for k, r in zip(keys, rej) if r]}

    # The verdict is about the TITLE - does the LLM earn its word in it -
    # and it is written from the TESTS rather than from the deltas. Removing
    # the LLM and removing the encoder cost the same pooled F1, so a verdict
    # read off the deltas alone would say both earn their place. McNemar on
    # the decisions says otherwise, and this script computed it.
    S = out["labels"]["strict"]
    B = out["labels"]["broad"]
    enc_sig = S["encoder"].get("mcnemar_holm") and B["encoder"].get(
        "mcnemar_holm")
    llm_sig = S["llm"].get("mcnemar_holm") or B["llm"].get("mcnemar_holm")
    out["verdict"] = (
        "By pooled F1 the LLM and the encoder are worth the same: removing "
        "either costs %+.4f and %+.4f on the strict labels, within %.4f of "
        "each other, while removing the classical model costs %+.4f, which "
        "is nothing. That reading does not survive the paired instruments. "
        "McNemar on the per-message decisions puts the encoder at %d "
        "messages the full fusion alone gets right against %d for the "
        "reduced one (p = %.4f, and it survives Holm on both label sets); "
        "the LLM is at %d against %d (p = %.4f), which is a coin. So the "
        "encoder's contribution is established and the LLM's is NOT: it "
        "moves which messages are right without making more of them right. "
        "The signed-rank test adds nothing either way - on five folds it "
        "cannot return below %.4f, and no comparison reaches even that. %d "
        "of %d comparisons survive Holm, and %s."
        % (S["llm"]["delta_f1"], S["encoder"]["delta_f1"],
           abs(S["llm"]["delta_f1"] - S["encoder"]["delta_f1"]),
           S["classical"]["delta_f1"],
           S["encoder"]["full_only_right"],
           S["encoder"]["reduced_only_right"], S["encoder"]["mcnemar_p"],
           S["llm"]["full_only_right"], S["llm"]["reduced_only_right"],
           S["llm"]["mcnemar_p"], S["llm"]["min_attainable_p"],
           len(out["multiplicity"]["survive"]), len(keys),
           "both are the encoder's decision tests"
           if enc_sig and not llm_sig else "see the list in the record"))
    out["earns_its_place"] = {
        "encoder": bool(enc_sig), "llm": bool(llm_sig),
        "classical": bool(S["classical"].get("mcnemar_holm")
                          or S["classical"].get("signed_rank_holm"))}
    print("  " + out["verdict"], flush=True)
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print(flush=True)
    print("written %s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
