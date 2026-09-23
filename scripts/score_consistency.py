# -*- coding: utf-8 -*-
"""Run an independent consistency checker over the figures this paper reports.

Fazekas and Kovacs (arXiv:2310.12527) give deterministic numerical tests for
whether a set of reported binary-classification scores is even arithmetically
attainable on the stated test set. They apply it to published medical papers
and find scores that cannot be produced by any confusion matrix of the claimed
size. Their package is mlscorecheck.

Pointing it at our own tables is the obvious thing for a paper about
evaluation to do. A pass is not a claim that the numbers are right - it says
only that they are attainable and mutually consistent - but a failure would be
decisive, and it costs nothing to look.

Figures are read from the released result files rather than retyped.
"""
import io
import json

from mlscorecheck.check.binary import check_1_testset_no_kfold

RESULTS = r"C:\Users\lenovo\Downloads\RESULTS_FINAL.json"
LLM = r"C:\Users\lenovo\Downloads\RESULTS_LLM.json"


def check(name, rec, n_pos, n_neg):
    """Feed the reported scores back as if we were an outside reader."""
    scores = {k: rec[k] for k in ("acc", "sens", "spec", "f1")
              if k in rec}
    # our records store precision/recall/f1; translate to the package's names
    scores = {}
    if "recall" in rec:
        scores["sens"] = rec["recall"]
    if all(k in rec for k in ("tn", "fp")):
        scores["spec"] = rec["tn"] / float(rec["tn"] + rec["fp"])
    if "f1" in rec:
        scores["f1p"] = rec["f1"]
    if all(k in rec for k in ("tp", "tn", "fp", "fn")):
        tot = rec["tp"] + rec["tn"] + rec["fp"] + rec["fn"]
        scores["acc"] = (rec["tp"] + rec["tn"]) / float(tot)
    result = check_1_testset_no_kfold(
        testset={"p": n_pos, "n": n_neg}, scores=scores, eps=1e-4)
    flag = "INCONSISTENT" if result["inconsistency"] else "consistent"
    print("  %-34s %s" % (name, flag))
    if result["inconsistency"]:
        print("     scores: %s" % scores)
    return result["inconsistency"]


def main():
    bad = 0
    print("=== classical and encoder results (RESULTS_FINAL.json) ===")
    try:
        d = json.load(io.open(RESULTS, encoding="utf-8"))
    except IOError:
        print("  RESULTS_FINAL.json not present yet")
        d = {}
    ds = d.get("dataset", {})
    for key, rec in sorted(d.items()):
        if not isinstance(rec, dict) or "tp" not in rec:
            continue
        pos = rec["tp"] + rec["fn"]
        neg = rec["tn"] + rec["fp"]
        bad += check(key, rec, pos, neg)

    print()
    print("=== LLM results (RESULTS_LLM.json) ===")
    try:
        d2 = json.load(io.open(LLM, encoding="utf-8"))
    except IOError:
        print("  RESULTS_LLM.json not present yet")
        d2 = {}
    for key, rec in sorted(d2.items()):
        if not isinstance(rec, dict) or "tp" not in rec:
            continue
        pos = rec["tp"] + rec["fn"]
        neg = rec["tn"] + rec["fp"]
        bad += check(key, rec, pos, neg)

    print()
    print("inconsistent records: %d" % bad)
    if ds:
        print("dataset as recorded: %d messages, %d strict positives"
              % (ds.get("messages", 0), ds.get("sensitive_strict", 0)))


if __name__ == "__main__":
    main()
