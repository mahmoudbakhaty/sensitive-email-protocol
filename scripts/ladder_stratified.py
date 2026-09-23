# -*- coding: utf-8 -*-
"""How much of the thread-grouping cost is leakage, and how much is imbalance?

The ladder in strengthen.py compares a StratifiedKFold run (R2) against a
plain GroupKFold run (R3). That changes two things at once. Thread disjointness
is the one we mean to measure. The other is class balance: StratifiedKFold
holds the test positive rate to within 0.003 across folds, while plain
GroupKFold lets it range over 0.048 on this corpus, because it does not
stratify. F1 on an imbalanced problem moves with the positive rate, so part of
the measured drop is a nuisance term rather than leakage.

StratifiedGroupKFold holds both: groups stay intact AND folds stay balanced.
Comparing R2 against a stratified-grouped R3 isolates the leakage.

Run under the same pinned environment as everything else.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_paths                                          # noqa: E402

import io
import json

import numpy as np
from sklearn.model_selection import (GroupKFold, StratifiedGroupKFold,
                                     StratifiedKFold)

import strengthen as S

OUT = io_paths.result_out("RESULTS_ladder_stratified.json")
SEEDS = list(range(42, 62))          # 20 partitions, as in the repeated run


def fold_balance(y, splits):
    r = [float(y[te].mean()) for _, te in splits]
    return round(max(r) - min(r), 4)


def main():
    raw = S.build(dedup=True)
    keep, dropped = S.near_dup_keep(raw.text.tolist())
    nd = raw[keep].reset_index(drop=True)
    X = nd.text.tolist()
    g = nd.thread_key.values
    Xv = S.vectorise(X)

    out = {"note": "R2 vs R3 with class balance held fixed",
           "near_duplicates_dropped": int(dropped),
           "environment": S.ENV}

    for lab in ("strict", "broad"):
        y = (nd.label if lab == "strict" else nd.label_either).values
        print("=== %s labels, n=%d, positive rate %.4f ==="
              % (lab, len(y), y.mean()), flush=True)

        # R2: ungrouped, stratified - the rung the ladder compares against
        r2 = [S.run_cv(X, y, g, "stratified", seed=s, Xv=Xv)[0]["f1"]
              for s in SEEDS]
        # R3 as published: grouped, NOT stratified
        r3_plain = S.run_cv(X, y, g, "grouped", Xv=Xv)[0]["f1"]
        # R3 with balance held: grouped AND stratified
        r3_strat = [S.run_cv(X, y, g, "grouped", seed=s, Xv=Xv)[0]["f1"]
                    for s in SEEDS]

        bal_plain = fold_balance(y, list(GroupKFold(n_splits=5)
                                         .split(X, y, g)))
        bal_strat = fold_balance(y, list(StratifiedGroupKFold(
            5, shuffle=True, random_state=42).split(X, y, g)))
        bal_r2 = fold_balance(y, list(StratifiedKFold(
            5, shuffle=True, random_state=42).split(X, y)))

        m2, m3 = float(np.mean(r2)), float(np.mean(r3_strat))
        rec = {
            "r2_stratified_mean": round(m2, 4),
            "r2_stratified_sd": round(float(np.std(r2)), 4),
            "r3_grouped_plain": round(r3_plain, 4),
            "r3_grouped_stratified_mean": round(m3, 4),
            "r3_grouped_stratified_sd": round(float(np.std(r3_strat)), 4),
            "drop_as_published": round(r3_plain - m2, 4),
            "drop_balance_held": round(m3 - m2, 4),
            "class_marginal_share": round((r3_plain - m3), 4),
            "fold_balance_spread": {"r2_stratified": bal_r2,
                                    "r3_grouped_plain": bal_plain,
                                    "r3_grouped_stratified": bal_strat},
            "seeds": len(SEEDS),
        }
        out[lab] = rec
        print("  R2 stratified          %.4f +/- %.4f  (fold spread %.4f)"
              % (rec["r2_stratified_mean"], rec["r2_stratified_sd"], bal_r2))
        print("  R3 grouped, plain      %.4f              (fold spread %.4f)"
              % (r3_plain, bal_plain))
        print("  R3 grouped, stratified %.4f +/- %.4f  (fold spread %.4f)"
              % (rec["r3_grouped_stratified_mean"],
                 rec["r3_grouped_stratified_sd"], bal_strat))
        print("  drop as published      %+.4f" % rec["drop_as_published"])
        print("  drop, balance held     %+.4f   <- the leakage" %
              rec["drop_balance_held"])
        print("  class-marginal share   %+.4f   <- not leakage" %
              rec["class_marginal_share"])
        print(flush=True)

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=2)
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
