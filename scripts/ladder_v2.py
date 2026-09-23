# -*- coding: utf-8 -*-
"""The leakage ladder, re-measured so that each rung changes only one thing.

The first ladder had a flaw its own paper warned against. Comparing R2 against
R3 changed three things at once:

  1. thread disjointness - the control being measured;
  2. class balance across folds - StratifiedKFold holds the test positive rate
     to within 0.003, plain GroupKFold lets it range over 0.048, and F1 on an
     imbalanced problem moves with the positive rate;
  3. the luck of one partition - every rung was a single split.

Only the first is leakage. This version fixes all three: every rung is the
mean over twenty seeded partitions, and the grouped rung uses
StratifiedGroupKFold so that the class balance matches the ungrouped rungs.

Whatever this returns replaces the published ladder. It is run under the same
pinned environment as the rest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_paths                                          # noqa: E402

import io
import json

import numpy as np

import strengthen as S

OUT = io_paths.result_out("RESULTS_ladder_v2.json")
SEEDS = list(range(42, 62))


def rung(X, y, g, split, Xv):
    f = [S.run_cv(X, y, g, split, seed=s, Xv=Xv)[0]["f1"] for s in SEEDS]
    m = [S.run_cv(X, y, g, split, seed=s, Xv=Xv)[0]["mcc"] for s in SEEDS]
    return (round(float(np.mean(f)), 4), round(float(np.std(f)), 4),
            round(float(np.mean(m)), 4))


def main():
    raw = S.build(dedup=False)
    ded = S.build(dedup=True)
    keep, dropped = S.near_dup_keep(ded.text.tolist())
    nd = ded[keep].reset_index(drop=True)

    out = {"environment": S.ENV, "seeds": len(SEEDS),
           "near_duplicates_dropped": int(dropped),
           "note": "every rung is the mean of 20 seeded partitions; the "
                   "grouped rung uses StratifiedGroupKFold so fold class "
                   "balance matches the ungrouped rungs"}

    for lab in ("strict", "broad"):
        col = "label" if lab == "strict" else "label_either"
        rows = []
        for name, df, split in (
                ("R0  none (duplicates kept)", raw, "stratified"),
                ("R1  + exact deduplication", ded, "stratified"),
                ("R2  + near-duplicate removal", nd, "stratified"),
                ("R3  + thread-disjoint split", nd, "grouped")):
            X = df.text.tolist()
            y = df[col].values
            g = df.thread_key.values
            f, sd, mcc = rung(X, y, g, split, S.vectorise(X))
            rows.append({"rung": name, "n": int(len(y)), "f1": f,
                         "f1_sd": sd, "mcc": mcc,
                         "trivial_floor": S.trivial_f1(y)})
            print("  %-30s n=%4d  F1 %.4f +/- %.4f  MCC %.4f"
                  % (name, len(y), f, sd, mcc), flush=True)

        f = [r["f1"] for r in rows]
        deltas = {"exact_deduplication": round(f[1] - f[0], 4),
                  "near_duplicate_removal": round(f[2] - f[1], 4),
                  "thread_disjoint_split": round(f[3] - f[2], 4),
                  "total_R0_to_R3": round(f[3] - f[0], 4)}
        out[lab] = {"rungs": rows, "deltas": deltas}
        print("  deltas: %s" % deltas)
        print(flush=True)

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=2)
    print("written %s" % OUT)


if __name__ == "__main__":
    print("=== ladder v2: one control per rung, 20 seeds, balance held ===")
    main()
