# -*- coding: utf-8 -*-
"""Is the permutation null too easy because it ignores thread structure?

The null in strengthen.py is Ojala and Garriga's Test 1: permute the labels,
null hypothesis that the data and the labels are independent. It permutes at
the message level, which destroys the fact that messages in one thread tend to
share a label. Real data has that coherence and the permuted data does not, so
the two differ in a way the test does not intend, and the p-value could be
optimistic.

The conservative version permutes at the level of the dependence structure:
whole threads exchange their label blocks, so within-thread coherence survives
and only the association between text and label is destroyed. Blocks are
exchanged among threads of equal size, which keeps every permutation a valid
relabelling of the same corpus.

If both nulls give the same answer the original p-value stands. If the
thread-level null is much higher, the paper should report that one.
"""
import io
import json
from collections import defaultdict

import numpy as np

import strengthen as S

OUT = r"C:\Users\lenovo\Downloads\RESULTS_perm_thread.json"
N_PERM = 500
SEED = 42


def thread_blocks(g):
    """Message indices per thread, grouped by thread size."""
    by = defaultdict(list)
    for i, t in enumerate(g):
        by[t].append(i)
    blocks = {t: np.array(v) for t, v in by.items()}
    by_size = defaultdict(list)
    for t, idx in blocks.items():
        by_size[len(idx)].append(t)
    return blocks, by_size


def permute_by_thread(y, blocks, by_size, rng):
    """Exchange whole label blocks between threads of the same size."""
    yp = np.empty_like(y)
    for size, threads in by_size.items():
        order = list(threads)
        shuffled = list(threads)
        rng.shuffle(shuffled)
        for src, dst in zip(order, shuffled):
            yp[blocks[dst]] = y[blocks[src]]
    return yp


def run(X, y, g, Xv, observed, kind, rng, blocks=None, by_size=None):
    null = []
    for i in range(N_PERM):
        if kind == "message":
            yp = rng.permutation(y)
        else:
            yp = permute_by_thread(y, blocks, by_size, rng)
        null.append(S.run_cv(X, yp, g, "grouped", Xv=Xv)[0]["f1"])
        if (i + 1) % 100 == 0:
            print("    %s %d/%d" % (kind, i + 1, N_PERM), flush=True)
    null = np.array(null)
    p = (1.0 + float((null >= observed).sum())) / (N_PERM + 1.0)
    return {"null_mean": round(float(null.mean()), 4),
            "null_sd": round(float(null.std()), 4),
            "null_p95": round(float(np.percentile(null, 95)), 4),
            "null_max": round(float(null.max()), 4),
            "p_value": round(p, 5)}


def main():
    ded = S.build(dedup=True)
    keep, _ = S.near_dup_keep(ded.text.tolist())
    nd = ded[keep].reset_index(drop=True)
    X = nd.text.tolist()
    g = nd.thread_key.values
    Xv = S.vectorise(X)
    blocks, by_size = thread_blocks(g)
    sizes = sorted(by_size)
    print("threads %d | sizes %s | exchangeable threads %d"
          % (len(blocks), sizes[:6],
             sum(len(v) for v in by_size.values() if len(v) > 1)))
    print()

    out = {"environment": S.ENV, "permutations": N_PERM,
           "note": "message-level vs thread-level label permutation"}

    for lab in ("strict", "broad"):
        y = (nd.label if lab == "strict" else nd.label_either).values
        obs = S.run_cv(X, y, g, "grouped", Xv=Xv)[0]["f1"]
        print("=== %s labels, observed F1 %.4f ===" % (lab, obs), flush=True)
        rng = np.random.RandomState(SEED)
        m = run(X, y, g, Xv, obs, "message", rng)
        rng = np.random.RandomState(SEED)
        t = run(X, y, g, Xv, obs, "thread", rng, blocks, by_size)
        out[lab] = {"observed_f1": round(obs, 4), "message_level": m,
                    "thread_level": t}
        print("  message-level null %.4f +/- %.4f  max %.4f  p = %.5f"
              % (m["null_mean"], m["null_sd"], m["null_max"], m["p_value"]))
        print("  thread-level  null %.4f +/- %.4f  max %.4f  p = %.5f"
              % (t["null_mean"], t["null_sd"], t["null_max"], t["p_value"]))
        print(flush=True)

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=2)
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
