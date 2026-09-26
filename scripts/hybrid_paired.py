# -*- coding: utf-8 -*-
"""What separates the fusion from its components, and what does not.

RESULTS_HYBRID.json reports the fusion ahead of every component on all four
measures under both label sets. That is not a ranking, and this paper has
already said why: overlapping marginal intervals license none, and with five
folds the project's paired Wilcoxon cannot return below p = 0.0625. Applied to
the fusion it returns p = 0.1875. Left there, the framework run would be a
table of point estimates the paper's own rules forbid reading.

A marginal interval is the wrong instrument for a difference. Both arms are
scored on the SAME messages, so the shared sample cancels in a paired
resample; two intervals can overlap while the paired difference is reliably
non-zero. This resamples the 1103 threads with replacement - the same
clustering, the same 2000 draws and the same seed as the run's own
thread_bootstrap - and takes the DIFFERENCE within each draw.

It can only do this for the threshold-free measures. hybrid_framework.py saves
the out-of-fold scores but not the per-fold thresholds or the binary
decisions, so no paired F1 or MCC is recoverable from the record. That is a
regression against the project's own practice, it is noted in the record this
writes, and the fix belongs in the next run rather than in a reconstruction
here.

What a positive result means here is narrow: the fused SCORE ranks sensitive
messages above harmless ones better than the component's score does, on this
corpus, this partition, these fitted models. It says nothing about F1, nothing
about a re-fitted fusion, and nothing about a different thread partition -
RESULTS_ROBERTA_SEEDS.json shows the encoder alone moving 0.349 to 0.393
across partitions, a spread wider than the gain being tested.

    python scripts/hybrid_paired.py
"""
import io
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import io_paths                                              # noqa: E402
import strengthen as S                                       # noqa: E402

OUT = io_paths.result_out("RESULTS_HYBRID_PAIRED.json")
N_BOOT, SEED = 2000, 42
COMPONENTS = ("llm", "encoder", "classical")


def load(name):
    return json.load(io.open(io_paths.result_in(name, required=True),
                             encoding="utf-8"))


def paired(y, a, b, groups, n=N_BOOT, seed=SEED):
    """The difference a - b, resampled over threads.

    The draw is the run's own: sorted unique groups, sampled with
    replacement, indices concatenated. Both arms are read through the SAME
    index vector, which is what makes it paired."""
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(groups)))
    by = {t: np.where(groups == t)[0] for t in uniq}
    d_roc, d_pr = [], []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), True)
        idx = np.concatenate([by[t] for t in pick])
        yy = y[idx]
        if len(set(yy)) < 2:
            continue
        d_roc.append(roc_auc_score(yy, a[idx]) - roc_auc_score(yy, b[idx]))
        d_pr.append(average_precision_score(yy, a[idx])
                    - average_precision_score(yy, b[idx]))

    def summarise(d):
        d = np.asarray(d)
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        return {"delta": round(float(d.mean()), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "share_above_zero": round(float((d > 0).mean()), 4),
                "draws": int(len(d))}
    return {"roc_auc": summarise(d_roc), "pr_auc": summarise(d_pr)}


def main():
    sc = load("SCORES_HYBRID.json")
    rec = load("RESULTS_HYBRID.json")
    df = S.build(dedup=True)
    g = df.thread_key.values
    assert len(df) == rec["dataset"]["messages"], "corpus does not match"

    print("paired thread-level differences, %d draws, seed %d"
          % (N_BOOT, SEED), flush=True)
    print("  scores read from the run; thresholds and decisions are not in "
          "the record, so F1 and MCC cannot be paired here", flush=True)
    print(flush=True)

    out = {"note": "paired thread-cluster bootstrap of the difference "
                   "between the fusion and each component, on the "
                   "threshold-free measures only",
           "limitation": "hybrid_framework.py saves out-of-fold scores but "
                         "not per-fold thresholds or binary decisions, so no "
                         "paired F1 or MCC is recoverable from this run",
           "draws": N_BOOT, "seed": SEED, "labels": {}}

    for lab in ("strict", "broad"):
        y = (df.label if lab == "strict" else df.label_either).values
        arms = {k: np.asarray(sc[lab][k], dtype=float)
                for k in COMPONENTS + ("hybrid",)}

        # Self-check first: if the saved scores do not reproduce the
        # record's own ROC and PR, nothing below is about the published run.
        #
        # They reproduce it closely but not exactly, and the reason is in the
        # record rather than in the analysis: hybrid_framework.py dumps the
        # scores rounded to six decimals (line 403) while it computed the
        # published metrics on the full-precision vectors. Rounding merges
        # near-ties, which moves a rank-based measure slightly - most on the
        # LLM arm, whose calibrated score takes only a few hundred distinct
        # values. The deviation is recorded rather than waved through, and
        # the tolerance is tight enough to catch a real mismatch.
        dev, coarse = {}, {}
        for k in arms:
            r = rec["%s_%s" % (k, lab)]
            got = (float(roc_auc_score(y, arms[k])),
                   float(average_precision_score(y, arms[k])))
            dev[k] = round(max(abs(got[0] - r["roc_auc"]),
                               abs(got[1] - r["pr_auc"])), 6)
            coarse[k] = len(np.unique(arms[k])) < len(y) // 2
            # A rank measure only moves under rounding where the score is
            # already tied, so the tolerance follows the arm's resolution.
            # The LLM's does not survive a tight one and should not: its raw
            # answer takes a few hundred distinct values, Platt is monotone
            # and cannot add any, and two thirds of the messages end up
            # sharing a value with another message. Every other arm keeps
            # over a thousand distinct values and reproduces to 5e-5.
            lim = 5e-3 if coarse[k] else 5e-4
            assert dev[k] < lim, (
                "%s/%s: saved scores give %.4f/%.4f, the record says "
                "%.4f/%.4f - too far apart to be the rounding in the dump"
                % (lab, k, got[0], got[1], r["roc_auc"], r["pr_auc"]))
        fine = max(v for k, v in dev.items() if not coarse[k])
        print("  %s: reproduces the record to %.5f on the resolved arms; "
              "the LLM arm to %.5f, on %d distinct values"
              % (lab, fine, dev["llm"], len(np.unique(arms["llm"]))),
              flush=True)

        out["labels"][lab] = {
            "reproduction_deviation": dev,
            "llm_distinct_values": int(len(np.unique(arms["llm"]))),
            "n": int(len(y))}
        for k in COMPONENTS:
            d = paired(y, arms["hybrid"], arms[k], g)
            out["labels"][lab][k] = d
            for m in ("roc_auc", "pr_auc"):
                v = d[m]
                print("    hybrid - %-10s %-8s %+.4f  95%% [%+.4f, %+.4f]  %s"
                      % (k, m, v["delta"], v["ci95"][0], v["ci95"][1],
                         "separates" if v["excludes_zero"] else "includes 0"),
                      flush=True)
        print(flush=True)

    sep = [(lab, k, m)
           for lab, d in out["labels"].items()
           for k in COMPONENTS
           for m in ("roc_auc", "pr_auc") if d[k][m]["excludes_zero"]]
    out["separating"] = ["%s/%s/%s" % t for t in sep]

    # The comparison that decides the claim is against the BEST component,
    # not against each of them: beating a weak arm is not a result. The best
    # component differs by label set - the encoder on strict, the classical
    # model on broad - so it is read from the record rather than assumed.
    best, decisive = {}, {}
    for lab in ("strict", "broad"):
        b = max(COMPONENTS, key=lambda k: rec["%s_%s" % (k, lab)]["f1"])
        best[lab] = b
        decisive[lab] = {m: out["labels"][lab][b][m]["excludes_zero"]
                         for m in ("roc_auc", "pr_auc")}
    out["best_component"] = best
    out["separates_from_best"] = decisive

    held = [("%s %s" % (lab, m))
            for lab in ("strict", "broad")
            for m in ("roc_auc", "pr_auc") if decisive[lab][m]]
    out["verdict"] = (
        "against the best component of each label set (%s on strict, %s on "
        "broad) the fusion separates on %s. It does not separate on the "
        "others, and no claim about F1 or MCC can be made from this record "
        "at all, because the thresholds and decisions were not saved. "
        "Against the weaker components it separates more widely (%d of the "
        "%d comparisons overall), which is a smaller thing to have shown."
        % (best["strict"], best["broad"],
           " and ".join(held) if held else "nothing",
           len(sep), 2 * len(COMPONENTS) * 2))
    print("  " + out["verdict"], flush=True)
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print(flush=True)
    print("written %s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
