# -*- coding: utf-8 -*-
"""Is our Cohen's kappa depressed by the kappa paradox?

James (LREC 2026) notes that kappa can read low despite high observed
agreement when the class distribution is skewed, and recommends Gwet's AC1 as
more stable in that case. Our positive class is 18%, which is skewed enough
for the question to be fair, and the paper leans on kappa = 0.662 to argue the
judgement is contested. If AC1 is much higher, the paper is overstating how
contested it is and should say so.

The quantity the paper's argument actually rests on - one annotator scored as
a classifier against the other, F1 = 0.744 - involves no chance correction at
all, so it is reported here beside the others to show what does and does not
move.

Intervals are thread-level bootstraps, as everywhere else.
"""
import io
import json

import numpy as np
import pandas as pd

import strengthen as S

OUT = r"C:\Users\lenovo\Downloads\RESULTS_agreement_metrics.json"
N_BOOT = 2000
SEED = 42


def metrics(agree):
    """agree: array of 'both' / 'one' / 'neither' per message."""
    both = int((agree == "both").sum())
    one = int((agree == "one").sum())
    neither = int((agree == "neither").sum())
    n = both + one + neither
    if n == 0 or both == 0:
        return None

    po = (both + neither) / float(n)

    # The corpus records how many annotators chose a category, not which, so
    # the two marginals are equal by construction and the disagreements split
    # evenly in expectation. Stated as a limitation in the paper.
    p_pos = (both + one / 2.0) / float(n)

    # Cohen's kappa
    pe_k = p_pos ** 2 + (1 - p_pos) ** 2
    kappa = (po - pe_k) / (1 - pe_k) if pe_k < 1 else None

    # Gwet's AC1: chance agreement from the overall prevalence rather than
    # from the product of the marginals, which is what makes it stable when
    # one class is rare.
    pe_g = 2.0 * p_pos * (1 - p_pos)
    ac1 = (po - pe_g) / (1 - pe_g) if pe_g < 1 else None

    # One annotator scored as a classifier against the other. No chance
    # correction anywhere in this quantity.
    f1 = 2.0 * both / (2.0 * both + one)

    return {"observed_agreement": po, "cohens_kappa": kappa,
            "gwets_ac1": ac1, "annotator_f1": f1, "positive_rate": p_pos,
            "both": both, "one": one, "neither": neither, "n": n}


def main():
    df = S.build(dedup=True)
    a = df.agree.values
    g = df.thread_key.values
    point = metrics(pd.Series(a).values)

    rng = np.random.RandomState(SEED)
    uniq = np.array(sorted(set(g)))
    by = {t: np.where(g == t)[0] for t in uniq}
    draws = {k: [] for k in ("cohens_kappa", "gwets_ac1", "annotator_f1",
                             "observed_agreement")}
    for _ in range(N_BOOT):
        idx = np.concatenate([by[t] for t in rng.choice(uniq, len(uniq), True)])
        m = metrics(a[idx])
        if m is None:
            continue
        for k in draws:
            if m[k] is not None:
                draws[k].append(m[k])

    out = {"environment": S.ENV, "resamples": N_BOOT,
           "note": "thread-level bootstrap; Gwet's AC1 added after James "
                   "(LREC 2026) flagged the kappa paradox under skew"}
    print("counts: both %d | one %d | neither %d | n %d"
          % (point["both"], point["one"], point["neither"], point["n"]))
    print("positive rate %.4f" % point["positive_rate"])
    print()
    for k in ("observed_agreement", "cohens_kappa", "gwets_ac1",
              "annotator_f1"):
        v = point[k]
        lo = float(np.percentile(draws[k], 2.5))
        hi = float(np.percentile(draws[k], 97.5))
        out[k] = {"estimate": round(v, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        print("  %-20s %.4f   95%% CI [%.4f, %.4f]" % (k, v, lo, hi))
    print()
    d = point["gwets_ac1"] - point["cohens_kappa"]
    out["ac1_minus_kappa"] = round(d, 4)
    print("  AC1 - kappa = %+.4f" % d)
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=2)
    print()
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
