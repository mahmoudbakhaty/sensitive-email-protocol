# -*- coding: utf-8 -*-
"""Score a completed Section III annotation against our label mapping.

make_annotation_sample.py draws a blind, thread-disjoint, stratified sample and
builds the instrument. This reads what comes back and answers the question the
release cannot answer by computation: does the mapping from seven Enron genre
categories agree with the paper's own definition, on messages read one at a
time by a person applying that definition?

WHAT IS REPORTED, AND WHY EACH ONE:

  * Precision of our mapping - of the messages we call sensitive, how many does
    the annotator agree are sensitive. A low value means we are labelling
    things sensitive that the definition does not cover.
  * Recall of our mapping - of the messages the annotator calls sensitive, how
    many do we catch. A low value means the genre categories are blind to a
    kind of sensitivity the definition includes, which is the failure mode
    mapping_sensitivity.py explicitly cannot see.
  * Agreement, four ways, matching agreement_metrics.py so the number is
    comparable with the 0.662 between the corpus's own two annotators.

THE FLOOR IS A PER-JUDGEMENT ERROR RATE, NOT THE PAIR RATE. A repeat pair
disagrees if EITHER of its two judgements flipped, so a person whose
per-judgement error is e shows a pair-disagreement of 2e(1-e) - roughly twice
e. The quantity being compared against it, "our label against theirs", has a
null of e, not 2e. An earlier version compared the two directly, which put the
bar at twice its correct height, and then compared a Wilson LOWER bound on 50
judgements against a Wilson UPPER bound on 20 pairs, which is a test at about
alpha 0.005 rather than 0.05. Between them the BEYOND NOISE verdicts were
close to unable to fire. The pair rate is now inverted to e before anything is
compared with it, and the comparison is a one-sided binomial test against e's
upper confidence bound.

THE REPEAT BLOCK IS WHAT MAKES THE RECALL NUMBER MEAN ANYTHING. Simulating
annotators against this instrument showed that one who is 90% self-consistent
and a mapping that is genuinely blind to a tenth of the definition produce the
same recall - nothing in the data separates them. Some messages are therefore
shown twice under different ids, and the disagreement rate between a person and
themselves becomes the floor that every other disagreement rate is read
against. Repeats are excluded from the table itself; they are evidence about
the annotator, not about the corpus.

THE SAMPLE IS STRATIFIED, SO EVERY CORPUS-LEVEL FIGURE IS WEIGHTED. Half the
sample is drawn from 18% of the corpus, so counting the sample directly would
report a corpus that does not exist. Each positive-stratum message stands for
N_pos/n_pos messages and each negative-stratum message for N_neg/n_neg, and the
bootstrap resamples within stratum so the interval respects the design.
Unweighted sample figures are printed beside them, because the weighting
multiplies a handful of negative-stratum disagreements into a large estimate
and the reader should see how thin that evidence is.

OUT-OF-SCOPE AND UNSURE ARE EXCLUDED FROM THE RATES AND COUNTED IN THE OPEN.
Section III puts identifier-bearing messages outside the task, so scoring them
as errors would measure the wrong thing; but a mapping whose disagreements are
mostly "unsure" has not been validated, it has been postponed, and the count
says which happened.

    python scripts/score_annotation.py annotation/annotations.json
"""
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import io_paths                                              # noqa: E402

OUT = io_paths.result_out("RESULTS_SECTION_III_ANNOTATION.json")
KEY = os.path.join(REPO, "annotation", "key.json")
N_BOOT, SEED = 2000, 42
SENSITIVE, NOT = "SENSITIVE", "not sensitive"


def per_judgement_error(pair_rate):
    """Invert 2e(1-e) = pair_rate for e, the per-judgement error rate.

    A repeat pair disagrees when either judgement flipped, so the observed pair
    rate is about twice the rate that matters. Returns 0.5 when the pair rate
    is at or above 0.5, where the annotator carries no information."""
    d = 1.0 - 2.0 * pair_rate
    return 0.5 if d <= 0 else (1.0 - np.sqrt(d)) / 2.0


def binom_greater(k, n, p):
    """One-sided P(X >= k) under Binomial(n, p). No scipy dependency."""
    if p <= 0:
        return 0.0 if k > 0 else 1.0
    if p >= 1 or k <= 0:
        return 1.0
    from math import comb
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i)
                     for i in range(int(k), n + 1)))


def wilson(k, n, z=1.96):
    """Binomial interval that behaves at 0 and at small n, unlike the normal
    approximation - which matters here because the repeat block is 20 items."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / float(n)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


def weighted_table(rows, w_pos, w_neg):
    """2x2, our label against the annotator's, weighted to the corpus."""
    t = {(1, 1): 0.0, (1, 0): 0.0, (0, 1): 0.0, (0, 0): 0.0}
    for ours, theirs, stratum in rows:
        t[(ours, theirs)] += w_pos if stratum == "positive" else w_neg
    return t


def metrics(t):
    tp, fp = t[(1, 1)], t[(1, 0)]
    fn, tn = t[(0, 1)], t[(0, 0)]
    n = tp + fp + fn + tn
    if n <= 0:
        return None
    po = (tp + tn) / n
    p1 = (tp + fp) / n                       # our positive rate
    p2 = (tp + fn) / n                       # the annotator's
    pe_k = p1 * p2 + (1 - p1) * (1 - p2)
    pbar = (p1 + p2) / 2.0
    pe_g = 2.0 * pbar * (1 - pbar)
    return {
        "precision_of_our_mapping": tp / (tp + fp) if tp + fp else None,
        "recall_of_our_mapping": tp / (tp + fn) if tp + fn else None,
        "f1_of_our_mapping": (2 * tp / (2 * tp + fp + fn)
                              if 2 * tp + fp + fn else None),
        "observed_agreement": po,
        "cohens_kappa": (po - pe_k) / (1 - pe_k) if pe_k < 1 else None,
        "gwets_ac1": (po - pe_g) / (1 - pe_g) if pe_g < 1 else None,
        "our_positive_rate": p1,
        "annotator_positive_rate": p2,
    }


def main():
    path = (sys.argv[1] if len(sys.argv) > 1
            else os.path.join(REPO, "annotation", "annotations.json"))
    if not os.path.isfile(path):
        raise SystemExit(
            "no annotations at %s\n\nRun scripts/make_annotation_sample.py, "
            "work through annotation/annotate.html, press Export, and put the "
            "downloaded annotations.json there." % path)
    if not os.path.isfile(KEY):
        raise SystemExit("no key at %s - re-run make_annotation_sample.py"
                         % KEY)

    ann = json.load(io.open(path, encoding="utf-8"))
    kf = json.load(io.open(KEY, encoding="utf-8"))
    key, corpus_rate = kf["key"], kf["corpus_positive_rate"]
    if ann.get("seed") != kf.get("seed"):
        raise SystemExit("seed mismatch: annotations %s, key %s. These are "
                         "different samples." % (ann.get("seed"),
                                                 kf.get("seed")))

    def binary(aid):
        d = ann["annotations"].get(aid, {}).get("derived")
        return 1 if d == SENSITIVE else 0 if d == NOT else None

    rows, skipped = [], {"out of scope": 0, "unsure": 0, "incomplete": 0,
                         "unknown aid": 0}
    for aid, a in ann["annotations"].items():
        if aid not in key:
            skipped["unknown aid"] += 1
            continue
        if key[aid].get("repeat_of"):
            continue                      # evidence about the annotator only
        d = a.get("derived")
        if d == SENSITIVE:
            rows.append((key[aid]["our_label"], 1, key[aid]["stratum"]))
        elif d == NOT:
            rows.append((key[aid]["our_label"], 0, key[aid]["stratum"]))
        else:
            skipped[d if d in skipped else "incomplete"] += 1

    pairs = [(v["repeat_of"], aid) for aid, v in key.items()
             if v.get("repeat_of")]
    seen = [(binary(a), binary(b)) for a, b in pairs]
    seen = [(x, y) for x, y in seen if x is not None and y is not None]
    self_dis = (sum(1 for x, y in seen if x != y) / float(len(seen))
                if seen else None)

    n_done = len(rows)
    n_total = sum(1 for v in key.values() if not v.get("repeat_of"))
    npos_s = sum(1 for r in rows if r[2] == "positive")
    nneg_s = n_done - npos_s
    print("annotations scored: %d of %d in the sample" % (n_done, n_total))
    print("  scored:   %d from the positive stratum, %d from the negative"
          % (npos_s, nneg_s))
    print("  excluded: %s"
          % ", ".join("%d %s" % (v, k) for k, v in skipped.items() if v))
    if npos_s < 5 or nneg_s < 5:
        raise SystemExit("\ntoo few scored in one stratum to say anything. "
                         "Finish more of the sample.")
    if n_done < n_total:
        print("  PARTIAL: %d unscored. Figures below describe what is done."
              % (n_total - n_done))
    print()

    print("=== how consistent is the annotator with themselves ===")
    if not seen:
        print("  no repeat pairs completed - every rate below is "
              "uninterpretable, because there is nothing to read it against")
    else:
        lo, hi = wilson(sum(1 for x, y in seen if x != y), len(seen))
        print("  %d messages judged twice, disagreed with themselves on %d"
              % (len(seen), sum(1 for x, y in seen if x != y)))
        print("  self-disagreement %.1f%%, 95%% CI [%.1f%%, %.1f%%]"
              " <- the noise floor"
              % (100 * self_dis, 100 * lo, 100 * hi))
    print()

    npos_k = sum(1 for v in key.values() if v["our_label"])
    w_pos = (corpus_rate / npos_s) if npos_s else 0.0
    w_neg = ((1 - corpus_rate) / nneg_s) if nneg_s else 0.0
    m_w = metrics(weighted_table(rows, w_pos, w_neg))
    m_u = metrics(weighted_table(rows, 1.0, 1.0))

    print("=== our mapping judged against Section III ===")
    print("  %-30s %12s %12s" % ("", "weighted", "in-sample"))
    for k, lab in (("precision_of_our_mapping", "precision of our mapping"),
                   ("recall_of_our_mapping", "recall of our mapping"),
                   ("f1_of_our_mapping", "F1 of our mapping"),
                   ("observed_agreement", "observed agreement"),
                   ("cohens_kappa", "Cohen's kappa"),
                   ("gwets_ac1", "Gwet's AC1")):
        print("  %-30s %12s %12s"
              % (lab,
                 "-" if m_w[k] is None else "%.3f" % m_w[k],
                 "-" if m_u[k] is None else "%.3f" % m_u[k]))
    print()
    print("  weighted = back to the corpus's %.1f%% positive rate; in-sample "
          "= the 50/50 draw" % (100 * corpus_rate))

    rng = np.random.RandomState(SEED)
    pos_rows = [r for r in rows if r[2] == "positive"]
    neg_rows = [r for r in rows if r[2] == "negative"]
    boots = {k: [] for k in ("precision_of_our_mapping",
                             "recall_of_our_mapping", "cohens_kappa")}
    for _ in range(N_BOOT):
        s = ([pos_rows[i] for i in rng.randint(0, len(pos_rows),
                                               len(pos_rows))] +
             [neg_rows[i] for i in rng.randint(0, len(neg_rows),
                                               len(neg_rows))])
        mm = metrics(weighted_table(s, w_pos, w_neg))
        for k in boots:
            if mm and mm[k] is not None:
                boots[k].append(mm[k])
    ci = {k: ([round(float(np.percentile(v, 2.5)), 3),
               round(float(np.percentile(v, 97.5)), 3)] if v else None)
          for k, v in boots.items()}
    print()
    print("  95%% CI, stratified bootstrap, %d resamples:" % N_BOOT)
    for k, lab in (("precision_of_our_mapping", "precision"),
                   ("recall_of_our_mapping", "recall"),
                   ("cohens_kappa", "kappa")):
        print("    %-12s [%s]" % (lab, ", ".join("%.3f" % x for x in ci[k])
                                  if ci[k] else "-"))

    print()
    print("=== where it disagrees ===")
    dis = [(aid, a) for aid, a in ann["annotations"].items()
           if aid in key and not key[aid].get("repeat_of")
           and a.get("derived") in (SENSITIVE, NOT)
           and (a["derived"] == SENSITIVE) != bool(key[aid]["our_label"])]
    print("  %d disagreements of %d scored" % (len(dis), n_done))
    fp = [d for d in dis if key[d[0]]["our_label"] == 1]
    fn = [d for d in dis if key[d[0]]["our_label"] == 0]
    print("  %d we call sensitive and the definition does not" % len(fp))
    print("  %d the definition calls sensitive and we miss" % len(fn))
    for lab, group in (("we say sensitive, Section III does not", fp),
                       ("Section III says sensitive, we miss it", fn)):
        if not group:
            continue
        print()
        print("  %s:" % lab)
        for aid, a in group[:8]:
            q = (' quoted "%s"' % a["quote"]) if a.get("quote") else ""
            print("    %s  harm=%s quotable=%s id=%s%s"
                  % (aid, a.get("harm"), a.get("quotable"),
                     a.get("identifier"), q))

    prec, rec = m_w["precision_of_our_mapping"], m_w["recall_of_our_mapping"]
    lo_p = (ci["precision_of_our_mapping"] or [0])[0]
    lo_r = (ci["recall_of_our_mapping"] or [0])[0]

    # Each stratum's disagreement rate against the annotator's own error rate,
    # not against their pair rate: a pair disagrees if either judgement
    # flipped, so the pair rate is about twice the quantity being tested.
    n_self = sum(1 for x, y in seen if x != y) if seen else 0
    pair_hi = wilson(n_self, len(seen))[1] if seen else 1.0
    e_hat = per_judgement_error(self_dis) if seen else None
    e_hi = per_judgement_error(pair_hi)

    p_dis, p_ci = len(fp) / float(npos_s), wilson(len(fp), npos_s)
    n_dis, n_ci = len(fn) / float(nneg_s), wilson(len(fn), nneg_s)
    # One-sided binomial against the UPPER bound of the annotator's own error
    # rate, so the annotator's uncertainty is charged against the finding.
    p_p = binom_greater(len(fp), npos_s, e_hi) if seen else 1.0
    p_n = binom_greater(len(fn), nneg_s, e_hi) if seen else 1.0
    ALPHA = 0.05

    print()
    print("=== read against the annotator's own noise ===")
    if seen:
        print("  %d of %d repeat pairs disagreed -> per-judgement error "
              "%.1f%% (upper bound %.1f%%)"
              % (n_self, len(seen), 100 * e_hat, 100 * e_hi))
    else:
        print("  no repeat pairs: the floor is unknown, so nothing below can "
              "be called real")
    print("  our positives the definition rejects : %2d of %2d = %.1f%% "
          "[%.1f%%, %.1f%%]   p=%.3f%s"
          % (len(fp), npos_s, 100 * p_dis, 100 * p_ci[0], 100 * p_ci[1], p_p,
             "  BEYOND NOISE" if p_p < ALPHA else ""))
    print("  our negatives the definition accepts : %2d of %2d = %.1f%% "
          "[%.1f%%, %.1f%%]   p=%.3f%s"
          % (len(fn), nneg_s, 100 * n_dis, 100 * n_ci[0], 100 * n_ci[1], p_n,
             "  BEYOND NOISE" if p_n < ALPHA else ""))
    print("  p is a one-sided binomial test against the annotator's own "
          "error rate at its upper bound")

    over = bool(seen) and p_p < ALPHA
    blind = bool(seen) and p_n < ALPHA
    print()
    print("=== verdict ===")
    if not seen:
        v = ("no repeat pairs were completed, so the annotator's own "
             "consistency is unknown and none of these rates can be told "
             "apart from noise. Finish the repeats before quoting anything")
    elif e_hi >= 0.25:
        v = ("THE ANNOTATION IS TOO NOISY TO CONCLUDE FROM: the annotator's "
             "own per-judgement error could be as high as %.0f%%, which is "
             "the same size as the effects being measured" % (100 * e_hi))
    elif over and blind:
        v = ("THE MAPPING DIVERGES IN BOTH DIRECTIONS beyond the annotator's "
             "own noise: %.0f%% of our positives rejected (p=%.3f) and %.0f%% "
             "of our negatives accepted (p=%.3f). The benchmark's positive "
             "class is not what Section III describes"
             % (100 * p_dis, p_p, 100 * n_dis, p_n))
    elif over:
        v = ("THE MAPPING OVER-LABELS beyond noise: %.0f%% of what we call "
             "sensitive fails the definition (p=%.3f, precision %.2f, lower "
             "bound %.2f). The positive class is broader than Section III"
             % (100 * p_dis, p_p, prec, lo_p))
    elif blind:
        v = ("THE MAPPING IS BLIND beyond noise: %.0f%% of what we call "
             "harmless meets the definition (p=%.3f, recall %.2f, lower bound "
             "%.2f). This is exactly the failure mapping_sensitivity.py could "
             "not see, because every mapping in that family shares it"
             % (100 * n_dis, p_n, rec, lo_r))
    else:
        v = ("neither disagreement rate clears the annotator's own error rate "
             "of %.0f%% (p=%.2f and p=%.2f): precision %.2f [%.2f, -], recall "
             "%.2f [%.2f, -]. The mapping is consistent with the definition as "
             "far as %d judgements can show, which is not the same as being "
             "confirmed by them"
             % (100 * e_hi, p_p, p_n, prec, lo_p, rec, lo_r, n_done))
    print("  " + v)

    doc = {"scored": n_done, "sample_size": n_total, "excluded": skipped,
           "self_disagreement": (None if self_dis is None
                                 else round(self_dis, 4)),
           "repeat_pairs_scored": len(seen),
           "pair_disagreement_upper": round(pair_hi, 4),
           "per_judgement_error": (None if e_hat is None
                                   else round(e_hat, 4)),
           "per_judgement_error_upper": round(e_hi, 4),
           "p_over_labels": round(p_p, 4),
           "p_blind": round(p_n, 4),
           "our_positives_rejected": [len(fp), npos_s],
           "our_negatives_accepted": [len(fn), nneg_s],
           "over_labels_beyond_noise": bool(over),
           "blind_beyond_noise": bool(blind),
           "corpus_positive_rate": corpus_rate,
           "sample_positives_available": npos_k,
           "weighted": {k: (None if v is None else round(v, 4))
                        for k, v in m_w.items()},
           "in_sample": {k: (None if v is None else round(v, 4))
                         for k, v in m_u.items()},
           "ci95": ci, "disagreements": len(dis),
           "we_say_sensitive_definition_does_not": len(fp),
           "definition_says_sensitive_we_miss": len(fn),
           "verdict": v}
    json.dump(doc, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print()
    print("written %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
