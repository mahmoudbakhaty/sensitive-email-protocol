# -*- coding: utf-8 -*-
"""Does the paper's conclusion depend on our label mapping?

The first limitation the README lists is the one that cannot be checked by any
amount of re-running:

    Labels are the Enron subset's original category judgements mapped to our
    definition. The mapping is ours and is a source of construct error.

The Enron subset was annotated against a 1990s genre taxonomy - coarse genre,
forwarded material, primary topic, emotional tone - and nobody annotating it
was thinking about sensitivity. Seven of its category pairs were chosen as
"sensitive":

    (1,2)  Purely Personal
    (1,5)  Employment arrangements (job seeking, hiring, recommendations)
    (2,8)  Legal documents (complaints, lawsuits, advice)
    (3,4)  company image - changing / influencing
    (3,5)  political influence / contributions / contacts
    (3,10) legal advice
    (4,10) secrecy / confidentiality

That choice is ours, it is defensible, and it is not the only defensible one.
The usual remedy is to have people annotate a sample against the paper's own
Section III definition and report how well the mapping agrees. That needs
annotators. This does not, and it answers a question that matters more.

WHETHER THE MAPPING IS RIGHT IS UNANSWERABLE WITHOUT PEOPLE. WHETHER IT
MATTERS IS MEASURABLE RIGHT NOW. If every defensible mapping yields the same
conclusions, construct error in the mapping cannot be what is driving them,
and the limitation - while still real - stops threatening the result.

Twenty-one mappings are declared below before any of them was run: our seven,
each one dropped in turn, each plausible addition added in turn, and five
holistic alternatives a different researcher might have written down instead.
Each is a position someone could defend from the taxonomy text alone.

Two conclusions are then tested on every one of them.

  A. THE PAPER'S CONTRIBUTION: thread-disjoint evaluation scores lower than a
     random split, so random splits flatter a model. Published as -0.021 F1.
     If this flips sign or vanishes under a plausible remapping, the central
     claim is an artefact of our seven categories.

  B. THE SYSTEM'S HEADLINE: the filter automates only a small share of traffic
     under a 5% leak contract. Published as 5.5%. If some other mapping
     automates most of the corpus, the ceiling is a property of our labels and
     not of the task.

Reported as the range across mappings, with the worst case named. A conclusion
that only holds for the mapping its authors chose is not a conclusion.

SEVENTEEN MAPPINGS MEANS SEVENTEEN TESTS, AND THAT IS CHARGED FOR. A one-sided
"gap - 2 SE > 0" test has alpha 0.023, so across seventeen mappings the
expected number of spurious flags is 0.39 and the chance of seeing at least one
when every mapping is null is 32%. An earlier version reported "the effect
reverses under 1 of 17" without that context, which reads as a finding when it
is the single most likely outcome under the null. The count is now reported
beside what chance predicts, and a Holm-corrected flag is reported separately.
"""
import io
import json
import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import filter_system as FS                                   # noqa: E402
import io_paths                                              # noqa: E402
import strengthen as S                                       # noqa: E402

OUT = io_paths.result_out("RESULTS_MAPPING_SENSITIVITY.json")
SEEDS, FOLDS, SEED0 = 10, 5, 42
REQUEST = 0.05
MIN_POSITIVES = 60        # below this the filter cannot set a policy honestly

NAMES = {
    (1, 2): "Purely Personal",
    (1, 3): "Personal but in professional context",
    (1, 5): "Employment arrangements",
    (2, 6): "Government action(s)",
    (2, 8): "Legal documents",
    (3, 2): "internal projects - progress and strategy",
    (3, 3): "company image - current",
    (3, 4): "company image - changing / influencing",
    (3, 5): "political influence / contributions",
    (3, 7): "internal company policy",
    (3, 9): "alliances / partnerships",
    (3, 10): "legal advice",
    (4, 10): "secrecy / confidentiality",
    (4, 11): "worry / anxiety",
    (4, 18): "shame",
}

CORE = {(4, 10), (3, 10), (2, 8), (3, 5), (3, 4), (1, 5), (1, 2)}

# Defensible under a context-dependent reading of sensitivity, and left out.
ADDITIONS = [(1, 3), (3, 2), (3, 3), (3, 7), (3, 9), (2, 6), (4, 18), (4, 11)]

# The weakest members of our own set - each is arguably routine.
REMOVALS = [(3, 4), (1, 5), (3, 5)]

HOLISTIC = [
    ("confidentiality only", {(4, 10)}),
    ("legal only", {(2, 8), (3, 10)}),
    ("personal only", {(1, 2), (1, 3)}),
    ("narrow - the unambiguous three", {(4, 10), (3, 10), (2, 8)}),
    ("broad - core plus every addition", CORE | set(ADDITIONS)),
]


def mappings():
    """Declared before running. Each is a position defensible from the text."""
    out = [("ours (published)", set(CORE))]
    for c in sorted(REMOVALS):
        out.append(("ours minus %s" % NAMES[c], CORE - {c}))
    for c in ADDITIONS:
        out.append(("ours plus %s" % NAMES[c], CORE | {c}))
    out.extend(HOLISTIC)
    return out


# ---- corpus, read once ---------------------------------------------------

def load_once():
    """Every message with its raw categories, so relabelling costs nothing.

    strengthen.build() re-reads seventeen hundred files per call and reads the
    mapping from a module global. Rather than call it twenty-one times, the
    same loop runs once here and keeps the categories - and the result is
    checked against strengthen.build() under the published mapping, because a
    re-implementation that quietly differs would make every number below a
    number about this file instead of about the corpus."""
    import glob
    import hashlib
    import re
    rows, seen = [], set()
    for cf in sorted(glob.glob(os.path.join(S.BASE, "*", "*.cats"))):
        cats = [tuple(map(int, l.strip().split(",")))
                for l in open(cf) if l.strip().count(",") == 2]
        if any((t, s) in S.EMPTY for t, s, _ in cats):
            continue
        raw = open(cf.replace(".cats", ".txt"), encoding="utf-8",
                   errors="replace").read()
        i = raw.find("X-FileName:")
        body = (raw[raw.find("\n", i) + 1:] if i >= 0 else raw).strip()
        if len(body.split()) < 30:
            continue
        flat = S.WS.sub(" ", body.lower()).strip()
        h = hashlib.md5(flat.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        m = re.search(r"^Subject:[ \t]*(.*)$", raw, re.M | re.I)
        subj = m.group(1).strip() if m else ""
        rows.append({"cats": cats, "text": body,
                     "thread_key": S.WS.sub(
                         " ", S.SUBJ.sub("", subj).strip().lower())
                     or ("m:" + h[:10])})
    return rows


def labels_for(rows, sens):
    """Strict labels: both annotators assigned a category in the mapping."""
    return np.array([int(any((t, s) in sens and f >= 2
                             for t, s, f in r["cats"])) for r in rows])


def check_against_strengthen(rows):
    df = S.build(dedup=True)
    y = labels_for(rows, S.SENS)
    same = (len(rows) == len(df) and
            [r["text"] for r in rows] == df.text.tolist() and
            [r["thread_key"] for r in rows] == df.thread_key.tolist() and
            bool((y == df.label.values).all()))
    return same, len(df), int(df.label.sum())


# ---- A. the leakage effect ------------------------------------------------

def leakage_gap(tx, Xv, y, g):
    """F1 under a thread-disjoint split minus F1 under a random one.

    This is the published ladder's R2-to-R3 rung and nothing else: near
    duplicates already removed, StratifiedGroupKFold against StratifiedKFold
    so fold class balance matches, averaged over seeded partitions.
    strengthen.run_cv does the work, because a re-implementation here would
    make the comparison with -0.021 a comparison between two functions."""
    if y.sum() < 2 * FOLDS or (1 - y).sum() < 2 * FOLDS:
        return None, None, None, None
    rand, grp = [], []
    for s in range(SEEDS):
        rand.append(S.run_cv(tx, y, g, "random",
                             seed=SEED0 + s, Xv=Xv)[0]["f1"])
        grp.append(S.run_cv(tx, y, g, "grouped",
                            seed=SEED0 + s, Xv=Xv)[0]["f1"])
    rand, grp = np.array(rand), np.array(grp)
    # Paired over seeds: each seed gives one random partition and one grouped
    # one, so the difference is paired and its standard error is the spread of
    # those differences. Without this a gap of +0.03 on a 47-positive mapping
    # reads as a reversal when it is noise.
    d = grp - rand
    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    return (float(d.mean()), float(rand.mean()), float(grp.mean()), se)


def normal_sf(z):
    """P(Z > z). math.erfc keeps this dependency-free."""
    from math import erfc, sqrt
    return 0.5 * erfc(z / sqrt(2.0))


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni. Returns the boolean reject vector in input order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [False] * len(pvals)
    m = len(pvals)
    for rank, i in enumerate(idx):
        if pvals[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break
    return out


# ---- B. the automation ceiling -------------------------------------------

def automation(texts, y, groups):
    actions = np.full(len(y), FS.ESCALATE, dtype=object)
    for tr, te in GroupKFold(n_splits=FS.FOLDS).split(texts, y, groups):
        f = FS.SensitivityFilter(max_leak=REQUEST, max_false_block=REQUEST)
        f.fit([texts[i] for i in tr], y[tr], groups[tr])
        a, _r = f.actions([texts[i] for i in te])
        actions[te] = a
    oc = FS.operating_characteristics(actions, y)
    return oc["auto_share"], oc["leak_rate_of_sensitive"]


def main():
    rows = load_once()
    ok, n_ref, pos_ref = check_against_strengthen(rows)
    print("corpus read once: %d messages" % len(rows))
    print("  reproduces strengthen.build() under the published mapping: %s"
          % ("yes, %d messages and %d sensitive" % (n_ref, pos_ref) if ok
             else "NO - the relabelling loop differs, stop here"), flush=True)
    if not ok:
        return 1
    print()

    texts = [r["text"] for r in rows]
    groups = np.array([r["thread_key"] for r in rows])
    mask, dropped = S.near_dup_keep(texts)
    keep = np.where(mask)[0]
    tx_k = [texts[i] for i in keep]
    Xv_k = S.vectorise(tx_k)           # vectorised once, as the ladder does
    g_k = groups[keep]
    print("near-duplicates removed for the leakage rung: %d dropped, "
          "%d remain" % (dropped, len(keep)), flush=True)
    print()

    print("%-42s %6s %7s %9s %7s %9s %7s"
          % ("mapping", "sens.", "rate", "leak gap", "+-2SE", "automated",
             "leak"), flush=True)
    out = {"request": REQUEST, "seeds": SEEDS, "messages": len(rows),
           "near_dup_kept": int(len(keep)), "mappings": []}
    for name, sens in mappings():
        y = labels_for(rows, sens)
        rec = {"mapping": name,
               "categories": sorted("%d.%d" % c for c in sens),
               "sensitive": int(y.sum()),
               "positive_rate": round(float(y.mean()), 4)}
        gap, f_rand, f_grp, se = leakage_gap(tx_k, Xv_k, y[keep], g_k)
        rec.update(leakage_gap=None if gap is None else round(gap, 4),
                   leakage_gap_se=None if gap is None else round(se, 4),
                   f1_random=None if gap is None else round(f_rand, 4),
                   f1_grouped=None if gap is None else round(f_grp, 4),
                   p_one_sided=(None if gap is None or se <= 0
                                else round(normal_sf(gap / se), 4)),
                   reverses=None if gap is None else bool(gap - 2 * se > 0))
        if y.sum() >= MIN_POSITIVES:
            auto, leak = automation(texts, y, groups)
            rec.update(auto_share=auto, leak=leak,
                       contract_held=bool(leak <= REQUEST))
        else:
            rec.update(auto_share=None, leak=None, contract_held=None,
                       note="too few positives to set a policy honestly")
        out["mappings"].append(rec)
        print("%-42s %6d %6.1f%% %9s %7s %9s %7s"
              % (name[:42], y.sum(), 100 * y.mean(),
                 "-" if gap is None else "%+.4f" % gap,
                 "-" if gap is None else "%.4f" % (2 * se),
                 "-" if rec["auto_share"] is None
                 else "%.1f%%" % (100 * rec["auto_share"]),
                 "-" if rec["leak"] is None else "%.3f" % rec["leak"]),
              flush=True)

    gaps = [r["leakage_gap"] for r in out["mappings"]
            if r["leakage_gap"] is not None]
    autos = [r["auto_share"] for r in out["mappings"]
             if r["auto_share"] is not None]
    held = [r["contract_held"] for r in out["mappings"]
            if r["contract_held"] is not None]
    worst_gap = max(gaps)
    worst_auto = max(autos)
    worst_name = [r["mapping"] for r in out["mappings"]
                  if r["leakage_gap"] == worst_gap][0]
    worst_n = [r["sensitive"] for r in out["mappings"]
               if r["leakage_gap"] == worst_gap][0]
    auto_name = [r["mapping"] for r in out["mappings"]
                 if r["auto_share"] == worst_auto][0]
    # A gap is only a reversal if it is positive by more than two standard
    # errors. Counting sign alone would report noise on a 47-positive mapping
    # as a refutation of the paper. And seventeen tests get seventeen chances,
    # so the uncorrected count is reported against what chance predicts, with
    # a Holm-corrected count beside it.
    real = [r for r in out["mappings"] if r.get("reverses")]
    ALPHA1 = normal_sf(2.0)
    tested = [r for r in out["mappings"] if r.get("p_one_sided") is not None]
    rej = holm([r["p_one_sided"] for r in tested])
    survives = [r for r, k in zip(tested, rej) if k]
    expected = len(tested) * ALPHA1
    p_any = 1 - (1 - ALPHA1) ** len(tested)
    print()
    print("=== A. does the leakage effect survive remapping ===", flush=True)
    print("  published: -0.021 F1 (strict labels, 20 seeds)", flush=True)
    print("  across %d mappings: %+.4f to %+.4f, median %+.4f"
          % (len(gaps), min(gaps), max(gaps), float(np.median(gaps))),
          flush=True)
    print("  largest positive: %s at %+.4f" % (worst_name, worst_gap),
          flush=True)
    print("  mappings where the gap is positive by more than 2 SE: %d"
          % len(real), flush=True)
    print("    chance alone predicts %.2f of %d, and gives at least one "
          "%.0f%% of the time"
          % (expected, len(tested), 100 * p_any), flush=True)
    print("    surviving Holm correction across the %d tests: %d"
          % (len(tested), len(survives)), flush=True)
    for r in real:
        print("     %s  %+.4f +- %.4f  (%d sensitive)"
              % (r["mapping"], r["leakage_gap"], 2 * r["leakage_gap_se"],
                 r["sensitive"]), flush=True)
    if not real:
        vA = ("the thread-disjoint split never scores reliably higher: the "
              "largest positive gap is %+.4f (%s, %d sensitive) and it is "
              "within noise. The paper's central claim is not an artefact of "
              "our seven categories" % (worst_gap, worst_name, worst_n))
    elif not survives:
        vA = ("%d of %d mappings flag uncorrected, which is what chance "
              "predicts (%.2f expected, at least one %.0f%% of the time), and "
              "NONE survives Holm correction. The effect holds; the flag at "
              "%s is not evidence against it"
              % (len(real), len(tested), expected, 100 * p_any,
                 "; ".join("%s (%d sensitive)"
                           % (r["mapping"], r["sensitive"]) for r in real)))
    elif len(survives) <= 2:
        vA = ("the effect reverses under %d of %d mappings after Holm "
              "correction (%s) - it holds broadly but is not "
              "mapping-independent, and the paper should name where it fails"
              % (len(survives), len(tested),
                 "; ".join("%s, %d sensitive"
                           % (r["mapping"], r["sensitive"])
                           for r in survives)))
    else:
        vA = ("THE EFFECT DOES NOT SURVIVE REMAPPING: it reverses under %d "
              "of %d mappings even after Holm correction. The central claim "
              "is a property of our seven categories, not of the corpus"
              % (len(survives), len(tested)))
    print("  " + vA, flush=True)

    print()
    print("=== B. does the automation ceiling survive remapping ===",
          flush=True)
    print("  published: 5.5%% automated at a 5%% request", flush=True)
    print("  across %d mappings: %.1f%% to %.1f%%, median %.1f%%"
          % (len(autos), 100 * min(autos), 100 * max(autos),
             100 * float(np.median(autos))), flush=True)
    print("  contracts held: %d of %d" % (sum(held), len(held)), flush=True)
    if worst_auto <= 0.25:
        vB = ("no defensible mapping clears more than %.1f%% of traffic (%s), "
              "so the ceiling is a property of the task and not of our label "
              "choice" % (100 * worst_auto, auto_name))
    elif worst_auto <= 0.50:
        vB = ("one mapping reaches %.1f%% (%s) against our 5.5%% - the "
              "ceiling is real but its height depends on where the "
              "sensitivity line is "
              "drawn, and the paper should say so"
              % (100 * worst_auto, auto_name))
    else:
        vB = ("THE CEILING IS A PROPERTY OF OUR LABELS: %s automates "
              "%.1f%% of "
              "the same corpus under the same contract. 5.5%% is a fact about "
              "our seven categories" % (auto_name, 100 * worst_auto))
    print("  " + vB, flush=True)

    out["reversals_beyond_noise"] = [r["mapping"] for r in real]
    out["reversals_after_holm"] = [r["mapping"] for r in survives]
    out["expected_spurious_flags"] = round(expected, 3)
    out["p_at_least_one_by_chance"] = round(p_any, 3)
    out["verdict_leakage"] = vA
    out["verdict_automation"] = vB
    out["leakage_range"] = [round(min(gaps), 4), round(max(gaps), 4)]
    out["automation_range"] = [min(autos), max(autos)]
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print()
    print("written %s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
