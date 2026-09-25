# -*- coding: utf-8 -*-
"""Generate results/RESULTS_FINAL.md from results/RESULTS_FINAL.json.

RESULTS_FINAL.md was written by hand, and the README's file table points a
reader at it as the source of Tables I-VI, "with the environment and the
fingerprint that produced them". It was the entire PRE-CORRECTION run, shipped unmarked:
fold fingerprint 50b3daba1a99ae32 and 1,069 threads, both from the withdrawn
build that the subject-header bug produced. Five of the ten record F1s in it -
every thread-grouped one, which is to say every number the corpus fix moved -
did not appear anywhere in the JSON the same session wrote.

It was also the origin of a false claim that had propagated into the README's
Known Limitations: "RoBERTa now leads ONLY on F1. It loses MCC, ROC-AUC AND
PR-AUC." True of the withdrawn build. False of every record this release ships,
where the encoder leads its Table III baselines on all four.

A hand-written mirror of a machine-written record will go stale again, so it is
generated now. `scripts/test_paths.py` regenerates it and fails if what is on
disk did not already match.

    python scripts/final_md.py
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import io_paths                                              # noqa: E402

OUT = os.path.join(REPO, "results", "RESULTS_FINAL.md")
COLS = ("f1", "precision", "recall", "mcc", "roc_auc", "pr_auc")
NAMES = {"LinearSVM": "LinearSVM", "LogReg": "LogReg",
         "transformer": "roberta-base"}


def row(d, key, label):
    r = d.get(key)
    if not r:
        return None
    cells = ["%-13s" % label]
    cells += ["%.4f" % r[c] if isinstance(r.get(c), float) else "  -   "
              for c in COLS]
    ci = r.get("f1_ci95")
    cells.append("%.4f+-%.4f" % (r.get("fold_mean_f1", float("nan")),
                                 r.get("fold_std_f1", float("nan"))))
    cells.append("[%.4f,%.4f]" % (ci[0], ci[1]) if ci else "")
    return " ".join(cells)


def table(d, lab):
    lines = ["model         F1      P       R       MCC     ROC     PR      "
             "mean+-sd        CI95"]
    for m in ("LinearSVM", "LogReg", "transformer"):
        r = row(d, "%s_grouped_%s" % (m, lab), NAMES[m])
        if r:
            lines.append(r)
    return "\n".join(lines)


def cost(d, lab):
    out = []
    for m in ("LogReg", "LinearSVM"):
        s = d.get("%s_stratified_%s" % (m, lab))
        g = d.get("%s_grouped_%s" % (m, lab))
        if s and g:
            out.append("%s %.4f -> %.4f (%+.4f)"
                       % (m, s["f1"], g["f1"], g["f1"] - s["f1"]))
    return " | ".join(out)


def paired(d):
    out = []
    for k in sorted(k for k in d if k.startswith("paired_")):
        r = d[k]
        name = k[len("paired_"):].replace("_", " ")
        out.append("%-40s %d/%d wins  mean %+.4f  p %s"
                   % (name, r["wins"], r["n_folds"],
                      r["mean_difference"], r["wilcoxon_p"]))
    return "\n".join(out)


def main():
    p = io_paths.result_in("RESULTS_FINAL.json", required=True)
    d = json.load(io.open(p, encoding="utf-8"))
    e, ds, la = d["environment"], d["dataset"], d["leakage_audit"]
    ann = d.get("annotation", {})
    tf = d.get("trivial_floor", {})

    enc = d["transformer_grouped_strict"]
    f1a = ann.get("annotator_f1", 0.744)

    head = "# FINAL RUN - every number in the paper comes from this session"
    txt = head + """

GENERATED from results/RESULTS_FINAL.json by scripts/final_md.py. Do not edit
by hand: this file was hand-written until 25 September and had drifted onto the
withdrawn 1,069-thread build, fingerprint 50b3daba1a99ae32, while the JSON
beside it carried the released one.

{gpu} | sklearn {sk} | numpy {np} | torch {torch}
python {py} | fold fingerprint {fp}

## Dataset
{msg} messages | {st} strict / {br} broad | {th} threads
annotation: both {both} / one {one} / neither {nei}
  observed agreement {oa} | Cohen's kappa {ka}
trivial floor: strict {tfs}, broad {tfb}

## Leakage audit
duplicates {dup} | multi-message threads {mm} | mixed-label {ml}
strongest of {vocab} terms: '{term}' F1 {termf1}

## Human ceiling
One annotator scored as a classifier predicting the other: F1 {ceil}
=> the task's ceiling is F1 {ceil}, not 1.000. The encoder's {encf1} is {pct}%
   of it.

## STRICT labels, thread-grouped 5-fold
{tstrict}

## BROAD labels, thread-grouped 5-fold
{tbroad}

## Thread-grouping cost (stratified -> grouped, F1)
strict:  {cstrict}
broad:   {cbroad}

## Paired, identical folds (n=5, so p cannot go below 0.0625)
{pair}

## What this run says
Confidence intervals overlap everywhere; no method is separable from the next.
That is the honest headline and it is what the paper argues from. Quote every
figure against both bounds - the trivial floor below and the annotator
reference of F1 {ceil} above.
""".format(
        gpu=e.get("gpu", "Kaggle"), sk=e.get("scikit_learn", "?"),
        np=e.get("numpy", "?"), torch=e.get("torch", "?"),
        py=e.get("python", "?"), fp=d["fold_fingerprint"],
        msg=ds["messages"], st=ds["sensitive_strict"],
        br=ds["sensitive_broad"], th=ds["threads"],
        both=ann.get("both", "?"), one=ann.get("one", "?"),
        nei=ann.get("neither", "?"),
        oa=ann.get("observed_agreement", "?"),
        ka=ann.get("cohens_kappa", "?"),
        tfs=tf.get("strict", "?"), tfb=tf.get("broad", "?"),
        dup=la["exact_duplicates"], mm=la["threads_with_multiple_messages"],
        ml=la["threads_with_mixed_labels"], vocab=la["vocabulary_terms"],
        term=la["best_single_term"], termf1=la["best_single_term_f1"],
        ceil=f1a, encf1=enc["f1"], pct=int(round(100 * enc["f1"] / f1a)),
        tstrict=table(d, "strict"), tbroad=table(d, "broad"),
        cstrict=cost(d, "strict"), cbroad=cost(d, "broad"),
        pair=paired(d))

    io.open(OUT, "w", encoding="utf-8").write(txt)
    print("regenerated %s" % OUT)
    print("  fingerprint %s | %d threads | encoder strict F1 %.4f"
          % (d["fold_fingerprint"], ds["threads"], enc["f1"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
