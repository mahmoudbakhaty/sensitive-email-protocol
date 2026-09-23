# -*- coding: utf-8 -*-
"""Generate the change note from the result records.

The previous change note was written by hand and went stale twice in one day:
once when the leakage ladder was re-measured and once when a benchmark bug was
fixed and every grouped figure moved. Both times a number in the note
contradicted the paper it was describing. The note is derived here for the same
reason the paper's tables now are.
"""
import io
import json
import os

D = os.path.join(os.path.expanduser("~"), "Downloads")
OUT = os.path.join(D, "CHANGES_since_20_September.md")


def load(name):
    return json.load(io.open(os.path.join(D, name), encoding="utf-8"))


R = load("RESULTS_FINAL.json")
L = load("RESULTS_LLM.json")
RBS = load("RESULTS_ROBERTA_SEEDS.json")
LAD = load("RESULTS_ladder_v2.json")
THR = load("RESULTS_threading_v2.json")
AGR = load("RESULTS_agreement_metrics.json")
PRM = load("RESULTS_perm_thread.json")


def f(key, field="f1"):
    return R[key][field]


def llm(labels, shots, field="f1"):
    k = "Qwen2.5-7B_%sshot_%s" % ("zero" if shots == 0 else "few", labels)
    return L[k][field]


DOC = """# What changed since the version sent on 20 September

Nine items. Nothing in the earlier version was withdrawn. Four claims that
rested on assertion now rest on measurement, a defect in the benchmark was
found and repaired, three factual errors were corrected, every external claim
was checked for a citation, and the related work now states plainly which part
of the argument is not ours.

---

## 1. A benchmark bug, found and fixed

The expression reading the Subject header used a whitespace class that crosses
the newline. On a message whose subject is empty it captured the following
header instead. Forty-three messages, 3.1%% of the corpus, took a thread key
from the wrong line, and thirty-four of them shared one key and therefore
always fell in the same fold.

| | Before | After |
|---|---|---|
| Threads | 1,069 | %(threads)d |
| Fold fingerprint | 50b3daba1a99ae32 | %(fp)s |
| LogReg, strict, grouped | 0.3456 | %(lr_s).4f |
| LinearSVM, strict, grouped | 0.2685 | %(svm_s).4f |
| RoBERTa, strict, grouped | 0.3587 | %(rb_s).4f |

The size is the part worth noting. Logistic regression moved by 0.005, which
is what was first reported and would have been a comfortable footnote. The
encoder moved by 0.051 and the linear support vector machine by 0.045. A
defect touching 3%% of the corpus moved individual models by an order of
magnitude more than it moved the baseline.

**Every figure in the paper now comes from the corrected build**, and Tables
III to V, Table IX and both result figures read from the run records rather
than carrying hand-typed numbers. The build fails if the thread count or the
fingerprint in the text disagrees with the record. Hand-typed numbers are why
the paper kept the pre-bug figures after the benchmark was rebuilt.

## 2. The limitation about large language models is now a result

The earlier version recorded, as a limitation, that no instruction-tuned model
had been evaluated. The published evidence points both ways, so it was run.
Qwen2.5-7B-Instruct, same folds, with in-context examples drawn only from the
training threads of each fold.

| Method, strict labels | F1 | ROC-AUC |
|---|---|---|
| RoBERTa, fine-tuned | %(rb_s).3f | %(rb_roc).3f |
| Qwen2.5-7B, zero-shot | %(llm_s).3f | %(llm_roc).3f |
| Logistic regression | %(lr_s).3f | %(lr_roc).3f |
| LinearSVM | %(svm_s).3f | %(svm_roc).3f |
| Annotator reference | 0.744 | - |

It lands in the same band. On threshold-free discrimination it comes **last**,
below both classical models - the opposite of what model size would predict.
Few-shot prompting hurts on both label sets: F1 falls from %(llm_s).3f to
%(llm_s6).3f on the strict labels and from %(llm_b).3f to %(llm_b6).3f on the
broad.

The difficulty belongs to the task, not to our choice of model. This was the
objection the paper was most exposed to, and the answer was not predictable.

## 3. The encoder's own figure is the least stable in the paper

Running the encoder over %(nseeds)d seeded partitions instead of one gives
F1 = %(rbs_mean).4f +/- %(rbs_sd).4f on the strict labels, range
[%(rbs_lo).4f, %(rbs_hi).4f]. **The single-partition figure, %(rb_s).4f, lies
above that entire range.** On the broad labels the two agree to four decimals,
so this is not a general instability but one concentrated where the positive
class is scarcest. Recorded in Section VII-F and in the limitations.

## 4. The leakage claim is decomposed - and the first decomposition was wrong

Comparing an ungrouped rung against a grouped one changes three things at
once: thread disjointness, the class balance of the folds (GroupKFold does not
stratify; on this corpus the test positive rate ranges over 0.048 against 0.003
under StratifiedKFold), and the luck of one partition. Only the first is
leakage. Every rung is now the mean of twenty seeded partitions with the
grouped rung stratified.

| Control | Strict | Broad |
|---|---|---|
| Exact deduplication | %(d_dedup_s)+.4f | %(d_dedup_b)+.4f |
| Near-duplicate removal | %(d_near_s)+.4f | %(d_near_b)+.4f |
| Thread-disjoint splitting | %(d_thread_s)+.4f | %(d_thread_b)+.4f |
| Total | %(d_total_s)+.4f | %(d_total_b)+.4f |

Thread-disjoint splitting is the larger route on both label sets, by about half
as much again on the strict labels and an order of magnitude on the broad.
Near-duplicate removal costs nothing - it raises F1 and MCC while discarding
148 messages - so the paper recommends dropping a control its own protocol
contains.

## 5. The lower-bound claim about thread reconstruction is now demonstrated

The headers that would thread this corpus properly are absent: 3 of 1,382
messages carry In-Reply-To and none carries References. Subject-line matching
is the only available method, not a lazy one.

A finer grouping is still possible - subject plus shared correspondents plus a
thirty-day window, which can only split a subject group. It raises the thread
count from %(thr_subj)d to %(thr_ref)d and changes the measured leakage cost
from %(thr_cost_s0).4f to %(thr_cost_s1).4f on the strict labels. The coarse
grouping was understating the leakage by about a factor of three, which is the
direction the lower-bound claim predicts.

## 6. The paper can now show its results are not chance

Five folds cannot produce a p below 0.0625 under a Wilcoxon test. A label
permutation test gives p = %(perm_p).3f on both label sets, and no shuffled run
came near the observed score. Repeating it with whole thread label-blocks
exchanged, so within-thread coherence survives, moves the null mean by less
than 0.005 and leaves p unchanged.

## 7. The agreement coefficient understates the agreement

Cohen's kappa reads low under a skewed class distribution - the kappa paradox -
and Gwet's AC1 is recommended instead. Our positive class is 18.1%%.

| Measure | Estimate | 95%% CI |
|---|---|---|
| Raw observed agreement | %(po).3f | [%(po_lo).3f, %(po_hi).3f] |
| Cohen's kappa | %(kap).3f | [%(kap_lo).3f, %(kap_hi).3f] |
| **Gwet's AC1** | **%(ac1).3f** | [%(ac1_lo).3f, %(ac1_hi).3f] |
| Annotator F1 | %(af1).3f | [%(af1_lo).3f, %(af1_hi).3f] |

AC1 is higher than kappa by %(ac1_d).3f, so describing this judgement as
heavily contested on kappa alone overstates the case, and the paper says so.
What does not move is the quantity the argument rests on: the annotator F1
carries no chance correction, and neither does the fact that two trained
annotators settled 172 of 1,382 messages differently.

## 8. Corrections and citations

- **"Human ceiling" renamed** to an annotator agreement reference point, after
  Richie et al. and Boguslav and Cohen, with an interval.
- **A misattributed citation** in the Arabic section pointed at the
  scikit-learn changelog where it meant AraBERT.
- **Fig. 1's caption** said 5,394 terms occur in at least five messages; 5,394
  is the count at ten, and 8,984 occur in at least five.
- **Every external claim was checked for a citation.** Three were real: the
  threshold rule now cites Cawley and Talbot; the introduction names the
  figures it compares against; and a sentence saying the measurement is "what
  is missing in this subfield" stopped being true once Zainab et al. were
  cited, and was narrowed.
- **An independent consistency checker** (Fazekas and Kovacs) passes all
  fourteen reported records.

## 9. Related work: what is ours and what is not

**Zainab et al. (5 August 2026) make the general claim that leakage inflates
sensitivity classification six weeks before our draft did**, and on 2 September
the same group reported 91.23%% F1 on that corpus. The paper says so plainly
and states what it adds: human annotators, measured agreement, and
conversational structure. **Roth (April 2026) decomposes leakage across 2,047
tabular datasets**, so that idea is not ours either; our claim is the narrower
one, that his four classes do not cover conversational structure.

The contrast is the sharpest positioning argument the paper has. The same
leakage-controlled methodology yields above 0.91 on cables and %(llm_s).3f
here. A cable's classification stamp is an administrative fact; our label is a
judgement two trained people applied differently to 172 messages.

---

## Summary of the document

|  | Before | Now |
|---|---|---|
| Length, IEEE | 10 pp | 15 pp |
| Contributions | 6 | 9 |
| References | 38 | 51 |
| Tables | 6 | 11 |
| Figures | 2 | 3 |

Every figure the paper prints is traceable to a released record; the artifact
audit reports no exceptions. The repository is at
`github.com/mahmoudbakhaty/sensitive-email-protocol`.
"""

vals = {
    "threads": R["dataset"]["threads"], "fp": R["fold_fingerprint"],
    "lr_s": f("LogReg_grouped_strict"), "svm_s": f("LinearSVM_grouped_strict"),
    "rb_s": f("transformer_grouped_strict"),
    "lr_roc": f("LogReg_grouped_strict", "roc_auc"),
    "svm_roc": f("LinearSVM_grouped_strict", "roc_auc"),
    "rb_roc": f("transformer_grouped_strict", "roc_auc"),
    "llm_s": llm("strict", 0), "llm_s6": llm("strict", 6),
    "llm_b": llm("broad", 0), "llm_b6": llm("broad", 6),
    "llm_roc": llm("strict", 0, "roc_auc"),
    "nseeds": len(RBS["seeds"]),
    "rbs_mean": RBS["strict"]["f1_mean"], "rbs_sd": RBS["strict"]["f1_sd"],
    "rbs_lo": RBS["strict"]["f1_range"][0],
    "rbs_hi": RBS["strict"]["f1_range"][1],
    "d_dedup_s": LAD["strict"]["deltas"]["exact_deduplication"],
    "d_near_s": LAD["strict"]["deltas"]["near_duplicate_removal"],
    "d_thread_s": LAD["strict"]["deltas"]["thread_disjoint_split"],
    "d_total_s": LAD["strict"]["deltas"]["total_R0_to_R3"],
    "d_dedup_b": LAD["broad"]["deltas"]["exact_deduplication"],
    "d_near_b": LAD["broad"]["deltas"]["near_duplicate_removal"],
    "d_thread_b": LAD["broad"]["deltas"]["thread_disjoint_split"],
    "d_total_b": LAD["broad"]["deltas"]["total_R0_to_R3"],
    "thr_subj": THR["subject_threads"], "thr_ref": THR["refined_threads"],
    "thr_cost_s0": abs(THR["strict"]["subject"]["leakage_cost"]),
    "thr_cost_s1": abs(THR["strict"]["refined"]["leakage_cost"]),
    "perm_p": PRM["strict"]["message_level"]["p_value"],
    "po": AGR["observed_agreement"]["estimate"],
    "po_lo": AGR["observed_agreement"]["ci95"][0],
    "po_hi": AGR["observed_agreement"]["ci95"][1],
    "kap": AGR["cohens_kappa"]["estimate"],
    "kap_lo": AGR["cohens_kappa"]["ci95"][0],
    "kap_hi": AGR["cohens_kappa"]["ci95"][1],
    "ac1": AGR["gwets_ac1"]["estimate"],
    "ac1_lo": AGR["gwets_ac1"]["ci95"][0],
    "ac1_hi": AGR["gwets_ac1"]["ci95"][1],
    "af1": AGR["annotator_f1"]["estimate"],
    "af1_lo": AGR["annotator_f1"]["ci95"][0],
    "af1_hi": AGR["annotator_f1"]["ci95"][1],
    "ac1_d": abs(AGR["ac1_minus_kappa"]),
}

io.open(OUT, "w", encoding="utf-8").write(DOC % vals)
print("written %s" % OUT)
print("threads %d | fingerprint %s" % (vals["threads"], vals["fp"]))
