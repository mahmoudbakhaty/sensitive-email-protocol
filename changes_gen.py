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
UNI = load("RESULTS_UNIFORM.json")
HYB = load("RESULTS_HYBRID_CPU.json")
CTL = load("RESULTS_HYBRID_CONTROL.json")


# Reference figures are counted from the list, never typed. The note names
# 54 entries and a target count; both moved when [20] gained an arXiv id.
import re as _re
import paper_refs_v2 as _R

_DOI = [r for r in _R.REFS if _re.search(r"doi:\s*10\.", r, _re.I)]
_ARX = [r for r in _R.REFS if _re.search(r"arXiv:\s*\d{4}\.\d{4,5}", r, _re.I)]
_URL = [r for r in _R.REFS if _re.search(r"https?://", r)]
REF_COUNTS = {"n_refs": len(_R.REFS), "n_doi": len(_DOI),
              "n_arxiv": len(_ARX), "n_url": len(_URL),
              "n_targets": len(_DOI) + len(_ARX) + len(_URL)}
assert not [r for r in _R.REFS
            if r not in _DOI and r not in _ARX and r not in _URL],     "a reference carries no identifier"


def uni(model, labels, field="f1"):
    return UNI["%s_uniform_%s" % (model, labels)][field]


def f(key, field="f1"):
    return R[key][field]


def llm(labels, shots, field="f1"):
    k = "Qwen2.5-7B_%sshot_%s" % ("zero" if shots == 0 else "few", labels)
    return L[k][field]


DOC = """# What changed since the version sent on 20 September

%(n_items)s items. Nothing in the earlier version was withdrawn. Four claims that
rested on assertion now rest on measurement; a defect in the benchmark was
found and repaired; the main tables turned out not to be comparing like with
like and were rebuilt; the framework the registered thesis title promises was
built; three factual errors were corrected; every external claim was checked
for a citation; and the related work now states plainly which part of the
argument is not ours.

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

## 1b. Tables III and IV were not comparing like with like

Checking whether a character n-gram baseline could join Table III showed it
could not, because the table's own rows were treated differently. The classical
rows took a fixed operating point and the whole training fold; the encoder row
selected its threshold on held-out threads and therefore trained on seventy per
cent of it. Neither is leakage - a fixed threshold is tuned on nothing - but
the rows were being read against each other.

| Model | Published F1 | One treatment | Change |
|---|---|---|---|
| LinearSVM, strict | %(t_svm_s_p).4f | %(t_svm_s_u).4f | %(t_svm_s_d)+.4f |
| LinearSVM, broad | %(t_svm_b_p).4f | %(t_svm_b_u).4f | %(t_svm_b_d)+.4f |
| LogReg, broad | %(t_lr_b_p).4f | %(t_lr_b_u).4f | %(t_lr_b_d)+.4f |
| LogReg, strict | %(t_lr_s_p).4f | %(t_lr_s_u).4f | %(t_lr_s_d)+.4f |

The support vector machine was recalling %(svm_r_p).3f of the positive class at
its default operating point and %(svm_r_u).3f at a selected one. It was not a
weak model; it was a model at the wrong operating point.

We claim nothing new in the principle. That a threshold choice can confound a
comparison of F1 scores is textbook - which is why ROC-AUC is recommended for
comparing classifiers, and the paper already reports it. That an experimental
choice can reorder a published ranking is the finding of a line running from
Ferrari Dacrema et al. (RecSys 2019) to Musgrave et al. (ECCV 2020), both now
cited.

What is ours is narrower and less flattering: Section IV-D states that the
threshold must be chosen on data the reported score does not come from, and we
did not apply that rule to every row of our own tables. **%(t_max).3f F1 is
what that cost on one model - larger than any difference between two models
anywhere in the paper.** New Table VI-A.

The reordering is not cosmetic. Under the old arrangement the encoder led on
both label sets. Under one treatment it leads on the strict labels, %(rb_s).3f
against %(best_s).3f, and comes **last** on the broad: LinearSVM %(u_svm_b).3f,
LogReg %(u_lr_b).3f, character n-grams %(u_ch_b).3f, encoder %(rb_b).3f.

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
- **One result file was left behind by the benchmark fix, and the paper
  quoted it.** `RESULTS_ladder_stratified.json` was last written on 22
  September, over the old 1069-thread grouping; every other result file was
  regenerated when the Subject-header bug was fixed the next day. It carried
  its own environment block claiming the pinned library, so nothing looked
  wrong. The paper took one number from it as a literal - that GroupKFold
  "lets the test positive rate range over 0.048" - and on the corrected
  benchmark that spread is 0.105, more than double. The claim's direction is
  unchanged and in fact strengthened; the number was wrong. Regenerated, and
  the paper now reads it, with a build-time assertion that the file's recorded
  library matches the definitive run's. Two further constants taken from the
  same file were referenced nowhere and are deleted rather than updated: a
  constant nothing reads is a stale number waiting to be quoted.
- **Table II's thread counts were two runs out of date.** It printed 173
  multi-message threads and 55 with mixed labels. The Subject-header
  correction moved those to 171 and 54; the result record updated and the two
  hand-typed literals in the paper did not. Recomputed from the corrected
  corpus to confirm - 171 and 54 - and the table now reads them from the
  record, like every other figure in it.
- **A table caption named a source its numbers do not come from.** Table X's
  caption said its English figures were from the run of Table III. They were
  not, and had not been since Tables III and IV were rebuilt under one
  treatment: Table X prints LinearSVM at 0.264 where Table III prints 0.381,
  a gap of 0.117 F1. The figures themselves are right - they are the
  cross-lingual run's own English arm, on the same 1069-key grouping as its
  Arabic arm, which is exactly what the comparison between the two needs. The
  caption now says so. Sending a reader to a table the numbers do not come
  from is the same defect as a mis-attributed citation, and the earlier check
  could not catch it: it asked whether each printed number appears somewhere
  in the release, which a number in the wrong row under the wrong caption
  passes. `scripts/audit_table_sources.py` now binds each row to its record.
- **The notebook the reproduction instructions point at still had the
  benchmark bug.** `KAGGLE_FINAL.txt` is what a reader pastes into Kaggle, and
  the Subject-header correction had been applied to the script it is copied
  from but not to the notebook. Anyone following the instructions rebuilt the
  buggy benchmark - 1069 thread keys instead of 1103 - and got numbers that do
  not match the paper. Regenerated, with a test that fails if they part again.
  A module three released scripts import, `colab_v2.py`, was also missing from
  the release entirely.
- **Section VIII rests on the pre-correction grouping, and now says so.** The
  cross-lingual run predates the Subject-header fix and groups the same 1382
  messages into 1069 thread keys. Its English and Arabic arms share that
  grouping, so the comparison it makes is sound; its figures are not comparable
  with Tables III to V, and the section states that rather than leaving a
  reader to discover it.
- **The abstract claimed less scope than it needed.** It said nothing we
  tried on the strict labels exceeds F1 = 0.410; the conclusion says the same
  thing but adds "under validation-only threshold selection", and that clause
  is the one that matters. The released control experiments reach 0.4324,
  with the threshold picked on the training fold rather than on held-out
  validation threads and stacked on pooled out-of-fold scores whose optimism
  the same experiment measures at +0.0049. Nothing is wrong with either
  number, but a reader who opened the artifact would have found the higher one
  with no explanation. The abstract now carries the conclusion's scope, and
  the README names the two files, their figure and why it is not comparable.
- **An F1 of 1.000 was attributed to published work that does not report
  one.** Four sentences read as though the literature reports a perfect score
  on this task. It does not, so far as we can establish: the figures this
  paper itself surveys are 98.2%% accuracy, 99%% accuracy and F1 = 0.92, and a
  search found no published 1.000 on context-dependent sensitivity. The only
  1.000 we can document is our own earlier dataset, which the paper already
  says and whose cause it already gives - thirty duplicates and complete
  keyword separability. Projecting our own number onto other people's work is
  the same defect as a mis-attributed citation, and the argument never needed
  it: what the annotator reference point contradicts is a near-perfect score,
  whatever its third decimal. The four sentences are now anchored on the
  figures actually cited.
- **Every reference was fetched, not just counted.** Earlier rounds checked
  that each of the %(n_refs)s entries carries an identifier. This round resolved all
  %(n_targets)s targets - %(n_doi)s DOIs through the Crossref registry, %(n_arxiv)s arXiv identifiers
  through their abstract pages, %(n_url)s direct links - and compared the title and
  author list that came back with the one printed. Two entries failed.
  Reference [20] named an author who is not on that paper and omitted one who
  is; it is corrected against arXiv:2310.17884 and the ICLR 2024 proceedings.
  Reference [49] pointed at a page that now returns 404 after a publisher
  site restructure, and is re-pointed at the live listing. The OpenReview link
  on [20] was also replaced: OpenReview now answers automated requests with a
  browser challenge, so that link could not be verified by a reviewer's script
  either. Three further entries the checker flagged were artefacts of the
  checker, not defects, and are documented as such in `paper_refs_v2.py`.

## 8b. The framework the thesis title promises is built

The registered title promises a hybrid LLM-based framework. What existed was a
protocol, a benchmark and four models compared under it; the components had
never been combined. `hybrid_framework.py` combines an instruction-tuned model,
a fine-tuned encoder and tf-idf, with every component calibrated on held-out
validation threads, the fusion weights fitted there and the threshold chosen
there. An earlier fusion was cut from the paper for being miscalibrated and
unvalidated; both are addressed rather than repeated. It needs a GPU and the
weekly quota is exhausted, so it is written and tested but not yet run.

The two-component half runs now, cross-fitted from released scores. On the
broad labels it gains %(hyb_gain_b)+.4f F1 over the better component; on the
strict labels the ranking improves but the gain does not survive thresholding.

A control matters here. Fusing two redundant components gains
%(ctl_gain_b)+.4f, which is the procedural optimism of cross-fitted stacking,
measured rather than assumed. Roughly half the apparent gain would have
happened from any fusion.

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
| References | 38 | 54 |
| Tables | 6 | 12 |
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
    "t_svm_s_p": f("LinearSVM_grouped_strict"),
    "t_svm_s_u": uni("LinearSVM", "strict"),
    "t_svm_s_d": uni("LinearSVM", "strict") - f("LinearSVM_grouped_strict"),
    "t_svm_b_p": f("LinearSVM_grouped_broad"),
    "t_svm_b_u": uni("LinearSVM", "broad"),
    "t_svm_b_d": uni("LinearSVM", "broad") - f("LinearSVM_grouped_broad"),
    "t_lr_s_p": f("LogReg_grouped_strict"),
    "t_lr_s_u": uni("LogReg", "strict"),
    "t_lr_s_d": uni("LogReg", "strict") - f("LogReg_grouped_strict"),
    "t_lr_b_p": f("LogReg_grouped_broad"),
    "t_lr_b_u": uni("LogReg", "broad"),
    "t_lr_b_d": uni("LogReg", "broad") - f("LogReg_grouped_broad"),
    "svm_r_p": f("LinearSVM_grouped_strict", "recall"),
    "svm_r_u": uni("LinearSVM", "strict", "recall"),
    "t_max": max(abs(uni(m, l) - f("%s_grouped_%s" % (m, l)))
                 for m in ("LinearSVM", "LogReg")
                 for l in ("strict", "broad")),
    "best_s": max(uni(m, "strict") for m in ("LinearSVM", "LogReg", "CharNgram")),
    "u_svm_b": uni("LinearSVM", "broad"), "u_lr_b": uni("LogReg", "broad"),
    "u_ch_b": uni("CharNgram", "broad"),
    "rb_b": f("transformer_grouped_broad"),
    "hyb_gain_b": HYB["broad"]["gain_over_best_component"],
    "ctl_gain_b": CTL["broad"]["control_gain"],
}

vals.update(REF_COUNTS)
_words = {9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve",
          13: "Thirteen", 14: "Fourteen", 15: "Fifteen"}
_heads = [l for l in DOC.splitlines()
          if l.startswith("## ") and not l.startswith("## Summary")]
vals["n_items"] = _words.get(len(_heads), str(len(_heads)))
io.open(OUT, "w", encoding="utf-8").write(DOC % vals)
print("written %s" % OUT)
print("threads %d | fingerprint %s" % (vals["threads"], vals["fp"]))
