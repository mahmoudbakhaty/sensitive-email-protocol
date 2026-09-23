# What changed since the version sent on 20 September

Eleven items. Nothing in the earlier version was withdrawn. Four claims that
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
header instead. Forty-three messages, 3.1% of the corpus, took a thread key
from the wrong line, and thirty-four of them shared one key and therefore
always fell in the same fold.

| | Before | After |
|---|---|---|
| Threads | 1,069 | 1103 |
| Fold fingerprint | 50b3daba1a99ae32 | 63e3aea5c3d37629 |
| LogReg, strict, grouped | 0.3456 | 0.3505 |
| LinearSVM, strict, grouped | 0.2685 | 0.2235 |
| RoBERTa, strict, grouped | 0.3587 | 0.4099 |

The size is the part worth noting. Logistic regression moved by 0.005, which
is what was first reported and would have been a comfortable footnote. The
encoder moved by 0.051 and the linear support vector machine by 0.045. A
defect touching 3% of the corpus moved individual models by an order of
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
| LinearSVM, strict | 0.2235 | 0.3814 | +0.1579 |
| LinearSVM, broad | 0.4667 | 0.5500 | +0.0833 |
| LogReg, broad | 0.5095 | 0.5438 | +0.0343 |
| LogReg, strict | 0.3505 | 0.3683 | +0.0178 |

The support vector machine was recalling 0.160 of the positive class at
its default operating point and 0.640 at a selected one. It was not a
weak model; it was a model at the wrong operating point.

We claim nothing new in the principle. That a threshold choice can confound a
comparison of F1 scores is textbook - which is why ROC-AUC is recommended for
comparing classifiers, and the paper already reports it. That an experimental
choice can reorder a published ranking is the finding of a line running from
Ferrari Dacrema et al. (RecSys 2019) to Musgrave et al. (ECCV 2020), both now
cited.

What is ours is narrower and less flattering: Section IV-D states that the
threshold must be chosen on data the reported score does not come from, and we
did not apply that rule to every row of our own tables. **0.158 F1 is
what that cost on one model - larger than any difference between two models
anywhere in the paper.** New Table VI-A.

The reordering is not cosmetic. Under the old arrangement the encoder led on
both label sets. Under one treatment it leads on the strict labels, 0.410
against 0.381, and comes **last** on the broad: LinearSVM 0.550,
LogReg 0.544, character n-grams 0.538, encoder 0.530.

## 2. The limitation about large language models is now a result

The earlier version recorded, as a limitation, that no instruction-tuned model
had been evaluated. The published evidence points both ways, so it was run.
Qwen2.5-7B-Instruct, same folds, with in-context examples drawn only from the
training threads of each fold.

| Method, strict labels | F1 | ROC-AUC |
|---|---|---|
| RoBERTa, fine-tuned | 0.410 | 0.710 |
| Qwen2.5-7B, zero-shot | 0.401 | 0.696 |
| Logistic regression | 0.350 | 0.711 |
| LinearSVM | 0.224 | 0.714 |
| Annotator reference | 0.744 | - |

It lands in the same band. On threshold-free discrimination it comes **last**,
below both classical models - the opposite of what model size would predict.
Few-shot prompting hurts on both label sets: F1 falls from 0.401 to
0.371 on the strict labels and from 0.526 to 0.489 on the
broad.

The difficulty belongs to the task, not to our choice of model. This was the
objection the paper was most exposed to, and the answer was not predictable.

## 3. The encoder's own figure is the least stable in the paper

Running the encoder over 5 seeded partitions instead of one gives
F1 = 0.3761 +/- 0.0151 on the strict labels, range
[0.3490, 0.3929]. **The single-partition figure, 0.4099, lies
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
| Exact deduplication | -0.0150 | -0.0010 |
| Near-duplicate removal | +0.0120 | +0.0069 |
| Thread-disjoint splitting | -0.0207 | -0.0147 |
| Total | -0.0237 | -0.0088 |

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
count from 1103 to 1160 and changes the measured leakage cost
from 0.0059 to 0.0219 on the strict labels. The coarse
grouping was understating the leakage by about a factor of three, which is the
direction the lower-bound claim predicts.

## 6. The paper can now show its results are not chance

Five folds cannot produce a p below 0.0625 under a Wilcoxon test. A label
permutation test gives p = 0.002 on both label sets, and no shuffled run
came near the observed score. Repeating it with whole thread label-blocks
exchanged, so within-thread coherence survives, moves the null mean by less
than 0.005 and leaves p unchanged.

## 7. The agreement coefficient understates the agreement

Cohen's kappa reads low under a skewed class distribution - the kappa paradox -
and Gwet's AC1 is recommended instead. Our positive class is 18.1%.

| Measure | Estimate | 95% CI |
|---|---|---|
| Raw observed agreement | 0.875 | [0.856, 0.894] |
| Cohen's kappa | 0.662 | [0.610, 0.710] |
| **Gwet's AC1** | **0.803** | [0.768, 0.834] |
| Annotator F1 | 0.744 | [0.703, 0.782] |

AC1 is higher than kappa by 0.141, so describing this judgement as
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
  paper itself surveys are 98.2% accuracy, 99% accuracy and F1 = 0.92, and a
  search found no published 1.000 on context-dependent sensitivity. The only
  1.000 we can document is our own earlier dataset, which the paper already
  says and whose cause it already gives - thirty duplicates and complete
  keyword separability. Projecting our own number onto other people's work is
  the same defect as a mis-attributed citation, and the argument never needed
  it: what the annotator reference point contradicts is a near-perfect score,
  whatever its third decimal. The four sentences are now anchored on the
  figures actually cited.
- **Every reference was fetched, not just counted.** Earlier rounds checked
  that each of the 54 entries carries an identifier. This round resolved all
  63 targets - 22 DOIs through the Crossref registry, 22 arXiv identifiers
  through their abstract pages, 19 direct links - and compared the title and
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
broad labels it gains +0.0236 F1 over the better component; on the
strict labels the ranking improves but the gain does not survive thresholding.

A control matters here. Fusing two redundant components gains
+0.0049, which is the procedural optimism of cross-fitted stacking,
measured rather than assumed. Roughly half the apparent gain would have
happened from any fusion.

## 9. Related work: what is ours and what is not

**Zainab et al. (5 August 2026) make the general claim that leakage inflates
sensitivity classification six weeks before our draft did**, and on 2 September
the same group reported 91.23% F1 on that corpus. The paper says so plainly
and states what it adds: human annotators, measured agreement, and
conversational structure. **Roth (April 2026) decomposes leakage across 2,047
tabular datasets**, so that idea is not ours either; our claim is the narrower
one, that his four classes do not cover conversational structure.

The contrast is the sharpest positioning argument the paper has. The same
leakage-controlled methodology yields above 0.91 on cables and 0.401
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
