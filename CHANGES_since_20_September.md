# What changed since the version sent on 20 September

Nine items. Nothing in the earlier version was withdrawn. Four claims that
rested on assertion now rest on measurement, a defect in the benchmark was
found and repaired, three factual errors were corrected, every external claim
was checked for a citation, and the related work now states plainly which part
of the argument is not ours.

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
| References | 38 | 51 |
| Tables | 6 | 11 |
| Figures | 2 | 3 |

Every figure the paper prints is traceable to a released record; the artifact
audit reports no exceptions. The repository is at
`github.com/mahmoudbakhaty/sensitive-email-protocol`.
