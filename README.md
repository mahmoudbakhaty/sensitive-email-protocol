# Context-Dependent Sensitive Data Classification in Enterprise Email: A Leakage-Audited Benchmark and Protocol

Artifacts for the paper by M. E. A. Bakhaty, A. M. Sauber and M. B. Rizk,
Department of Mathematics and Computer Science, Faculty of Science,
Menoufia University.

The contribution is an evaluation protocol and a benchmark, not a classifier.
These artifacts exist so both can be checked rather than taken on trust.

## Headline numbers

| | |
|---|---|
| Messages / threads | 1,382 / 1,069 |
| Sensitive, strict (both annotators) / broad (either) | 250 (18.1%) / 422 (30.5%) |
| **Cohen's kappa** | **0.662** (observed agreement 0.876) |
| Trivial all-positive floor | F1 0.306 strict, 0.468 broad |
| Strongest single term of 8,984 | `'in'`, F1 0.321 |
| Best method, strict | RoBERTa F1 0.359, 95% CI [0.309, 0.408] |
| Best method, broad | RoBERTa F1 0.545, 95% CI [0.503, 0.584] |

Nothing here is near-perfect, every confidence interval overlaps its
neighbours, and that is the result rather than a disappointment. Quote every
figure against the trivial floor.

## Reproduce it

```bash
# the run every number in the paper comes from: ~70 min on a GPU
# paste KAGGLE_FINAL.txt into one Colab or Kaggle cell and run it first,
# before any other cell, with the accelerator set to GPU
```

`KAGGLE_FINAL.txt` (= `scripts/final_run.py`) pins scikit-learn, records the
full environment, computes the fold fingerprint, and produces every table,
interval and paired test in one session. It refuses to start without a GPU and
refuses to continue if the pin did not take.

`KAGGLE_ARABIC_V2.txt` (= `scripts/arabic_v2.py`) is the cross-lingual
extension: ~35 min given a cached translation, ~105 min without one.

## What is here

| Path | What it is |
|---|---|
| `KAGGLE_FINAL.txt`, `scripts/final_run.py` | The definitive run. Identical content, one for pasting and one for diffing. |
| `KAGGLE_ARABIC_V2.txt`, `scripts/arabic_v2.py` | The Arabic extension, AraBERT with its documented preprocessing. |
| `scripts/addendum.py` | Standalone bootstrap intervals and paired tests, superseded by `final_run.py` but kept because the reproducibility section refers to it. |
| `scripts/figs_v2.py` | Regenerates both paper figures from the data. |
| `results/RESULTS_FINAL.md` | Every figure of Tables I-VI, with the environment and fingerprint that produced them. |
| `results/RESULTS_arabic_v2.json` | The Arabic run, raw. Tables VII and VIII are generated from this file directly, so they cannot drift from it. |
| `results/RESULTS_arabic_v2.md` | The same, annotated with what the section should and should not claim. |
| `results/FINDING_sklearn_groupkfold.md` | The full record of the version-dependence investigation behind Section VII-E. |
| `results/superseded/` | Earlier runs. Kept because Section VII-E compares against them; **do not quote these as results.** |
| `paper_refs_v2.py` | The 35 references plus `VERIFIED`, recording how each was resolved. |
| `fig_v2_*.png` | The two paper figures. |

**Not yet included:** `RESULTS_FINAL.json`, the raw record of the definitive
run. `results/RESULTS_FINAL.md` carries every figure quoted in the paper, but
the machine-readable file and the per-item predictions are still to be added.

## The data is not redistributed, it is rebuilt

We do not ship the corpus. The script downloads the public Enron subset
carrying two independent annotators' judgements

    https://bailando.berkeley.edu/enron/enron_with_categories.tar.gz

and derives the benchmark deterministically: seven category pairs map to a
binary label, administrative-only messages and bodies under thirty words are
dropped, exact duplicates removed by hashing the whitespace-normalised
lower-cased body. Anyone running it gets the same 1,382 messages over 1,069
threads, and Cohen's kappa = 0.662 from the counts 250 / 172 / 960.

## The four controls

1. **Thread-grouped splitting.** Grouped by normalised subject after stripping
   reply and forward prefixes, split with `GroupKFold`. Disjointness is
   asserted at runtime, not assumed.
2. **Duplicate control.** Exact duplicates removed before any split is drawn.
3. **Keyword-shortcut auditing.** Every term in at least five messages scored
   as a one-rule classifier. Best: `'in'`, F1 0.321 against a floor of 0.306 -
   the ten strongest are all function words.
4. **Threshold discipline.** Chosen on a validation split of whole threads
   carved from each training fold, applied once to the test fold. Never on
   test data.

## Check the fingerprint before comparing anything

    fold-assignment fingerprint: 50b3daba1a99ae32

If your run prints a different value you do not have our partition, and no
score comparison is meaningful. This is not hypothetical. The same code, data
and seed produced thread-grouped F1 = 0.3220 under scikit-learn 1.6.1 and
1.8.0 (fingerprint 3556420f17e7e4fa) and 0.3458 under 1.9.1 (fingerprint
50b3daba1a99ae32), because 1.9.0 changed `GroupKFold` to stable sorting. The
paper pins 1.9.1. `results/FINDING_sklearn_groupkfold.md` has the full
account, including the fact that we first read the evidence backwards.

## Known limitations

Stated in the paper and repeated here so nobody is surprised.

* Labels are the Enron subset's original category judgements mapped to our
  definition. The mapping is ours and is a source of construct error.
* One corpus, one organisation, one era. Generalisation to contemporary
  enterprise email is unestablished.
* Threads are approximated by normalised subject line, which both over- and
  under-merges, so the measured cost of thread leakage is a lower bound.
* No method is statistically separable from another. Paired over identical
  folds, the encoder beats logistic regression in 4 of 5 folds on the strict
  labels and 3 of 5 on the broad, and with n = 5 a Wilcoxon test cannot return
  below p = 0.0625 regardless. Make no ranking claim from these numbers.
* On the strict labels the encoder leads on F1 and loses MCC, ROC-AUC and
  PR-AUC to a bag-of-words baseline.
* RoBERTa and AraBERT were fine-tuned at a competent default configuration,
  not a searched one.
* The Arabic results are a transfer experiment on machine-translated text, not
  an Arabic benchmark. 3.1% of translations were materially damaged, and
  translationese, pretraining-corpus differences and translation quality are
  confounded in the English-Arabic gap.

## Reference verification

Auditing our own reference list found one work attributed to the wrong
authors, one incorrect publication year and one truncated title. All corrected;
`VERIFIED` in `paper_refs_v2.py` records the source used for each of the 35.
