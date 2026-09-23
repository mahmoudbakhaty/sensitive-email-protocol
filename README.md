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
| **Human ceiling** (second annotator as a classifier) | **F1 0.744**, MCC 0.662 |
| Strongest single term of 8,984 | `'in'`, F1 0.321 |
| Best method, strict | RoBERTa F1 0.359, 95% CI [0.309, 0.408] |
| Best method, broad | RoBERTa F1 0.545, 95% CI [0.503, 0.584] |

Nothing here is near-perfect, every confidence interval overlaps its
neighbours, and that is the result rather than a disappointment. Quote
every figure against both bounds: the trivial floor below and the human
ceiling above. The best model reaches 48% of what a second annotator
achieves, and a published F1 of 1.000 on this task would claim to exceed
the agreement of the people who defined the labels.

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
| `results/RESULTS_arabic_v2.json` | The Arabic run, raw. Tables IX and X are generated from this file directly, so they cannot drift from it. |
| `results/RESULTS_arabic_v2.md` | The same, annotated with what the section should and should not claim. |
| `results/FINDING_sklearn_groupkfold.md` | The full record of the version-dependence investigation behind Section VII-F. |
| `results/superseded/` | Earlier runs. Kept because Section VII-F compares against them; **do not quote these as results.** |
| `paper_refs_v2.py` | The 54 references plus `VERIFIED`, recording how each was resolved. Every entry carries a DOI, an arXiv id or a stable URL. |
| `fig_v2_*.png` | The three paper figures. |
| `scripts/fig_ladder.py` | Builds Fig. 3 straight from the results file, so the figure cannot drift from the numbers. |
| `scripts/agreement_metrics.py` | Cohen's kappa beside Gwet's AC1 and the raw agreement, after the kappa paradox was raised against us. |
| `scripts/perm_thread.py` | The permutation null repeated at thread level, to check the message-level version was not flattered by clustering. |
| `results/RESULTS_agreement_metrics.json` | Four agreement measures with intervals. |
| `results/RESULTS_perm_thread.json` | Both permutation nulls. |
| `scripts/uniform_protocol.py` | Every non-GPU model under the encoder's treatment, so Tables III and IV compare like with like. |
| `scripts/fair_comparison.py` | Each model under both treatments, which is how the gap was measured. |
| `scripts/char_baseline.py` | Character n-grams under the paper's protocol. |
| `results/RESULTS_UNIFORM.json` | Source of Tables III and IV's classical rows. |
| `results/RESULTS_FAIR_COMPARISON.json` | The treatment gap, per model. |
| `scripts/roberta_seeds.py` | The encoder over five seeded partitions. Its single-partition strict figure lies above the whole seeded range. |
| `scripts/threading_v2.py` | A finer thread reconstruction - subject plus shared correspondents plus a 30-day window - and what it does to the measured leakage cost. |
| `scripts/thread_check.py` | How many messages carry In-Reply-To or References. Three of 1,382 and none. |
| `scripts/score_consistency.py` | Runs Fazekas and Kovacs's checker over our own reported scores. |
| `results/RESULTS_ROBERTA_SEEDS.json` | The seeded encoder runs. |
| `results/RESULTS_threading_v2.json` | The finer-threading comparison. |
| `scripts/ladder_v2.py` | The corrected leakage ladder: one control per rung, twenty seeded partitions, fold class balance held fixed. **Supersedes the ladder in `strengthen.py`.** |
| `scripts/ladder_stratified.py` | Separates the thread-grouping cost from the class-marginal term that plain GroupKFold introduces. |
| `results/RESULTS_ladder_v2.json` | The corrected ladder. Source of Table VII and Fig. 3. |
| `results/RESULTS_ladder_stratified.json` | The leakage / class-imbalance separation. |
| `scripts/strengthen.py` | The leakage ladder, the permutation null, the repeated cross-validation and the agreement interval. Runs on CPU. |
| `scripts/compare_versions.py` | Builds the scikit-learn 1.8.0 against 1.9.1 comparison from the two result files. |
| `scripts/llm_protocol.py` | An instruction-tuned LLM under the same protocol. Produced Table IX. |
| `scripts/test_llm_protocol.py` | Checks the fold bookkeeping with the model stubbed out: no thread crosses a fold, no in-context example comes from a test thread. |
| `scripts/test_judge_path.py` | Checks the model call itself on a tiny real model. The first Kaggle run died in exactly the line this covers, which the stubbed test could not reach. |
| `results/RESULTS_FINAL.json` | The run behind Tables III to V, raw. Environment, fold fingerprint, per-fold dispersion and bootstrap intervals. |
| `results/PREDICTIONS_FINAL.json` | Per-message predictions and scores for all ten classical configurations, so any figure can be recomputed. |
| `results/RESULTS_LLM.json` | Qwen2.5-7B-Instruct under the protocol, four runs. Source of Table IX. |
| `results/RESULTS_strengthen_sklearn191.json` | The strengthening run under the pinned library. The figures in Sections VII-G and VII-H come from this file. |
| `results/RESULTS_strengthen_sklearn180.json` | The same suite under 1.8.0, with every other library held fixed. |
| `results/RESULTS_strengthen.md` | The four measurements, written up, with what each does and does not establish. |
| `results/RESULTS_version_effect.md` | What the library version alone changes, with six ungrouped controls that do not move. |
| `quoted_figures.py` | Every figure quoted from another paper, with the page it was read from. These cannot be regenerated here. |
| `LITERATURE_SURVEY_2026-09-21.md` | Where the field stands and where this work sits in it, with each figure attributed. |

**A benchmark bug, found on 22 September and fixed.** The expression reading
the Subject header used a whitespace class that crosses the newline, so on a
message with an empty subject it captured the following header. Forty-three
messages, 3.1% of the corpus, took a thread key from the wrong line, and
thirty-four of them shared one key and therefore always fell in the same fold.
Threads go from 1,069 to 1,103 and the fingerprint from `50b3daba1a99ae32` to
`63e3aea5c3d37629`. Logistic regression moved by 0.005, the encoder by 0.051.
Every figure in the paper and in this repository comes from the corrected
build.

**The definitive run, re-executed.** The original session's record was never
retrieved, so the run was repeated from `scripts/final_run.py` on the released
pin, months later and on different hardware. The fold fingerprint reproduced
exactly (`50b3daba1a99ae32`) and every deterministic figure reproduced to four
decimal places. The GPU-trained encoder did not reproduce bit for bit: it is
0.006 F1 low on the broad labels, because GPU training is not deterministic
across hardware even under a fixed seed. Both records are now here.

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
`VERIFIED` in `paper_refs_v2.py` records the source used for each of the 38.
