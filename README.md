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
| `scripts/verify_references.py` | Fetches every reference target and compares the returned title and authors with the printed ones. Exits non-zero on any failure. |
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

## The pasteable notebook, and which build produced what

`KAGGLE_FINAL.txt` is what the README tells a reader to paste into Kaggle, and
it had drifted from `scripts/final_run.py`. The one difference was the whole
difference: the Subject-header correction was applied to the script and not to
the notebook, so anyone following the reproduction instructions rebuilt the
**buggy** benchmark - 1069 thread keys instead of 1103 - and got numbers that
do not match the paper. The notebook is regenerated from the script, and
`scripts/test_notebooks.py` fails if they part again.

Two files are deliberately kept as they ran, with the pre-correction regex,
because they are the record of runs that produced released numbers:

| file | what it produced | grouping |
|---|---|---|
| `scripts/colab_v2.py` | Section VIII and Table VI | 1069 thread keys |
| `KAGGLE_ARABIC_V2.txt` | `results/RESULTS_arabic_v2.json` | 1069 thread keys |

`colab_v2.py` was imported by three released scripts and was not in the
repository at all, so `arabic_v2.py` could not even be imported. Both files now
carry a header saying what they are; the test fails if the header goes.

**Section VIII is not comparable with Tables III to V.** Its English and Arabic
arms share the 1069-key grouping, which is what the comparison between them
requires, but the main tables use 1103. The paper says so in Section VIII
rather than leaving it here.

The corrected regex is now spelled the same way everywhere, with the escape
rather than a literal tab: a tab in source is one editor setting away from
becoming spaces, and `[ ]*` would quietly bring back a variant of the bug.
Rebuilding after the change still gives 1382 messages over 1103 threads.

## Do these scripts reproduce these numbers?

Every CPU-only experiment script was re-run from this repository on 23
September 2026, on the corpus rebuilt from the freshly downloaded archive, and
compared key by key with the record it ships. The local library was
scikit-learn 1.8.0; the published runs used 1.9.1.

| script | keys | numbers that differ |
|---|---|---|
| `agreement_metrics.py` | 14 | **0** |
| `threading_v2.py` | 27 | **0** |
| `ladder_v2.py` | 16 | **0** |
| `ladder_stratified.py` | 29 | 14 |
| `char_baseline.py` | 43 | 32 |
| `uniform_protocol.py` | 110 | 96 |

The split is the paper's Section VII-F claim, reproduced on six scripts it does
not discuss. The three that reproduce exactly are the three that do not rest on
a single `GroupKFold` partition - two involve no splitting at all, and
`ladder_v2.py` averages over twenty seeded partitions per rung. The three that
differ all read one deterministic partition. **The version sensitivity is in
the partition, and averaging over seeded partitions removes it.**

Full table and what it does and does not mean:
`results/RESULTS_rerun_2026-09-23.md`.

## Can you actually run this?

Twenty absolute paths into the author's home directory were spread across
seventeen scripts: six pointing at the corpus, fourteen at result files. The
write-only ones failed elsewhere with a missing-directory error. The reading
ones were worse - `score_consistency.py` and `fig_ladder.py` are named above as
checks a reader can run, and they read files by a path no reader has, while a
copy of each sits in `results/`.

Every script now resolves through two small modules:

| | order |
|---|---|
| `scripts/corpus_path.py` | `$ENRON_DIR`, the usual locations, the author's original path last |
| `scripts/io_paths.py` | reads: `$RESULTS_DIR`, `<repo>/results`, `~/Downloads` - writes: `$RESULTS_DIR` else `~/Downloads` |

```
python scripts/test_paths.py    # fails if any script names one machine again
```

Checked by cloning this repository to an empty directory with an empty home:
`score_consistency.py` and `fig_ladder.py` both run, both read the clone's own
`results/`, and `fig_ladder.py` regenerates Fig. 3 with the published numbers.

## Did you get the same corpus we did?

The benchmark is not redistributed; it is rebuilt from a public archive. Until
now nothing let a reader check that their archive was ours - no checksum, and
no corpus-level digest - and six scripts, `strengthen.py` among them, carried
an absolute path into a temp directory on the author's machine, so they ran
there and nowhere else. Both are fixed.

```
python scripts/verify_corpus.py        # downloads, checks, extracts, rebuilds
```

It reports three things and they are not interchangeable:

| | what it is | expected |
|---|---|---|
| **ARCHIVE** | sha256 of the tarball as served | `08625500ab4c032f...` , 4,523,350 bytes |
| **CORPUS** | digest of the rebuilt messages and labels | `493f90a5df5b8f40` , 1382 messages / 1103 threads / 250 strict / 422 broad |
| **FOLDS** | the published `fold_fingerprint` | `63e3aea5c3d37629` **under scikit-learn 1.9.1 only** |

The corpus digest is the one that answers "same data?". It is taken over the
messages themselves, order-independently, so it does not move with the library
version or the operating system - `sorted(glob(...))` orders Windows and POSIX
paths differently, which would otherwise change it for no reason.

**A different fold fingerprint is not a reproduction failure.** `GroupKFold`'s
assignment changed between scikit-learn versions, which is precisely what
Section VII-F of the paper measures; on 1.8.0 the same corpus gives
`a456e576074f733a`. Only the corpus block makes the script exit non-zero. It
was tested by tampering with one message: the digest moves and the check
fails.

Every script finds the corpus through `scripts/corpus_path.py`: `$ENRON_DIR`
first, then the usual locations, then the author's original path last so the
published runs still reproduce where they were produced.

## What a model is worth when it is allowed to decline

Every method lands between F1 0.368 and 0.397 at full coverage with
overlapping intervals, which is why the paper ranks none of them. That asks how
often a model is right when forced to answer about every message, and a
deployed filter is never in that position.

| model | full coverage | ~30% coverage | z vs random | AURC |
|---|---|---|---|---|
| logistic regression | 0.3683 | 0.4772 | +3.4 | 0.586 |
| character n-grams | 0.3675 | 0.4571 | +2.8 | 0.584 |
| linear SVM | 0.3814 | 0.4880 | +3.3 | 0.561 |
| fine-tuned encoder | 0.3971 | **0.5592** | **+5.1** | **0.525** |

The abstention rule is chosen on validation threads and never on test, the
achieved coverage is reported rather than the target, and every point is run
again with the same number of messages declined at random 200 times - random
abstention leaves F1 flat, so the gain is not a shrinking denominator.

It does not establish a ranking: the intervals overlap, as they do everywhere
in this paper. `results/RESULTS_SELECTIVE.md` has the table, the caveats, and
a correction of an earlier version of this analysis that reached the wrong
conclusion.

```
python scripts/selective.py        # ~4 minutes on CPU
```

## The filter as a system, and what it can promise

`scripts/filter_system.py` is the framework as something you can call: a
message in, one of ALLOW / ESCALATE / BLOCK out. Its thresholds are a contract
rather than an accuracy target - *at most this share of sensitive messages may
be auto-allowed* - because a leak and an interruption do not cost the same.

At the default contract, out of fold: 15.1% of traffic handled automatically,
4.0% of sensitive messages auto-allowed against a 5% promise, nothing
auto-blocked, 4.8% error among the automatic decisions.

The contract did not hold at first, and the three reasons are in
`results/RESULTS_FILTER_SYSTEM.md`. The one worth naming here: the binomial
bound assumes independent draws, and messages in a thread are not independent
- the premise of this whole protocol. Counting threads instead of messages
moved the block-side overshoot from 6.3x the promise to 3.1x.

**The guarantee is one-sided.** "No more than X% of sensitive messages pass
automatically" holds at every request measured. "No more than Y% of harmless
messages are blocked automatically" does not.

**And one promise this benchmark cannot support at all.** At a 1% request the
system automates nothing, correctly: promising a 1% leak rate at 90%
confidence needs 230 threads carrying a sensitive message and validation has
69. With 1,382 messages the tightest promise available is about 5%, and no
model changes that.

```
python scripts/filter_system.py
python scripts/policy_transfer.py
```

## Why the block promise breaks, and the calibrator that fixes it

The system keeps one half of its contract. The leak bound holds at every
request; the false-block bound overshoots by 1.2x to 3.1x. Same thresholds,
same validation threads, same bound - only one holds.

The obvious explanation is wrong. Measured across folds, the block threshold
is **four times more stable** than the leak one (0.063 against 0.252 standard
deviations), so distribution shift is not the cause. It was the hypothesis and
it is refuted.

The cause is ties. On one test fold the risk score takes **62 distinct values
over 277 messages**, and at the block threshold **28 of 225 harmless messages
sit on exactly that value** - against 0 at the leak threshold. So "at most 5%
above this cut" has no solution: a hair down and 12% go over, a hair up and 0%
do. The bound was not failing to estimate a quantile; there was no quantile to
estimate.

Isotonic regression did it. It is a step function, chosen for calibration
quality, and it destroys the resolution a rate guarantee needs. Platt scaling
is strictly monotone:

| calibrator | distinct values | on the cut | Brier | block contract |
|---|---|---|---|---|
| isotonic | 48.6 | 4.8% | **0.2268** | **breaks at every request** |
| Platt | **275.6** | **0.0%** | 0.2449 | **holds at every request** |

The price is about 8% worse calibration and roughly half the automation.
`results/RESULTS_WHY_BLOCK_FAILS.md` has the full tables.

**This generalises.** Anyone putting a rate guarantee on an isotonically
calibrated score meets the same wall, and it does not announce itself: the
score looks like a probability, the bound looks sound, and the contract fails
in one direction only. The diagnosis is one line - count how many items share
the threshold's exact value.

```
python scripts/why_block_fails.py
python scripts/calibrator_ties.py
```

## Does it decline what the annotators argued over?

The benchmark carries 172 messages two trained annotators settled differently.
If the messages a model declines are disproportionately those, its uncertainty
tracks human disagreement rather than its own ignorance.

| model | best enrichment | p |
|---|---|---|
| logistic regression | 1.03x | 0.32 |
| linear SVM | 1.05x | 0.27 |
| character n-grams | 1.00x | 0.53 |
| fine-tuned encoder | **1.19x** | **0.0013** |

The three classical models sit at 1.00x: whatever makes abstention work for
them, it is not contestedness. The encoder's declined set is enriched, modestly
- 14.8% contested against a 12.4% base.

Twenty-eight tests were run (seven coverages, four models), so the threshold
that matters is Bonferroni's 0.00179, and exactly one result clears it. Stated
in `results/RESULTS_DECLINED.md` rather than left for a reader to work out,
because a paper arguing that near-perfect published scores measure dataset
construction cannot then quote an uncorrected p of 0.03 out of a grid of 28.

```
python scripts/declined.py         # ~4 minutes on CPU
python scripts/fig_selective.py    # the risk-coverage figure
```

## The two control files, and why their numbers are higher

`results/RESULTS_HYBRID_CONTROL.json` and `results/RESULTS_HYBRID_CPU.json`
record a strict F1 of 0.4324 for an encoder-plus-character-n-gram fusion. The
paper says nothing we tried on the strict labels exceeds 0.410. Both are
correct, and the difference is the point of the experiments:

* The paper's figures select the decision threshold **only on held-out
  validation threads**. These two select it on the training fold, which is not
  leakage into test but is a different, easier procedure - the same treatment
  asymmetry that Section VII-D of the paper measures at up to 0.158 F1.
* They stack **pooled out-of-fold scores**, which carries a known optimism.
  `hybrid_control.py` exists to measure that optimism rather than assume it
  away: fusing two deliberately redundant components - logistic regression and
  a linear SVM over the same features - gains +0.0049 from the procedure alone.

So these are controls on the fusion machinery, not results under the protocol,
and they are not comparable with Tables III and IV. The fusion and pipeline
claims were cut from the paper on the supervisor's review; these files are
released because the measurements behind that decision should be inspectable.

The framework the thesis title promises, `scripts/hybrid_framework.py`, does
select every threshold on validation threads. It is written and tested but
**has not been run**: it needs a GPU and the weekly quota is exhausted. No
number from it appears anywhere.

## Reference verification

Auditing our own reference list found one work attributed to the wrong
authors, one incorrect publication year and one truncated title. All corrected;
`VERIFIED` in `paper_refs_v2.py` records the source used for each of the 54.

A later round showed that recording a source is not the same as resolving one.
`scripts/verify_references.py` fetches every target - DOIs through the Crossref
registry, arXiv ids through their abstract pages, plain URLs directly - and
compares the title and author list that come back with the ones printed. It
found two defects that the identifier count had passed: one entry naming an
author who is not on the paper while omitting one who is, and one link that had
started returning 404 after a publisher restructured its site. Run it yourself;
it exits non-zero if any entry fails, and writes
`results/RESULTS_reference_resolution.json`.

Two hosts refuse scripted clients (OpenReview answers with a browser challenge,
ai.meta.com with a 400). Those entries are flagged `bot-walled` and verified by
hand. Where such a link was the only identifier, it was replaced with one a
reviewer's script can also resolve.
