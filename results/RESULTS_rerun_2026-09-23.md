# Do the released scripts reproduce the released numbers?

Every CPU-only experiment script was re-run on 23 September 2026 from the
repository, against the corpus rebuilt from the freshly downloaded archive,
and its output compared key by key with the record shipped in `results/`.

The local environment is scikit-learn **1.8.0**; the published runs used
**1.9.1**. That difference is the subject of Section VII-F of the paper, so
this re-run is also a second, independent measurement of it.

| script | keys compared | numbers that differ |
|---|---|---|
| `agreement_metrics.py` | 14 | **0** |
| `threading_v2.py` | 27 | **0** |
| `ladder_v2.py` | 16 | **0** |
| `ladder_stratified.py` | 29 | 14 |
| `char_baseline.py` | 43 | 32 |
| `uniform_protocol.py` | 110 | 96 |

In every case the only other difference is the recorded `scikit_learn`
version.

## The split is not arbitrary

The three that reproduce exactly are the three that do not depend on a single
`GroupKFold` partition:

* `agreement_metrics.py` computes kappa, Gwet's AC1 and the annotator F1 from
  the two annotators' labels. No splitting is involved at all - observed
  agreement 0.8755, kappa 0.6618, AC1 0.8031, annotator F1 0.7440, every
  bootstrap interval identical to four decimals.
* `threading_v2.py` reconstructs threads. No model, no folds.
* `ladder_v2.py` **averages over twenty seeded partitions per rung**. It is the
  one modelling script that does, and it is the one modelling script whose
  every number survives a library change.

The three that differ all read a single deterministic partition:
`char_baseline.py`, `uniform_protocol.py`, and the `*_plain` rungs of
`ladder_stratified.py`. On the broad labels the character n-gram F1 moves
0.5378 to 0.5315 and its five fold values are a different set entirely.

This is the paper's Section VII-F claim, reproduced on six scripts it does not
discuss: **the version sensitivity is in the partition, and averaging over
seeded partitions removes it.** The paper already recommends that; this is
evidence for the recommendation rather than a new finding.

## What it means for the published tables

Nothing in the paper changes. Tables III and IV are all-one-session figures
from the pinned environment, as Section VII-F requires and as the environment
block in each result file records. A reader on a different scikit-learn should
expect the single-partition numbers to move by a few thousandths and the
seeded-partition numbers not to move at all, and can now check that
expectation against this table.

## One hazard this exposed, and closed

`char_baseline.py` and `uniform_protocol.py` wrote straight to `~/Downloads`
and ignored `RESULTS_DIR`, so re-running them into a scratch directory
overwrote the working copies the paper build reads - with numbers from the
wrong library version. The repository's copies made it recoverable and the
files were restored and verified identical. Those scripts now write through
`scripts/io_paths.py` like the rest, so `RESULTS_DIR` actually contains a
re-run.
