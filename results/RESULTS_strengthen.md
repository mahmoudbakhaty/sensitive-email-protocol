# Four measurements that make the leakage claim quantitative

> **This suite ran on 21 September, the day before the subject-header bug was
> found.** It is the withdrawn 1,069-thread build. Which parts that affects:
>
> | part | build-dependent? | superseded by |
> |---|---|---|
> | ladder R0/R1/R2 (random splits, dedup) | no - no thread keys involved | - |
> | ladder R3 (thread-disjoint) | **yes** | `RESULTS_ladder_v2.json` (23 Sep) |
> | repeated CV over thread-to-fold assignments | **yes** | `RESULTS_ladder_v2.json` |
> | thread-level permutation null | **yes** | `RESULTS_perm_thread.json` (23 Sep) |
> | annotator agreement | no - independent of threading | `RESULTS_agreement_metrics.json` adds AC1 |
>
> Every build-dependent figure here has a post-fix replacement. Quote those.
> `scripts/ladder_v2.py` supersedes this ladder, as the README says.


Runs of 21 September 2026, Python 3.14.0, numpy 2.4.2, scipy 1.17.1,
pandas 3.0.1. Source: `scratchpad/strengthen.py`.

The suite was run twice, under **scikit-learn 1.9.1** and under **1.8.0**, with
every other library held fixed so that scikit-learn is the only difference.
**1.9.1 is the reference throughout**, because it is the environment the
published figures come from; 1.8.0 is shown beside it wherever the two differ.
Raw figures: `RESULTS_strengthen_sklearn191.json` and
`RESULTS_strengthen_sklearn180.json`. The comparison itself is in
`RESULTS_version_effect.md`.

---

## 1. The leakage ladder

The paper's existing evidence is one contrast: a random message split against a
thread-disjoint one. That bundles every contamination route into a single
number. Adding one control at a time separates them.

### Strict labels

| Rung | Control added | n | F1 (1.9.1) | MCC | Floor | F1 (1.8.0) |
|---|---|---|---|---|---|---|
| R0 | none - duplicates kept, random split | 1481 | 0.3704 | 0.2302 | 0.3026 | 0.3704 |
| R1 | + exact deduplication | 1382 | 0.3571 | 0.2242 | 0.3064 | 0.3571 |
| R2 | + near-duplicate removal (148 dropped) | 1234 | 0.3762 | 0.2581 | 0.3073 | 0.3762 |
| R3 | + thread-disjoint split (full protocol) | 1234 | **0.3264** | 0.2091 | 0.3073 | 0.3073 |

### Broad labels

| Rung | Control added | n | F1 (1.9.1) | MCC | Floor | F1 (1.8.0) |
|---|---|---|---|---|---|---|
| R0 | none - duplicates kept, random split | 1481 | 0.5246 | 0.3067 | 0.4653 | 0.5246 |
| R1 | + exact deduplication | 1382 | 0.5226 | 0.3069 | 0.4678 | 0.5226 |
| R2 | + near-duplicate removal (148 dropped) | 1234 | 0.5365 | 0.3330 | 0.4680 | 0.5365 |
| R3 | + thread-disjoint split (full protocol) | 1234 | **0.5117** | 0.2922 | 0.4680 | 0.5172 |

Only the R3 row differs between the two library versions; every ungrouped rung
is identical to four decimals.

### What the decomposition says

Cost of each control, on 1.9.1:

| Control | Strict | Broad |
|---|---|---|
| Exact deduplication | -0.0133 | -0.0020 |
| Near-duplicate removal | **+0.0191** | **+0.0139** |
| Thread-disjoint split | **-0.0498** | **-0.0248** |
| Total, R0 to R3 | -0.0440 | -0.0129 |

**Conversation structure is the route that matters; duplication is not.** The
thread-disjoint split accounts for more than the whole net drop on both label
sets. Exact deduplication costs little. Removing near-duplicates does not cost
anything at all - it *raises* F1 on both label sets, so near-duplicates were not
inflating this corpus.

This sharpens the protocol's advice from a general warning into a priority: if
only one control can be afforded, group by conversation.

**Near-duplicate removal can be dropped from the protocol for this corpus.** It
discards 148 messages, 11% of the benchmark, and buys no reduction in inflation.
Keeping a control that costs data and removes no bias is not conservatism, it is
waste. This is a recommendation against our own protocol's current contents, and
it is what the measurement supports.

**The inflation is larger for the harder label.** Total R0 to R3: 0.0440 on
strict against 0.0129 on broad, a factor of about three. The rarer and more
contested the target, the more the model leans on structure it should not have.
A paper reporting only a broad, common label would see very little leakage
effect and could reasonably conclude there was none.

**It must be said plainly that this is a modest effect.** CTSCAN reports a 69%
relative drop and CASSED collapses from 0.996 to 0.349. Ours is 0.044 absolute
on the strict label, about 12% relative. The honest summary is not that leakage
destroys the result, but that *the score was never high to begin with*, and the
part of it that was inflated came from conversation structure specifically.

---

## 2. The empirical null

Five folds cannot produce a p below 0.0625 under a Wilcoxon signed-rank test,
which is above every conventional threshold. A label permutation test
(Ojala and Garriga, JMLR 11:1833-1863, 2010) holds the class proportion fixed
and destroys the association between text and label, 500 times, through the
same grouped pipeline.

| Label set | Observed F1 | Null mean | Null 95th pct | Null max (500) | p |
|---|---|---|---|---|---|
| strict | 0.3264 | 0.0933 ± 0.0318 | 0.1505 | 0.1914 | **0.002** |
| broad | 0.5117 | 0.2337 ± 0.0323 | 0.2895 | 0.3213 | **0.002** |

Not one permutation out of 500 came near the observed score on either label set:
the best a shuffled-label run managed was 0.1914 against an observed 0.3264.
0.002 is the smallest p this design can return, so it should be read as "below
the resolution of 500 permutations", not as a precise value.

### This resolves an ambiguity that arose in this run

Under scikit-learn 1.8.0 the strict protocol score, 0.3073, came out
**identical to the trivial all-positive floor, 0.3073**. That invites the
reading that the classifier has no signal at all. It is the wrong reading, and
three further measurements on the same model say so:

| Measurement | What it suggests |
|---|---|
| F1 = 0.3073, equal to the floor | no signal |
| MCC = 0.1968 | signal present |
| predicts positive 11.9% of the time, base rate 18.2% | not degenerate |
| p = 0.002 against 500 permutations | signal confirmed |

A direct check confirms the classifier agrees with an all-positive predictor on
only 11.9% of messages. The identical F1 is a coincidence of the metric, not
degeneracy.

**The coincidence is not a stable property, and should not be presented as
one.** Under scikit-learn 1.9.1 - the environment the published figures come
from - the same model on the same data scores 0.3264 against the same floor of
0.3073, comfortably above it. The exact equality belongs to one partition under
one library version.

That is worth reporting for a different reason than the one it first appeared
to serve: the same model, same corpus and same seed sits *on* the trivial floor
under one splitter and *above* it under another. Whether a result looks like
"no better than trivial" can turn on a library version. This is the same
fragility measured in Section 3 and in the version comparison, seen from the
reader's side.

**This is the paper's own argument turned on itself.** F1 on an imbalanced
problem misleads in both directions: it inflates when leakage is present, and it
conceals real signal when the positive class is rare. It must be read beside the
trivial floor and beside MCC, which is what the protocol already requires - and
here is a worked case showing why.

---

## 3. Repeated cross-validation

GroupKFold is deterministic, so the five published fold scores describe one
assignment of threads to folds. Varying the assignment separates the variance of
the method from the luck of a partition. Twenty assignments, same data, same
model:

| Label set | Pooled F1 | Range across 20 partitions |
|---|---|---|
| strict | 0.3375 ± 0.0130 | [0.3118, 0.3665] |
| broad | 0.5144 ± 0.0110 | [0.4954, 0.5388] |

The strict range spans 0.0547 - wider than the entire leakage effect of 0.0440
measured in Section 1. **A single-partition result is not a stable quantity at
this sample size**, and the paper should report the spread rather than one
partition's number. This is the paper's own advice applied to itself.

**Both rows are identical under scikit-learn 1.8.0 and 1.9.1**, to four
decimals, while the single-partition score moves by 0.0191. The repeated runs
use seeded `StratifiedGroupKFold` rather than the deterministic `GroupKFold`
whose sorting changed in 1.9.0, so they are immune to it. That turns the
paper's reproducibility finding from a warning into a remedy; see
`RESULTS_version_effect.md`.

---

## 4. An interval for the annotator agreement reference

The paper leans on a point estimate: one annotator scored as a classifier
against the other reaches F1 = 0.744. A bare number invites more weight than it
can carry. Bootstrapping over threads, 2000 resamples:

| Quantity | Estimate | 95% CI |
|---|---|---|
| Agreement F1 | 0.7440 | **[0.7032, 0.7808]** |
| Cohen's kappa | 0.6618 | **[0.6114, 0.7099]** |

The point estimates reproduce the published 0.744 and 0.662 exactly. The
interval is what is new, and it matters for the reframing that Richie et al.
(BioNLP 2022) force: the quantity is a reference point describing how
contestable the judgement is, not a ceiling, and it is now reported with the
uncertainty a reference point should carry.

---

## What still has to be done

1. ~~The scikit-learn 1.9.1 re-run.~~ Done; see `RESULTS_version_effect.md`.
2. An instruction-tuned LLM under the same protocol. The literature is split -
   Ntwali et al. have GPT-4o rescuing structured-data detection, Antypas et al.
   have zero-shot LLMs losing to fine-tuned models on contested content
   judgement - so the question has to be settled by running it, not by citing.
   Harness written and tested: `scratchpad/llm_protocol.py`.
3. Folding all of this into the paper, with the seven new references.
