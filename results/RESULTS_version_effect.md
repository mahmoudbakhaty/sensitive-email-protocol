# What the scikit-learn version alone changes

Same code, same corpus, same seed. Only the library differs:
**1.8.0** against **1.9.1**.

GroupKFold gained stable sorting in 1.9.0 (PR #28464), so the two versions assign threads to folds differently. Rungs R0-R2 use a random message split and are not grouped, so they are the control: they should not move.

## Leakage ladder, strict labels

| Rung | Control | F1 (1.8.0) | F1 (1.9.1) | change |
|---|---|---|---|---|
| R0 | none (duplicates kept, random split) | 0.3704 | 0.3704 | +0.0000 |
| R1 | + exact deduplication | 0.3571 | 0.3571 | +0.0000 |
| R2 | + near-duplicate removal | 0.3762 | 0.3762 | +0.0000 |
| R3 | + thread-disjoint split (full protocol) | 0.3073 | 0.3264 | **+0.0191 ** |

- **Repeated cross-validation, pooled F1**: 0.3375 -> 0.3375 (+0.0000)
- **Permutation null, p-value**: 0.0020 -> 0.0020 (+0.0000)
- **Permutation null, observed F1**: 0.3073 -> 0.3264 (+0.0191)

## Leakage ladder, broad labels

| Rung | Control | F1 (1.8.0) | F1 (1.9.1) | change |
|---|---|---|---|---|
| R0 | none (duplicates kept, random split) | 0.5246 | 0.5246 | +0.0000 |
| R1 | + exact deduplication | 0.5226 | 0.5226 | +0.0000 |
| R2 | + near-duplicate removal | 0.5365 | 0.5365 | +0.0000 |
| R3 | + thread-disjoint split (full protocol) | 0.5172 | 0.5117 | **-0.0055 ** |

- **Repeated cross-validation, pooled F1**: 0.5144 -> 0.5144 (+0.0000)
- **Permutation null, p-value**: 0.0020 -> 0.0020 (+0.0000)
- **Permutation null, observed F1**: 0.5172 -> 0.5117 (-0.0055)

## Annotator agreement reference

This is computed from the annotation counts and never touches a splitter, so it must be identical. If it is not, something other than the splitter changed.

- agreement_f1: 0.7440 vs 0.7440 - identical
- cohens_kappa: 0.6618 vs 0.6618 - identical


---

## What this establishes

**The effect is isolated.** Six quantities act as controls here: rungs R0, R1
and R2 on both label sets use a random message split and never touch a grouped
splitter. All six are identical to four decimal places. The annotator agreement
figures, computed from annotation counts alone, are identical too. Only the two
grouped rungs move.

Nothing else in the pipeline is version-sensitive. The paper previously reported
one number moving by 0.024 and inferred a cause; this is a controlled experiment
with a control group, and it confirms the inference.

**The direction is not consistent.** Strict labels move up by 0.0191, broad
labels move down by 0.0055. Neither version is systematically optimistic. What
changes is which threads land in which fold, and that is a lottery, not a bias.

**Repeating the cross-validation removes the sensitivity completely.** This is
the finding worth the most:

| Quantity | 1.8.0 | 1.9.1 | change |
|---|---|---|---|
| Single deterministic partition, strict | 0.3073 | 0.3264 | +0.0191 |
| Mean of 20 seeded partitions, strict | 0.3375 | 0.3375 | **0.0000** |
| Single deterministic partition, broad | 0.5172 | 0.5117 | -0.0055 |
| Mean of 20 seeded partitions, broad | 0.5144 | 0.5144 | **0.0000** |

The repeated runs use `StratifiedGroupKFold` with explicit seeds rather than the
deterministic `GroupKFold` whose sorting changed, so they are unaffected by the
change and reproduce exactly across both environments.

### The protocol gains a remedy, not just a warning

The paper's current advice is defensive: pin the versions, publish the fold
fingerprint so a reader can detect a mismatch. That remains worth doing. But it
tells a reader how to *notice* the problem rather than how to *avoid* it.

The measurement above supports a stronger recommendation:

> Do not report a single deterministic partition. Average over several seeded
> partitions and report the spread. A result obtained this way did not move at
> all under a library change that shifted the single-partition score by 0.019.

That is testable advice with a test behind it, and it costs a few minutes of
CPU on a corpus this size.

### One caveat, stated rather than buried

`GroupKFold` and `StratifiedGroupKFold` are different splitters - the second
balances labels across folds as well as keeping groups intact - so the seeded
mean is not a like-for-like replacement for the deterministic number, and it
sits higher (0.3375 against 0.3073 and 0.3264). The claim here is only about
*stability across library versions*, which is what the two identical columns
show. Whether the stratified variant should also become the protocol's default
is a separate question this run does not settle.
