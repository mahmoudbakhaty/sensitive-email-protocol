# When a model declines, is it declining what the annotators argued over?

The benchmark carries 172 messages that two trained annotators settled
differently. Selective prediction gives them a second use beyond measuring how
contested the judgement is: if the messages a model declines are
disproportionately those 172, its uncertainty is tracking human disagreement
rather than its own ignorance.

Enrichment is the contested rate among declined messages over the 12.4% base
rate, with a hypergeometric p-value. The decline sets come from the
validation-chosen cut-offs in `scripts/selective.py`; no coverage is chosen
here.

| model | best enrichment | at coverage | p |
|---|---|---|---|
| logistic regression | 1.03× | 0.30 | 0.32 |
| linear SVM | 1.05× | 0.50 | 0.27 |
| character n-grams | 1.00× | 0.40 | 0.53 |
| fine-tuned encoder | **1.19×** | 0.40 | **0.0013** |

## What it says

**The three classical models sit at 1.00× throughout.** Their uncertainty has
no relationship to which messages the annotators argued over. Whatever makes
abstention work for them - and it does work, `RESULTS_SELECTIVE.md` has the
curve - it is not contestedness.

**The encoder's declined set is enriched, and modestly.** 14.8% contested
against a 12.4% base: a real tendency, not a large one. It is the only one of
the four whose uncertainty relates to human disagreement at all.

## Multiple testing, because 28 tests were run

Seven coverages by four models. The Bonferroni threshold is 0.05/28 = 0.00179,
and exactly one result clears it: the encoder at coverage 0.40, p = 0.00132.
The encoder's other coverages (p = 0.0099, 0.025, 0.029) do not, and are
reported as part of a consistent pattern rather than as findings on their own.

This is stated because a paper arguing that near-perfect published scores
measure dataset construction cannot then quote an uncorrected p of 0.03 from a
grid of 28.

## What it does not say

It does not say the encoder knows when humans disagree. 1.19× is a tendency,
and the arm that produced it is the weaker one: the encoder cannot be
retrained without a GPU, so its cut-offs are cross-fitted from released
out-of-fold scores. The classical arms are retrained by the script.

It also does not explain why abstention helps. That question is still open,
and the answer for three of the four models is evidently not "it declines the
contested ones".

## Reproduce

```
python scripts/declined.py        # ~4 minutes on CPU
```
