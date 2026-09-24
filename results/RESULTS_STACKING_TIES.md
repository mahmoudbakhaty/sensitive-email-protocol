# Stacking is not why the ties fell the wrong way — and it does the opposite

Two hypotheses for why an isotonic rate contract broke on this benchmark and
held on three public corpora had already been refuted: distribution shift
(`RESULTS_WHY_BLOCK_FAILS.md`) and model weakness
(`RESULTS_TIES_AND_STRENGTH.md`). Before reaching for a third there was one
difference between the two setups that this repository introduced and had
never controlled for:

| | how the risk score is built |
|---|---|
| here | two components, each isotonic-calibrated, then a fusion on top |
| external | one model, one isotonic calibration |

Two step functions stacked under a third fitted layer is not the same object
as one step function. It could compound the ties in a way a single calibrated
model does not — which would make the failure a confound of our own making
rather than a fact about the data.

That is not fishing for a hypothesis that fits. It is the one structural
difference between the measurement that failed and the measurements that held.

## The test

Each external corpus run three ways at 2%, 5% and 10% requests, same folds,
everything else fixed:

* **single** — one model, isotonic; what `external_calibration.py` did
* **stacked** — two components, each isotonic, fused; what we do
* **stacked-P** — the same two components under Platt, fused

| architecture | contracts held | mean share of negatives on the cut |
|---|---|---|
| single | **9 of 9** | 9.2% |
| stacked | 8 of 9 | **1.1%** |
| stacked-P | 8 of 9 | **0.0%** |

## Refuted, and in the most direct way

**Stacking reduces ties.** 9.2% of negatives sit on the threshold's exact
value with a single calibrated model, 1.1% when two are fused. Fusing with a
logistic layer on top of two calibrated columns produces a nearly continuous
score. The hypothesis predicted the opposite.

**And the one break is not about ties at all.** Both breaks are the same
thing:

| corpus | architecture | request | delivered | margin |
|---|---|---|---|---|
| SMS Spam | stacked | 0.02 | 0.021 | **+0.001** |
| Enron-Spam | stacked-P | 0.02 | 0.021 | **+0.001** |

One of them is in the **Platt** arm, which has no ties at all — 0.0% on the
cut. A break there cannot be caused by ties. Both are at the tightest request
and both miss by a thousandth. This is noise at 0.02, the same marginal
boundary case that broke in the model-strength test.

## A verdict that was too weak, again

The script's first verdict read the count and said *"SUPPORTED: stacking breaks
1 contract that a single calibrated model keeps."* That is the same mistake
`ties_and_model_strength.py` made before it was tightened: a difference of one
is not a mechanism, and here the tie-free arm breaks just as often.

The logic now requires the tie mechanism to explain the break — if the arm
with zero ties breaks too, the hypothesis is refuted rather than supported —
and calls a gap of one or two inconclusive rather than supporting.

## Where this leaves it

**Three hypotheses, three refutations:**

| hypothesis | verdict |
|---|---|
| the failing threshold is less stable | refuted — it is four times *more* stable |
| the model is weaker | refuted — a corpus at worse Brier kept its contract |
| the architecture stacks calibrators | refuted — stacking *reduces* ties |

What stands is the mechanism without its direction:

> Isotonic calibration puts a large share of items on the threshold's exact
> value — up to 59.5% of negatives on one corpus. Whether that lands on the
> permissive or the conservative side of a rate promise is unexplained.
> Neither stability, model strength nor architecture distinguishes the corpora
> where it falls one way from those where it falls the other.

Three refutations is a result. It is a narrower one than a mechanism with a
direction would be, and it is what was measured.

## Reproduce

```
python scripts/stacking_ties.py
```
