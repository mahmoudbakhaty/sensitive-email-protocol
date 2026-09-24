# The block promise is not hard to estimate. It has no solution.

`filter_system.py` keeps one half of its contract. *"No more than X% of
sensitive messages pass automatically"* holds at every request measured; *"no
more than Y% of harmless messages are blocked automatically"* overshoots by
1.2x to 3.1x. Both thresholds are set the same way, on the same validation
threads, with the same bound. Only one holds.

An observation is not a result until the asymmetry has a cause.

## The obvious explanation is wrong

The threshold is a quantile of a validation distribution applied to a test
distribution, so the natural guess is that the harmless upper tail moves
between folds more than the sensitive lower tail does. Measured across five
folds, in units of the score's own standard deviation:

| side | mean shift, validation to test | max |
|---|---|---|
| leak | 0.252 sd | 0.432 |
| block | **0.063 sd** | 0.257 |

The block threshold is **four times more stable**, not less. Distribution
shift is not the cause. It was the hypothesis, and it is refuted.

## The cause is ties

What does separate the two sides is how crowded each threshold is: 6.1% of
harmless test messages sit within a tenth of a standard deviation of the block
threshold, against 2.1% for the leak threshold. Counting exact ties says why.

On one test fold the risk score takes **62 distinct values over 277 messages**.

| | messages exactly on the cut |
|---|---|
| leak threshold | 0 of 52 sensitive |
| block threshold | **28 of 225 harmless — 12.4%** |

So *"at most 5% of harmless messages above this cut"* has no solution. Move the
cut a hair down and 12% go over; a hair up and 0% do. Nothing in between
exists. The binomial bound was not failing to estimate a quantile. There was
no quantile there to estimate.

## The calibrator did it

Isotonic regression is a step function. It maps whole intervals of input to a
single output, and with a few hundred validation points those steps are wide.
It is chosen for calibration quality, it is good at that, and it destroys
exactly the resolution a rate guarantee needs.

Platt scaling - a fitted sigmoid - is strictly monotone and preserves every
distinction the underlying model made. `scripts/calibrator_ties.py` measures
the trade on the same folds:

| calibrator | distinct values | messages on the cut | Brier |
|---|---|---|---|
| isotonic | 48.6 | 10.6 (4.8% of harmless) | **0.2268** |
| Platt | **275.6** | **0.0** | 0.2449 |

And whether the contract then holds:

| calibrator | requested | delivered leak | delivered false block | automated |
|---|---|---|---|---|
| isotonic | 0.02 | 0.000 | **0.063** ✗ | 8.8% |
| isotonic | 0.05 | 0.040 | **0.080** ✗ | 26.0% |
| isotonic | 0.10 | 0.052 | **0.116** ✗ | 34.4% |
| Platt | 0.02 | 0.000 | 0.014 ✓ | 1.5% |
| Platt | 0.05 | 0.012 | 0.035 ✓ | 10.6% |
| Platt | 0.10 | 0.024 | 0.086 ✓ | 22.4% |

**Every isotonic promise on the block side breaks. Every Platt promise holds.**

The price is visible and small: calibration about 8% worse by Brier score, and
roughly half the automation at the same request. A filter that honours its
contract while automating less is worth more than one that automates more and
breaks it.

## What generalises

This is not a fact about this corpus. Anyone building a filter with a rate
guarantee on an isotonically calibrated score meets the same wall, and it
does not announce itself: the score looks like a probability, the bound looks
sound, and the contract quietly fails in one direction only. The diagnosis is
one line - count how many messages share the threshold's exact value.

Isotonic calibration and a rate contract are in tension. Pick the calibrator
for the guarantee you need, not for the Brier score alone.

## Reproduce

```
python scripts/why_block_fails.py     # shift, density, and the ties
python scripts/calibrator_ties.py     # isotonic against Platt, both contracts
```
