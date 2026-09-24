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

## A note on what has changed since

This file measured a block side that bounded a **rate** - at most Y% of
harmless messages blocked. That contract was later found to be the wrong shape
(`RESULTS_BLOCK_IS_NOT_VIABLE.md`): it says nothing about how many blocks are
right, and the system was blocking 42 messages at 40% precision while
promising 90%.

The block side bounds **precision** now, and on this benchmark no cut
qualifies, so the system blocks nothing and both calibrators keep the block
promise trivially. Everything below is a true record of the rate-based
contract and of why it could not be honoured; it is not a description of the
system's current behaviour.

What survives into the current system is the resolution finding: isotonic
leaves about fifty distinct values against Platt's two hundred and seventy-six,
and that is why the default calibrator changed.

## What generalises, and what does not

An earlier version of this file claimed that anyone putting a rate guarantee
on an isotonically calibrated score meets the same wall. **That claim was
tested on three public corpora and it is wrong.** The correction is kept here
rather than quietly edited out.

`scripts/external_calibration.py` repeats the measurement on SMS Spam (5,574
messages, 13.4% positive), tweet_eval/hate (9,000 tweets, 42%) and Enron-Spam
(33,716 messages, 50.9%), at 2%, 5% and 10% requests.

**The mechanism replicates everywhere, and it is large.**

| | distinct values | negatives on the cut |
|---|---|---|
| isotonic | 19 to 59 | 0.5% to **59.5%** |
| Platt | 1,059 to 4,556 | 0.0% |

On Enron-Spam at a 10% request, **59.5% of the negatives sit on one value**.
The resolution loss is not a property of this benchmark.

**The contract failure does not replicate.** Isotonic held all nine external
contracts; Platt held eight of nine. Where our benchmark's ties landed on the
permissive side and broke the promise, the external ties landed on the
conservative side and kept it.

So the honest statement is narrower than the one it replaces:

> Isotonic calibration makes the delivered rate **unpredictable**, not
> systematically too high. A large share of items can sit on the threshold's
> exact value, and which side of the promise they fall on is a property of the
> data, not something the bound controls. On this benchmark it fell the wrong
> way.

The practical advice survives and the reasoning behind it changes. The
diagnosis is still one line - count how many items share the threshold's exact
value - but what it tells you is that the contract is at the mercy of a tie,
not that it will fail.

## Reproduce

```
python scripts/why_block_fails.py     # shift, density, and the ties
python scripts/calibrator_ties.py     # isotonic against Platt, both contracts
```
