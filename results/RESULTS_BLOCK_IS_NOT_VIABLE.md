# Automatic blocking is not viable on this benchmark, and the contract was the wrong shape

The filter was designed with three outcomes: allow, escalate, block. Two
measurements say the third does not belong there, and a third says the
contract that governed it was asking the wrong question.

## The signal that started it

At a 2% request the system automated 1.5% of traffic and **80% of those
automatic decisions were wrong** - 16 wrong blocks against 5 right ones. A
system automating almost nothing and getting most of it wrong is not a tuning
problem.

## Precision at the top of the risk score

Auto-blocking is only safe where the model is nearly always right. Measured
out of fold, pooled across the five thread-disjoint folds:

| top share by risk | messages | sensitive | precision |
|---|---|---|---|
| 1% | 14 | 4 | **28.6%** |
| 2% | 28 | 13 | 46.4% |
| 5% | 69 | 26 | 37.7% |
| 10% | 138 | 50 | 36.2% |
| 20% | 276 | 92 | 33.3% |

Base rate 18.1%. The model lifts precision to roughly twice base and never
approaches what blocking something automatically requires. **There is no
operating point at which auto-blocking is safe on this benchmark.** Block the
top 2% and more than half of what you block is harmless.

## The contract was the wrong shape

The original block contract bounded the *share of harmless messages blocked*:
at most 2% of them. That is 22 messages out of 1,132, and it says nothing
about how many of the blocks are right - 22 wrong against perhaps 13 right
satisfies it. A rate bound on one class cannot express "what I block should
usually be sensitive".

The block side now bounds **precision** instead: at least 90% of what is
blocked must be genuinely sensitive, as a one-sided Clopper-Pearson lower
limit, with the effective sample size taken as threads rather than messages.
When no cut qualifies it returns infinity and the system blocks nothing.

On this benchmark no cut qualifies. The system blocks nothing, and that is the
correct answer rather than a failure to find one.

## A bug in the bound, found by testing it

The first version returned certainty when every sampled item above the cut was
correct. "One correct out of one" is not certainty: the Clopper-Pearson lower
limit for h = n is `alpha**(1/n)`, which is 0.10 at n = 1, not 1.0. That let a
single lucky message open the gate, and the system blocked 42 messages at 40%
precision while promising 90%.

Corrected and checked on synthetic data with a known answer, 400 draws per
setting:

| true precision at the top | blocked in | promise broken |
|---|---|---|
| 0.95 | 132 of 400 | **0** |
| 0.80 | 2 of 400 | **0** |
| 0.50 | **0 of 400** | 0 |

It blocks when the population supports the promise, refuses when it does not,
and never breaks it.

## The system as it stands

| | |
|---|---|
| handled automatically | 76 of 1382 (5.5%) - all ALLOW |
| auto-blocked | **0** |
| escalated | 1306 (94.5%) |
| sensitive auto-allowed | 3 = 1.2%, contract allows 5% |
| error rate among automatic decisions | 4.0% |

Read from `RESULTS_FILTER_SYSTEM.json`. This was the third copy of an
isotonic-era table (209 / 15.1% / 10 / 4.8%) still presented as current after
the default calibrator changed to Platt.

**A two-decision filter, not a three-decision one**, and the reason is
measured rather than assumed.

## What this says about the framework

The registered title promises a hybrid framework. Part of what this
measurement establishes is what such a framework can honestly offer on a task
this contested: it can clear a sixth of the traffic with a bounded leak rate,
and it cannot block anything automatically. A design that shipped all three
actions would have been shipping one that does not work.

## Reproduce

```
python scripts/filter_system.py       # the system; blocks nothing, and says why
python scripts/encoder_ceiling.py     # what the encoder could add, at best
```
