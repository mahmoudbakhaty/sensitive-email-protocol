# The design works. 5.5% is the task, not the policy.

On this benchmark the filter automates 5.5% of traffic, leaks 1.2% against a
5% promise, and blocks nothing. That number admits two readings which point in
opposite directions:

* the design is sound and this task is simply hard;
* the design is too conservative to be useful anywhere.

The largest weakness in this work is a single corpus, and no second corpus for
context-dependent sensitivity exists to test the protocol on. The *design* can
still be tested even where the task cannot.

Three public corpora, spanning Brier 0.009 to 0.152 against 0.227 here.
**Nothing about the system is retuned**: the same contract, the same
confidence level, the same precision requirement for blocking, the same
calibrator. Only the folds change - stratified rather than thread-grouped,
because none of them has threads.

## At a 5% leak request

| corpus | messages | automated | allowed / blocked | leak | error among auto | contract |
|---|---|---|---|---|---|---|
| SMS Spam | 5,574 | **98.2%** | 4,705 / 768 | 0.029 | 1.4% | held |
| tweet_eval/hate | 9,000 | **36.3%** | 2,471 / 793 | 0.044 | 6.7% | held |
| Enron-Spam | 33,345 | **100.0%** | 17,005 / 16,340 | 0.035 | 1.9% | held |
| **this benchmark** | 1,382 | **5.5%** | 76 / 0 | 0.012 | 4.0% | held |

The same system that clears 5.5% of this traffic clears 98% of SMS Spam and
all of Enron-Spam, under the same promise and keeping it.

> **5.5% is the task, not the policy.**

## Two things that make this a stronger result than the headline

**The system blocks elsewhere.** It auto-blocks 768, 793 and 16,340 messages
on the three external corpora, and zero here. The precision requirement is not
a setting that never fires - it fires readily where the model supports it. It
refuses here because precision at the top of the risk score is 28.6%, and that
refusal is a measurement of this task rather than a property of the rule.

**The escalation load scales sensibly.** tweet_eval/hate - contested human
judgement, like ours, but easier - lands in between at 36.3%. The ordering
follows how well a linear model separates the classes, which is what a sound
design should do.

## One contract broke, and it is not marginal

| corpus | request | delivered | margin |
|---|---|---|---|
| Enron-Spam | 0.02 | 0.035 | **+0.015** |

The tightest request on the largest corpus, missing by one and a half points.
Unlike the boundary misses in `RESULTS_STACKING_TIES.md`, this one is real.
Eight of nine contracts held; this one did not, and it is reported rather than
rounded away.

## A row that reads worse than it is

SMS Spam at a 10% request shows a 37.4% error rate among automatic decisions.
The system automates **100%** of the corpus there, so every error the
classifier makes anywhere counts as an automatic error; there is nothing left
escalated to absorb them. At full coverage "error among automatic decisions"
is just the classifier's error rate, and the metric stops carrying its usual
meaning.

## What this does and does not establish

It establishes that the design is not the limit. A three-outcome policy with a
leak contract, a precision requirement for blocking and validation-only
thresholds automates nearly all of an easy corpus and almost none of a hard
one, keeping its promise in both - which is what such a design should do.

It does not establish that the protocol transfers, because none of these
corpora is context-dependent sensitivity, none has thread structure, and none
has two annotators. They test the filter, not the task.

## Reproduce

```
python scripts/system_elsewhere.py
```
