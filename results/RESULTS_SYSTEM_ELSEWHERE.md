# The design works. 5.5% is the task, not the policy.

On this benchmark the filter automates 5.5% of traffic, leaks 1.2% against a 5%
promise, and blocks nothing. That number admits two readings which point in
opposite directions:

* the design is sound and this task is simply hard;
* the design is too conservative to be useful anywhere.

The largest weakness in this work is a single corpus, and no second corpus for
context-dependent sensitivity exists to test the protocol on. The *design* can
still be tested even where the task cannot.

Three public corpora, spanning Brier 0.009 to 0.152 against 0.227 here.
**Nothing about the system is retuned**: the same contract, the same
confidence level, the same precision requirement for blocking, the same
calibrator. Only the folds change — stratified rather than thread-grouped,
because none of them has threads.

```
python scripts/external_calibration.py   # fetches the three corpora
python scripts/system_elsewhere.py
```

The corpora download from UCI, HuggingFace and a GitHub mirror on first use and
are fingerprinted. **They did not until 25 September** — the loaders read local
files that a clean clone never had, so this Reproduce block died on
`FileNotFoundError` before printing anything.

---

## At a 5% leak request

| corpus | messages | automated | allowed / blocked | leak | block precision | contract |
|---|---|---|---|---|---|---|
| SMS Spam | 5,574 | **98.2%** | 4,705 / 768 | 0.029 | 0.928 | held |
| tweet_eval/hate | 9,000 | **36.3%** | 2,471 / 793 | 0.044 | 0.938 | held |
| Enron-Spam | 33,345 | **51.5%** | 17,180 / 0 | 0.044 | — | held |
| **this benchmark** | 1,382 | **5.5%** | 76 / 0 | 0.012 | — | held |

The same system that clears 5.5% of this traffic clears 98% of SMS Spam and
half of Enron-Spam, under the same promise and keeping it.

> **5.5% is the task, not the policy.**

## Every contract holds — and this is the first run where that was tested

| request | SMS Spam | tweet_eval/hate | Enron-Spam |
|---|---|---|---|
| 0.02 | 52.6%, leak 0.005 | 26.4%, leak 0.017 | 50.0%, leak 0.018 |
| 0.05 | 98.2%, leak 0.029 | 36.3%, leak 0.044 | 51.5%, leak 0.044 |
| 0.10 | 87.1%, leak 0.066 | 47.3%, leak 0.095 | 54.0%, leak 0.093 |

**Nine of nine, on both halves of the promise.** That sentence replaces "eight
of nine contracts held", and the change is not a re-run producing nicer
numbers — two defects were found and fixed, and both of them mattered.

### The check tested half the promise

`system_elsewhere.py` computed `held = leak <= q` and never looked at the block
side, although the filter also promises block precision of at least 0.90. A
one-sided check on a two-sided promise is not a check: a policy that blocks the
whole corpus keeps any leak contract trivially. The same one-sided copy existed
in two other scripts. There is now one definition, in `filter_system.py`, and
all three call it.

### And the policy itself was broken where the thresholds crossed

`_set_policy` ended with:

```python
if self.t_block <= self.t_allow:      # degenerate: escalate nothing
    self.t_block = self.t_allow = float(np.median(risk))
```

When the allow threshold rises above the certifiable block cut, that discards
**both** bounds and replaces them with the median risk score — a number that is
neither a leak bound nor a precision bound. Everything above the middle of the
score distribution was then blocked with nothing behind the decision.

It fired on two of these three corpora, and the published table was its output:

| | as published | with the policy fixed |
|---|---|---|
| SMS Spam @ 0.10 | 100% automated, 2,823 blocked at **0.264** precision, 37.4% error among automatic decisions | 87.1%, **nothing blocked**, 1.0% error |
| Enron-Spam @ 0.02 | leak **0.035** against 0.02 — reported as the one real contract failure | leak **0.018** — holds |
| Enron-Spam, all three requests | byte-identical rows, because the outcome could not respond to the contract | 50.0% / 51.5% / 54.0% |

The allow threshold is the primary contract and is kept; no cut above it can be
certified for precision, which is what crossing means, so the system blocks
nothing. `test_bounds.py` now asserts that directly, and the check was confirmed
to fail against the old code before being trusted.

**The one failure this document used to report is gone**, and not because it was
rounded away: Enron-Spam at 0.02 delivered 0.035 *because the contract had been
discarded*, and delivers 0.018 once it is not.

## Two things that make this a stronger result than the headline

**The system blocks elsewhere.** It auto-blocks 768 and 793 messages on two of
the three external corpora, and zero here. The precision requirement is not a
setting that never fires — it fires readily where the model supports it. It
refuses here because precision at the top of the risk score is 28.6%, and that
refusal is a measurement of this task rather than a property of the rule.

**The escalation load scales sensibly.** tweet_eval/hate — contested human
judgement, like ours, but easier — lands lowest of the three at 36.3%. The
ordering follows how well a linear model separates the classes, which is what a
sound design should do.

## A row that no longer reads worse than it is

This document used to carry a section explaining away SMS Spam at a 10% request:
100% automation and a 37.4% error rate among automatic decisions, argued to be a
metric artefact because "at full coverage there is nothing left escalated to
absorb the errors".

That explanation was wrong. The 37.4% was 2,079 harmless messages auto-blocked
at 26.4% precision against a 90% promise — a genuine contract violation, not an
artefact of the denominator. The section defended the very row that should have
raised the alarm. With the policy fixed the row automates 87.1%, blocks nothing,
and errs on 1.0%.

## What this does and does not establish

It establishes that the design is not the limit. A three-outcome policy with a
leak contract, a precision requirement for blocking and validation-only
thresholds automates nearly all of an easy corpus and almost none of a hard one,
keeping both halves of its promise in every case — which is what such a design
should do.

It does not establish that the protocol transfers, because none of these corpora
is context-dependent sensitivity, none has thread structure, and none has more
than one annotator. They test the filter, not the task.
`RESULTS_SECOND_CORPUS.md` takes two of those three gaps further.
