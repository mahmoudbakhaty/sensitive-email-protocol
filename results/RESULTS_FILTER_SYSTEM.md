# The filter as a system, and what it can promise

Everything else here measures classifiers. A deployed filter is not a
classifier: it is a policy with three outcomes - let it through, hold it for a
person, block it - and the questions that decide whether it is useful are how
much traffic it clears without a human and what the traffic it does clear
costs.

`scripts/filter_system.py` is that, as something you can call:

```python
f = SensitivityFilter().fit(texts, labels, threads)
f.decide(message)   # {"action": "ESCALATE", "risk": 0.42, ...}
```

The thresholds are not chosen to maximise accuracy. In data-loss prevention a
sensitive message auto-allowed is a leak and a harmless message escalated is a
minute of someone's time, so the policy is a contract - *at most this share of
sensitive messages may be auto-allowed* - and whatever coverage that leaves is
the answer rather than the target.

## At the default contract

5% of sensitive messages may be auto-allowed, 1% of harmless ones auto-blocked.
Out of fold, thread-disjoint, thresholds set on validation threads inside each
training fold:

| | |
|---|---|
| handled automatically | 209 of 1382 (15.1%) |
| escalated to a person | 1173 (84.9%) |
| sensitive auto-allowed | 10 = **4.0%**, contract allows 5% |
| harmless auto-blocked | 0 = **0.0%**, contract allows 1% |
| error rate among automatic decisions | 4.8% |

Both sides hold. The block side holds by blocking nothing at all, which is the
honest answer: the sample cannot support a 1% promise, so the system declines
to make automatic blocks rather than make them and break the contract.

## The contract did not hold at first, and why

The first version set each threshold at the matching quantile of the
validation scores. The contract broke out of fold in both directions:

| requested | delivered leak | delivered false block |
|---|---|---|
| 1% | **4.0%** | **6.3%** |
| 5% | 7.2% | 9.9% |

Three separate causes, each found by measuring rather than reasoning.

**A point estimate of a tail quantile is not a bound.** The 5th percentile of
about seventy validation positives is an estimate with a wide interval, and
roughly half the time the true quantile sits below it - every one of those
times the contract breaks. Replaced with a one-sided binomial confidence
bound.

**The fallback was not a bound either.** When the sample could not support the
promise, the first bound returned the sample's most extreme value. Tested
directly: with fifty points from a uniform, a 1% promise was broken 60% of the
time, because the share beyond the extreme point of a sample of fifty is about
1/51. It now returns infinity, which automates nothing on that side.

**The binomial assumes independent draws, and these are not.** Messages in a
thread share a topic, an author and usually a label - the premise of the whole
protocol. Counting them as independent overstates the sample. Giving the
binomial the number of *threads* instead moved the block-side overshoot from
6.3x the promise to 3.1x.

## What holds and what does not

| | |
|---|---|
| *"no more than X% of sensitive messages pass automatically"* | **holds** at every request measured |
| *"no more than Y% of harmless messages are blocked automatically"* | **breaks**, by 1.2x to 3.1x |

The guarantee this system can offer is one-sided. For data-loss prevention
that is the useful side, but it should be stated rather than implied.

## The promise this benchmark cannot support at all

At a 1% request the system automates nothing, which is not a failure but the
correct refusal. The arithmetic is not about the model:

| promise | threads carrying a sensitive message needed | available in validation |
|---|---|---|
| 1% | 230 | 69 |
| 2% | 114 | 69 |
| 5% | 45 | 69 |
| 10% | 22 | 69 |

With 1,382 messages over 1,103 threads, the tightest leak rate that can be
promised at 90% confidence is about 5%. A better model does not change this.
Promising 1% needs roughly three times this benchmark's sensitive threads.

## Reproduce

```
python scripts/filter_system.py      # the system at the default contract
python scripts/policy_transfer.py    # requested against delivered, 8 policies
```

The second rebuilds the system forty times and takes a while.
