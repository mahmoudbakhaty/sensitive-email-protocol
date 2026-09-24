# What more labels buy, and which labels to buy

Two measurements said better components would not raise the system's
automation: the encoder's ceiling is −0.1 points even with the measurement
rigged in its favour, and the tightest leak rate the benchmark can promise at
all is about 5%, because promising 1% needs 230 threads carrying a sensitive
message and validation has 69. Both point at labelled data rather than
modelling.

Label efficiency is normally measured against accuracy. This measures it
against the operating contract, which is what a deployment actually signs up
to - and which only a system with a contract can ask about.

The benchmark is subsampled by **whole threads**, never by message, and the
whole system refitted and re-measured out of fold at each budget. A 5% leak
contract, Platt calibration, the same folds throughout.

## The curve is a cliff, not a slope

| messages | threads | threads carrying a sensitive message | automated | leak | contract |
|---|---|---|---|---|---|
| 201 | 167 | 33 | 0.0% | 0.000 | held 3/3 |
| 400 | 323 | 63 | 0.0% | 0.000 | held 3/3 |
| 600 | 481 | 98 | 0.0% | 0.000 | held 3/3 |
| 800 | 639 | 134 | 0.0% | 0.000 | held 3/3 |
| 1002 | 807 | 169 | 0.7% | 0.002 | held 3/3 |
| 1203 | 966 | 204 | **5.1%** | 0.011 | held 3/3 |
| 1382 | 1103 | 230 | 5.5% | 0.012 | held 3/3 |

Automation is **zero** below about a thousand messages, then turns on. The
switch is not the message count: it is the third column. The bound needs
roughly fifty threads carrying a sensitive message in the validation split,
and 169 total threads is where that lands.

The contract holds at every budget, including the ones where nothing is
automated - refusing to automate is how it holds.

**An average slope is the wrong statistic for this shape**, and the first
version of the script reported one: "+0.47 points per 100 labels, so ten more
points needs 2,147 labels". That is an artefact of averaging across a cliff.
The honest numbers are where it turns on, and what the last stretch bought:
from 1,203 to 1,382 messages - 179 more - automation rose 0.4 points.

**We are past the cliff, on the flat.** More labels of the same kind buy
little.

## Targeting the decision boundary makes it worse

An active-learning loop would label the cases the model is least sure about.
Simulated here with labels already in hand, which is an upper bound on what
such a loop could achieve:

| labels | random | targeted |
|---|---|---|
| 200 | 0.0% (33 sensitive threads) | 0.0% (14) |
| 800 | 0.0% (134) | 0.0% (84) |
| 1200 | **5.1%** (204) | **0.2%** (168) |

Targeting is **worse at every budget**, and the reason is in the bracketed
column. Threads near the decision boundary carry fewer sensitive messages, so
boundary-targeted labelling starves the one quantity the bound counts.

> When the bottleneck is a tail quantile rather than a decision surface, label
> **positives**, not boundary cases. Uncertainty sampling optimises the wrong
> thing.

That is a property of the guarantee, not of this corpus: a bound on "what
share of sensitive messages pass automatically" is estimated from sensitive
messages, and nothing else in the sample improves it.

## What this changes

It changes the recommendation that produced it. Labelling was proposed as the
lever because the bound is sample-limited, and it is - but not at this
operating point. The cliff has been crossed and the marginal return is small.
The useful form of the advice is narrower:

* below about 170 sensitive threads, a rate contract cannot be honoured at all
  except by automating nothing;
* crossing that point is worth a great deal and is a one-time gain;
* past it, more labels buy tenths of a point unless they are chosen to be
  positives.

## Reproduce

```
python scripts/label_budget.py
```

Twenty-five refits of the whole system; it takes a while.
