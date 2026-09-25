# Testing the calibration finding outside this corpus

`RESULTS_WHY_BLOCK_FAILS.md` reported that isotonic calibration makes a rate
guarantee impossible to honour, and claimed that this would hold anywhere. The
claim was made on one corpus of 1,382 messages. It is a claim about a
calibrator and an order statistic, so it should hold on other data - and if it
does not, the finding is about Enron and has to be stated that way.

Three public corpora, chosen to vary on what could explain it away:

| corpus | messages | positive rate | why |
|---|---|---|---|
| SMS Spam (UCI) | 5,574 | 13.4% | closest in shape, four times the size |
| tweet_eval/hate (CardiffNLP) | 9,000 | 42.0% | contested human judgement, but balanced |
| Enron-Spam | 33,716 | 50.9% | balanced, twenty-four times the size |

None has thread structure, so stratified folds replace the grouped ones.
Everything else is the protocol: fit on the training fold minus a 30%
validation split, calibrate there, set the threshold there, measure out of
fold, at 2%, 5% and 10% requests.

## The mechanism replicates, and it is large

| corpus | calibrator | distinct values | negatives on the cut |
|---|---|---|---|
| SMS Spam | isotonic | 19.2 | 1.7% – 9.3% |
| | Platt | 1,058.8 | 0.0% |
| tweet_eval/hate | isotonic | 58.8 | 1.2% – 2.8% |
| | Platt | 1,794.0 | 0.0% |
| Enron-Spam | isotonic | 34.4 | 0.5% – **59.5%** |
| | Platt | 4,556.4 | 0.0% |

Isotonic collapses a continuous score into a few dozen values on every corpus,
and at a 10% request on Enron-Spam it puts **59.5% of the negatives on a single
value**. Platt leaves thousands of distinct values and never ties anything to
the threshold. The resolution loss is a property of the calibrator, not of this
benchmark.

## The contract failure replicates everywhere

| calibrator | contracts held, of 9 |
|---|---|
| isotonic | **0** |
| Platt | **8** |

Isotonic breaks the promise on every corpus at every request. Platt keeps it
except at one boundary — SMS Spam at a 2% request, delivering 0.0201 against
0.0200, a miss of one ten-thousandth, about half a message in 4,824.

## This section said the opposite until 25 September

It read "the contract failure does not replicate — isotonic 9 of 9, every
external isotonic contract held", and on that basis it **withdrew** the claim
that a rate guarantee on an isotonically calibrated score meets the same wall
everywhere. The withdrawal was wrong. The claim was right.

The cause was one character. This script counted a test negative as compliant
unless it sat *strictly above* the cut:

```python
n_over += int((neg_te > t).sum())        # the system blocks at r >= t_block
```

The system acts at `>=`, so an item sitting exactly **on** the cut is acted on.
Excluding those items is not a rounding choice here — it removes precisely the
population isotonic creates. Isotonic collapses the score into a few dozen
values, so the cut *is* one of those values and a large share of the data sits
exactly on it.

Enron-Spam at a 10% request, the two counts side by side over the same folds:

| | delivered | verdict |
|---|---|---|
| counting `neg > t` (what this script did) | 0.0748 | held |
| counting `neg >= t` (what the system does) | **0.8367** | broke, by 8.4× |
| items sitting exactly on the cut | **12,567 of 16,493 negatives (76.2%)** | |

The measurement was biased in exactly the direction that reversed the
conclusion, on exactly the calibrator whose defining property is ties on the
threshold.

## The statement that stands

> **Isotonic calibration cannot keep a rate guarantee.** It collapses a
> continuous score into a few dozen values; the threshold lands on one of them;
> a large share of the data sits on it; and every one of those items is acted
> on. Nine of nine external contracts broken, on three corpora spanning Brier
> 0.009 to 0.152. Platt leaves thousands of distinct values, ties nothing to
> the threshold, and holds eight of nine with the ninth missing by 0.0001.

This is the original claim, withdrawn on a faulty measurement and reinstated on
a corrected one. Both the withdrawal and the reinstatement are recorded here
rather than edited away, because the sequence is the point: a one-character
comparison decided which of two opposite conclusions this release published.

## What it does not settle

Three corpora is not a survey, and all three are near-balanced or spam-like
tasks where a linear model separates the classes far better than it does on
contested sensitivity: Brier scores of 0.009 to 0.152 against 0.227 here. It
may be that ties only fall the permissive way when the model is weak, which
would make the direction predictable after all. That is the next measurement,
not a conclusion.

## Reproduce

```
python scripts/external_calibration.py
```

The three corpora download from UCI, HuggingFace and a GitHub mirror on first
use, into `$EXTERNAL_DATA`, and each is fingerprinted. **They did not until
25 September** - the loaders read local files a clean clone never had, so this
block died on `FileNotFoundError` before printing anything.
