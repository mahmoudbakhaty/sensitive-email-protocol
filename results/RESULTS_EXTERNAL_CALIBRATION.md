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

## The contract failure does not replicate

| calibrator | contracts held, of 9 |
|---|---|
| isotonic | **9** |
| Platt | 8 |

Every external isotonic contract held. The one Platt miss is SMS Spam at a 2%
request, delivered 0.020 against a requested 0.02 - at the boundary.

Where our benchmark's ties landed on the permissive side and broke the
promise, the external ties landed on the conservative side and kept it. The
ties are real everywhere; which way they fall is not.

## The corrected statement

> Isotonic calibration makes the delivered rate **unpredictable**, not
> systematically too high. A large share of items can sit on the threshold's
> exact value, and which side of the promise they fall on is a property of the
> data rather than something the bound controls. On this benchmark it fell the
> wrong way.

The earlier claim - that any rate guarantee on an isotonically calibrated
score meets the same wall - is withdrawn. It was published in this repository
for about an hour before this measurement contradicted it, and the correction
is recorded here rather than edited away.

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

The three corpora download from UCI, HuggingFace and a GitHub mirror; the
script reads them from `$EXTERNAL_DATA`.
