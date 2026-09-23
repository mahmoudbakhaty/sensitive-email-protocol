# What a model is worth when it is allowed to decline

Every method in this paper lands between F1 0.368 and 0.397 on the strict
labels at full coverage, with overlapping intervals, which is why the paper
ranks none of them. That comparison asks one question: how often is the model
right when it is forced to answer about every message. A deployed filter is
never in that position. It decides automatically where it is confident and
routes the rest to a person.

Measured under the paper's own protocol, the two questions give very different
answers.

| model | full coverage | ~30% coverage | gain | z vs random | AURC |
|---|---|---|---|---|---|
| logistic regression | 0.3683 | 0.4772 | +0.109 | +3.4 | 0.586 |
| character n-grams | 0.3675 | 0.4571 | +0.090 | +2.8 | 0.584 |
| linear SVM | 0.3814 | 0.4880 | +0.107 | +3.3 | 0.561 |
| **fine-tuned encoder** | 0.3971 | **0.5592** | **+0.162** | **+5.1** | **0.525** |

AURC is the area under the risk-coverage curve with risk = 1 − F1; lower is
better.

## What makes this a measurement rather than an artefact

**The abstention rule never sees test data.** Declining messages changes the
set the score is computed on, so a curve drawn by choosing the coverage that
looks best on test would mean nothing. Each fold carves 30% of its training
threads off, fits there, and takes both the decision threshold and the
confidence cut-off from that split. The cut-off is an absolute confidence
value, so the coverage actually reached on test is whatever it is, and the
table reports it beside the target — the 0.30 target lands at 0.266 to 0.355
depending on the model.

**Random abstention is the control.** Dropping 70% of the messages changes F1
by arithmetic alone. Every coverage is therefore run again with the same
number declined at random, 200 times. Random abstention leaves F1 flat at
0.368 for logistic regression against 0.477 for confidence-ordered abstention:
the gain is not the shrinking denominator.

**Intervals are thread-level bootstraps**, 2000 resamples, on the selected
subset.

## What this does and does not establish

It establishes that abstention buys a large improvement for every model here,
and that the improvement clears a random-abstention control at z = 2.8 to 5.4.
The encoder gains most - 0.162 F1, larger than any difference the paper
discusses elsewhere - and has the best AURC.

It does not establish a ranking. At 30% coverage the encoder's interval is
[0.464, 0.636] and the linear SVM's is [0.400, 0.569]; they overlap, as every
comparison in this paper does. The honest statement is that abstention works
and the encoder benefits most, not that the encoder is better.

## A correction worth recording

An earlier, quicker version of this analysis used a single global threshold
recovered from the released predictions instead of per-fold validation-selected
thresholds. Under it, logistic regression and the linear SVM appeared to fall
*below* the random band - their confidence looked anti-informative. That was an
artefact of ranking by distance from a threshold that was not the model's own:
logistic regression's recovered global threshold was 0.4999 while its per-fold
validation thresholds are not 0.5. Corrected here. The conclusion it suggested
- that only the encoder has usable confidence - is not supported.

## The weaker arm

The three classical models are retrained by this script, so nothing rests on a
released file. The encoder cannot be retrained without a GPU, so its curve is
cross-fitted from the released out-of-fold scores: for each test fold the
cut-off comes from the other folds, whose scores were produced by models that
never saw it. That is weaker than the retrained arms and is labelled as such in
the JSON. Re-running it properly needs a GPU, and so does adding the
instruction-tuned model, which has no released per-message scores at all.

## Reproduce

```
python scripts/selective.py        # ~4 minutes on CPU
```
