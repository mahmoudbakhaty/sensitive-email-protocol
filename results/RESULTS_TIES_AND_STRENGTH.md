# Model weakness does not explain which way the ties fall either

`RESULTS_EXTERNAL_CALIBRATION.md` withdrew a claim: isotonic calibration
collapses a score into a few dozen values on every corpus tested, but the rate
contract held on all three external ones and broke only here. Which side of
the promise the ties fall on looked like a property of the data.

One thing separated the four, and it was not size or class balance:

| corpus | Brier | contract |
|---|---|---|
| Enron-Spam | 0.0088 | held |
| SMS Spam | 0.0133 | held |
| tweet_eval/hate | 0.1524 | held |
| **this benchmark** | **0.2268** | **broke** |

A natural hypothesis, and a sharper warning if true: ties fall the permissive
way only when the model is weak - which would mean a rate guarantee fails
exactly when it is most needed.

## The test, and the prediction made before running it

Each external corpus re-run with its training fold cut to 30%, 10%, 5%, 2% and
1% of itself. The validation split is **not** shrunk, so the threshold is
estimated from the same amount of data throughout and sample size cannot be
mistaken for model strength. Everything else is fixed: corpus, calibrator,
protocol, a 5% request.

The prediction, written into the script before the run: breaks should appear
above a Brier of roughly 0.15, and corpora should cross over at similar Brier
values rather than at similar training sizes.

## The result

| corpus | worst Brier reached | contracts broken |
|---|---|---|
| Enron-Spam | 0.0274 | none |
| SMS Spam | 0.0656 | one, at Brier **0.0244** |
| tweet_eval/hate | **0.2332** | none |

**Refuted.** tweet_eval/hate was weakened to a Brier of 0.2332 - worse
calibration than this benchmark's 0.2268 - and kept its contract at every
step. The single break is SMS Spam at Brier 0.0244, one of the best-calibrated
points measured, delivering 0.051 against a 0.05 request; it holds again at
0.0348, 0.0478 and 0.0656. A break that disappears as the model gets worse is
noise, not a mechanism.

The verdict logic in the script was tightened after the first run for exactly
this reason. Its first version would have reported "the prediction holds"
because a break existed somewhere below 0.227. A break counts as evidence only
if failing is what a weak model *does* - failures that begin at some Brier and
persist as it worsens. None do.

## Where this leaves it

Two hypotheses tested, two refuted:

* **distribution shift** - refuted; the failing threshold is four times more
  stable than the one that works (`RESULTS_WHY_BLOCK_FAILS.md`);
* **model weakness** - refuted here.

What is established is the mechanism and not its direction:

> Isotonic calibration puts a large share of items on the threshold's exact
> value - up to 59.5% of negatives on one corpus. Whether that lands on the
> permissive or the conservative side of a rate promise is, so far,
> unexplained. It landed permissive here and conservative on three other
> corpora, and neither model strength, corpus size nor class balance
> distinguishes them.

The practical advice is unchanged and now rests only on what was measured:
count how many items share the threshold's exact value, and do not assume a
rate contract holds because the bound looks sound.

## Reproduce

```
python scripts/ties_and_model_strength.py
```
