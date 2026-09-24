# A second privacy corpus, annotated by humans — and one claim it does not support

`RESULTS_SYSTEM_ELSEWHERE.md` ends by naming what its three external corpora do
not establish:

> It does not establish that the protocol transfers, because none of these
> corpora is context-dependent sensitivity, none has thread structure, and none
> has two annotators.

The **Text Anonymization Benchmark** (Pilán et al., TACL 2022) — European Court
of Human Rights judgments annotated span by span for what must be masked so the
applicant cannot be re-identified — closes **two** of those three. Its `dev` and
`test` splits carry **207 documents annotated by two to ten people each**, and
every sentence sits inside a document, which is the same nuisance structure a
thread is.

The third it does **not** close, and that is reported below rather than assumed.

**8,182 sentences, 207 documents.** Nothing about the system is retuned.

```
python scripts/second_corpus_tab.py
```

The archive is fetched and its digest **checked against the one these numbers
were computed from** — upstream tracks a branch, so it can move under a reader:
`echr_dev.json` `8c3c7306f46b8d54…`, `echr_test.json` `cd0f0f15f84a8739…`.

---

## 0. Is this corpus context-dependent? **Not established.**

This was the reason for choosing TAB, and the first version of this document
asserted it, then measured it the wrong way, and reported 28.3% as though the
number settled it. It does not.

**What is true:** of 706 strings appearing in three or more documents, **200
(28.3%)** draw a different majority decision in a different document, with
annotator disagreement inside each document removed first.

**Why that is not enough.** Two checks were added, and both come back against
the claim:

| | |
|---|---|
| observed variation | 28.3% |
| **simulated from annotator noise alone** | **21.1%** |
| strings where the document's country predicts the decision (Holm-corrected, 57 tested) | **0** |
| nominal hits chance alone would give | ~2.9 |

The second check is the one that matters. If the decision were contextual,
something *about the document* should predict it — for `turkish`, `QUASI` where
the case is against Turkey and `NO_MASK` where it is not. Permuting decisions
across each string's documents destroys any such link while holding the amount
of variation fixed, so a string that varies at random scores zero by
construction.

Nothing survives correction. The four strongest candidates, and the one this
document built its argument on:

| string | documents | gain | p |
|---|---|---|---|
| `turkey` | 14 | +0.214 | 0.0260 |
| `united kingdom` | 95 | +0.011 | 0.0460 |
| `life imprisonment` | 11 | +0.273 | 0.0615 |
| `sweden` | 13 | +0.154 | 0.0805 |
| `turkish` | 43 | +0.046 | **0.2340** |

`turkish` — `NO_MASK` in 22 documents and `QUASI` in 21, the example the first
version of this document opened with — ranks **9th of the 57 tested**, at
p = 0.2340. The story told about it ("in a case about a Turkish applicant it
narrows the field") was an interpretation, not a measurement, and the
measurement does not support it.

> **The variation is real. Its cause is not shown to be context.** Country is
> one feature and a coarse one, so this does not prove the variation is noise
> either — but the claim that TAB is context-dependent is withdrawn, and with
> it the argument that TAB closes the third gap.

Everything below stands on its own and does not depend on this.

## 1. Is our agreement unusually low? Yes — and that is ours to explain

The paper leans on κ = 0.662 to argue the judgement is contested, and that
argument does real work: it is why the 5.5% automation ceiling is presented as a
property of the *task* rather than of our *labels*.

| label definition | positive | observed agr. | κ | κ (per-pair) | AC1 |
|---|---|---|---|---|---|
| `DIRECT` only | 5.6% | 0.990 | 0.897 | 0.943 | 0.989 |
| **`DIRECT` or `QUASI`** | 63.5% | 0.912 | **0.814** | 0.807 | 0.832 |
| confidential status | 4.3% | 0.963 | 0.516 | 0.282 | 0.960 |
| *ours (Enron, 2 annotators)* | *18.1%* | *0.876* | *0.662* | — | *0.803* |

The three definitions were named in the script before any of it ran.

### κ cannot be compared across corpora at different class balance

The rate that has to match is the **annotator-level** one, `(both + one/2) / n`,
because that is what the chance term is computed from. Ours is **24.3%**, not
the 18.1% label rate — our corpus calls a message sensitive only when *both*
annotators did.

| | κ | 95% CI |
|---|---|---|
| TAB, matched to our class balance | **0.800** | [0.776, 0.825] |
| ours | **0.662** | [0.610, 0.710] |

The interval now resamples **documents as well as dropped positives**. The first
version varied only which positives were dropped, so it carried no corpus
sampling uncertainty at all and came out implausibly tight at [0.789, 0.810].

> **Even matched on class balance, a published privacy benchmark agrees better
> than our annotators.** Contested judgement is not simply given in this kind of
> task. Our lower agreement is ours to explain, and the likeliest explanation is
> the instrument: TAB ships a 19 KB annotation guideline and each document names
> one narrow task ("annotate the document to anonymise the following person"),
> while our labels come from a broad genre taxonomy never designed to mark
> sensitivity. Nothing here establishes that TAB's annotators were more expert -
> only that they were better instructed, which is checkable from the archive.

**A defence this document previously made, and withdraws.** The first version
reported that the matched corpus was *more* contested than ours — 15.8% against
12.4% — and offered that as the answer to the obvious objection that matching
had thrown away the hard cases. Those two numbers are not the same quantity.
Ours has exactly two annotators, so 12.4% is a **pairwise** disagreement rate;
TAB sentences carry up to ten, where "anyone disagreed" is mechanically far
likelier. Like for like, the matched corpus sits at **7.3%** — *less* contested
than ours — and matching did lower TAB's own contestation (8.8% → 7.3%
pairwise), which is exactly what the check existed to detect. **The objection
stands unanswered.** The comparison still excludes our 0.662 by a wide margin,
but it is a weaker result than it was written up as.

The honest qualifier remains: TAB's own κ runs from 0.282 to 0.943 across the
three decisions above. Agreement in privacy annotation is enormously
task-dependent, and our 0.662 sits inside that range.

## 2. Our κ approximation does not bias the number

Our corpus records *how many* annotators chose a category, never *which*, so
`agreement_metrics.py` must assume the two marginals are equal. TAB records who
said what.

| | κ |
|---|---|
| per annotator pair, marginals **forced equal** | 0.802 |
| per annotator pair, **each rater's own** marginal | 0.807 |
| **difference** | **0.005** |

Both figures are now computed **per pair**, so the only thing varying is the
assumption. The first version compared a pooled-then-computed κ against a
per-pair-then-averaged κ, which changed the aggregation unit at the same moment
as the marginal assumption — the 0.007 it reported could not be attributed to
either.

And the assumption is not accidentally true: annotators' positive rates differ
by **0.152** on average within a document. It survives a spread that large.

> Our κ is a number about the annotators, not about the assumption.

**An earlier version of this check was worthless and said so in its output.** It
pooled every *ordered* annotator pair, counting (a, b) and (b, a) both ways,
which makes the two off-diagonal cells equal by construction — so the "exact" κ
was identical to the approximation it was meant to test, and it duly reported a
bias of exactly 0.000.

## 3. Both halves of the contract hold

The filter, unchanged, with **documents as groups** in place of threads:

| request | automated | allowed / blocked | leak | block precision | contract |
|---|---|---|---|---|---|
| 0.02 | 53.9% | 0 / 4,412 | 0.000 | 0.967 | held |
| 0.05 | **72.8%** | 1,546 / 4,412 | 0.020 | 0.967 | held |
| 0.10 | 80.1% | 2,141 / 4,412 | 0.046 | 0.967 | held |

**Block precision is checked, not assumed.** The filter promises a leak rate
*and* a block precision of at least 0.90; the first version of this script tested
the leak half alone and printed "held" for every row. A one-sided check on a
two-sided promise is not a check — a policy that blocks everything keeps any leak
contract trivially. Both halves are tested now, and both hold.

Consistent with `RESULTS_SYSTEM_ELSEWHERE.md`: the precision requirement fires
readily where the model supports it — 4,412 sentences auto-blocked here, zero on
our benchmark.

## 4. It declines what the annotators argued over

| | sentences | escalated |
|---|---|---|
| annotators disagreed | 1,452 | **39.5%** |
| annotators were unanimous | 6,730 | 24.5% |
| **difference** | | **+14.9 points**, 95% CI [+11.1, +18.6] |

Document-level bootstrap, 2,000 resamples, on its own RNG stream. The interval
excludes zero by a wide margin.

> The system sends a human what humans disagreed about, on a corpus it has never
> seen, under an annotation scheme it was never fitted to. The abstention tracks
> something real about the difficulty of the judgement, not just the quirks of
> our own labels.

This is the strongest result in this file and the one least touched by the
corrections above.

---

## What this does and does not establish

**Establishes.** Two of the three gaps `RESULTS_SYSTEM_ELSEWHERE.md` names are
closed: a second corpus with group structure and with multiple annotators. On it
both halves of the contract hold at every request, and abstention tracks human
disagreement with an interval far from zero. It also establishes that our κ is
not an artefact of the approximation our corpus forces — and that it is lower
than a guideline-driven benchmark's, which is a limitation to state rather
than explain away.

**Does not establish.** That TAB is context-dependent: 28.3% of strings vary
across documents against 21.1% from noise alone, and no string's decision is
predicted by the document's country once 57 tests are corrected. That was the
reason for choosing this corpus, and it is not shown. TAB is also court
judgments rather than workplace email, and its sensitive judgement is
re-identification risk rather than the paper's Section III definition.

**Still needs people.** Nothing here annotates our own corpus against Section
III. That remains the open item, and this does not substitute for it.
