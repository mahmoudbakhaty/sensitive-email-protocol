# Does the conclusion depend on our label mapping?

The first limitation the README lists is the one no amount of re-running can
check:

> Labels are the Enron subset's original category judgements mapped to our
> definition. The mapping is ours and is a source of construct error.

The Enron subset was annotated against a 1990s genre taxonomy — coarse genre,
forwarded material, primary topic, emotional tone — and nobody annotating it was
thinking about sensitivity. Seven of its category pairs were chosen as
*sensitive*:

| | |
|---|---|
| `1.2` | Purely Personal |
| `1.5` | Employment arrangements (job seeking, hiring, recommendations) |
| `2.8` | Legal documents (complaints, lawsuits, advice) |
| `3.4` | company image — changing / influencing |
| `3.5` | political influence / contributions / contacts |
| `3.10` | legal advice |
| `4.10` | secrecy / confidentiality |

That choice is ours. It is defensible, and it is not the only defensible one.

The usual remedy is to have people annotate a sample against the paper's own
Section III definition. That needs annotators, and it remains the right thing to
do — `scripts/make_annotation_sample.py` builds the instrument for it.

**This asks a different question, and it needs nobody.** Whether the mapping is
*right* is unanswerable without people. Whether it *matters* is measurable now.
If every defensible mapping yields the same conclusions, construct error in the
mapping cannot be what is driving them — the limitation stays real but stops
threatening the result.

```
python scripts/mapping_sensitivity.py
```

Seventeen mappings, declared in the script before any of them ran: ours, each of
our three weakest members dropped in turn, each of eight plausible additions
added in turn, and five holistic alternatives. Each is a position someone could
defend from the taxonomy text alone. The relabelling loop is checked against
`strengthen.build()` under the published mapping first, so these are numbers
about the corpus and not about a re-implementation.

---

## The table

`leak gap` is F1 under a thread-disjoint split minus F1 under a random one — the
published ladder's R2→R3 rung, run through `strengthen.run_cv` itself, paired
over 10 seeded partitions. `automated` and `leak` are the filter at a 5% request,
unchanged. Sorted by positive rate, which turns out to matter.

| mapping | sens. | rate | leak gap | ±2SE | automated | leak |
|---|---|---|---|---|---|---|
| legal only | 47 | 3.4% | **+0.0346** | 0.0324 | — | — |
| confidentiality only | 57 | 4.1% | −0.1057 | 0.0122 | — | — |
| personal only | 77 | 5.6% | −0.0663 | 0.0222 | 0.0% | 0.000 |
| narrow — the unambiguous three | 95 | 6.9% | −0.0595 | 0.0080 | 0.0% | 0.000 |
| ours − Employment arrangements | 191 | 13.8% | −0.0425 | 0.0180 | 0.8% | 0.000 |
| ours − political influence | 201 | 14.5% | −0.0108 | 0.0124 | 0.5% | 0.000 |
| ours − company image (changing) | 226 | 16.4% | −0.0268 | 0.0154 | 7.1% | 0.022 |
| **ours (published)** | 250 | 18.1% | **−0.0225** | 0.0164 | **5.5%** | 0.012 |
| ours + shame | 250 | 18.1% | −0.0225 | 0.0164 | 5.5% | 0.012 |
| ours + worry / anxiety | 255 | 18.4% | −0.0190 | 0.0132 | 6.2% | 0.016 |
| ours + Government action(s) | 257 | 18.6% | −0.0083 | 0.0108 | 5.3% | 0.008 |
| ours + company image (current) | 273 | 19.8% | −0.0181 | 0.0120 | 3.8% | 0.004 |
| ours + internal company policy | 281 | 20.3% | −0.0085 | 0.0174 | 4.9% | 0.007 |
| ours + alliances / partnerships | 281 | 20.3% | −0.0044 | 0.0132 | 4.5% | 0.011 |
| ours + Personal in professional context | 296 | 21.4% | −0.0068 | 0.0152 | 6.2% | 0.017 |
| ours + internal projects | 298 | 21.6% | −0.0232 | 0.0144 | 4.4% | 0.007 |
| broad — core plus every addition | 435 | 31.5% | **+0.0005** | 0.0102 | 6.2% | 0.018 |

Two mappings are too sparse for the filter to set a policy honestly (fewer than
60 positives) and their automation cells are left empty rather than filled with a
number the bound cannot support.

`ours + shame` is identical to `ours` throughout — no message carries `4.18` from
both annotators without already being sensitive. Every other row moves, so this
is a fact about the corpus, not a pipeline that ignores its input.

## A. The automation ceiling is robust

| | |
|---|---|
| published | **5.5%** |
| across 15 mappings | **0.0% to 7.1%**, median 4.9% |
| contracts held | **15 of 15** |

> **No defensible mapping clears more than 7.1% of this corpus.** Move the
> sensitivity line anywhere a reasonable person would put it and the system still
> sends almost everything to a human. The ceiling is a property of the task, not
> of our seven categories.

This is the stronger of the two results, and it is the one the system's headline
rests on.

## B. The leakage effect holds, and the one flag is what chance predicts

Counting sign alone would be the wrong test — a gap of +0.03 on a mapping with 47
positives is noise, and calling it a refutation would be the mirror image of
calling it a confirmation. A flag requires the gap to be positive by more than
two paired standard errors.

**But seventeen mappings is seventeen tests, and that has to be paid for.** A
one-sided "gap − 2 SE > 0" test has α = 0.023, so:

| | |
|---|---|
| published | −0.021 F1 |
| across 17 mappings | −0.106 to +0.035, median −0.019 |
| flags, uncorrected | **1** — `legal only`, 47 sensitive |
| **flags chance alone predicts** | **0.39** |
| **chance of at least one when every mapping is null** | **32%** |
| **flags surviving Holm correction** | **0** |

`legal only` clears the two-SE bar by 0.0022, on the smallest mapping in the
family, where the interval (±0.032) is almost as wide as the estimate. An earlier
version of this document reported "the effect reverses beyond noise under 1 of
17" and told the reader the paper should name where it fails. That reads as a
finding when it is the single most likely outcome under the null: **one flag in
seventeen is what you get a third of the time from nothing at all.**

`broad` at +0.0005 ± 0.0102 is not a reversal either; it is zero.

> The thread-disjoint split scores lower across the family, and **no mapping in
> it reverses the effect once the seventeen tests are corrected for.** The
> effect is not an artefact of our seven categories.

That is a statement about *this family*. Section C asks whether any mapping
breaks it, and finds four that do — so read the sentence above as bounded, not
universal.

**One claim from the earlier version is withdrawn.** It said the split "scores
lower under every mapping with a workable number of positives". `broad` has 435
positives — more than any other mapping in the family — and its gap is +0.0005.
That is zero, not lower, and the sentence was wrong as written.

## C. A family we chose is not a test — so here is a hostile one

Seventeen mappings we would all defend is exactly the objection a reviewer
raises: the family was picked by the people whose claim it supports. The
pattern above gives that objection a concrete shape — the effect shrinks as the
positive class widens, and `broad` at 31.5% is already at zero.

So the adversarial question is not whether our family is fair. It is whether a
mapping **exists** that reverses the effect. These are built only to widen the
positive class, past anything anyone would call a definition of sensitive:

| mapping | sens. | rate | leak gap | ±2SE | |
|---|---|---|---|---|---|
| every common category (indefensible, maximal) | 956 | 69.2% | **+0.0115** | 0.0054 | **reverses** |
| all of group 1 | 884 | 64.0% | **+0.0145** | 0.0066 | **reverses** |
| all of group 2 | 645 | 46.7% | +0.0020 | 0.0080 | — |
| all of group 3 | 253 | 18.3% | −0.0092 | 0.0082 | — |
| the 2 most frequent categories | 744 | 53.8% | **+0.0120** | 0.0062 | **reverses** |
| the 3 most frequent categories | 804 | 58.2% | +0.0060 | 0.0080 | — |
| the 5 most frequent categories | 907 | 65.6% | **+0.0070** | 0.0054 | **reverses** |
| the 8 most frequent categories | 937 | 67.8% | +0.0052 | 0.0062 | — |

**4 of them reverse it.** The effect is not universal, and an earlier draft of
this document would have gone on implying that it was.

What every reversal has in common is the thing that matters:

| | |
|---|---|
| mappings that **reverse** the effect | all at **53.8% positive or above** |
| mappings that **keep** it | all at **21.6% or below** |
| in between | no effect either way |

> **The effect is a property of the minority-class regime.** When the sensitive
> class is rare it concentrates in a few threads, so thread membership carries
> real information and thread-disjoint splitting takes it away. Make a majority
> of the corpus sensitive and thread membership stops predicting anything —
> then the grouped split scores slightly *higher*.

Our corpus sits at 18.1% positive, and any definition of "sensitive" worth the
name leaves it a minority. But the paper should state the boundary rather than
claim the effect holds everywhere, because it does not.

**This also settles the post-hoc observation below.** That section noticed the
relationship after looking at the table and flagged it as untested. The search
here varies prevalence deliberately, across a range the defensible family could
not reach, and the relationship holds across all of it. It is no longer a
noticing — though the counterexample `legal only` remains, and is discussed
there.

## An observation, post hoc, and weaker than it first looked

Sorted by positive rate, the three largest effects sit among the narrowest
mappings: `confidentiality only` (4.1%, −0.106), `personal only` (5.6%, −0.066),
`narrow` (6.9%, −0.060). The smallest sits at the widest: `broad` (31.5%,
+0.0005).

A plausible reading: when the sensitive class is rare it concentrates in a few
threads, so knowing a thread tells you a lot and thread-disjoint splitting takes
a lot away. Spread sensitivity across a third of the corpus and thread membership
stops carrying the signal.

**The exception is the narrowest mapping of all.** `legal only` sits at 3.4%
positive — narrower than any of the three — and its gap is **+0.035**, the wrong
way. An earlier version of this section listed the three that fit and did not
mention it, which is choosing the evidence. With it included the relationship is
suggestive, not monotone, and it rests on four sparse mappings whose intervals
are wide.

If the reading is right, the paper's −0.021 is a **conservative** figure, measured
at a middling prevalence. That was noticed after looking at the table, has a
counterexample in its own data, and is not a tested hypothesis.

---

## What this does and does not establish

**Establishes.** The two conclusions the release rests on survive being
relabelled by anyone who reads the same taxonomy and disagrees with us. The
automation ceiling never rises above 7.1%; the leakage effect holds across the
family and no mapping in it reverses the effect after correction.

It also establishes a **boundary**, which is more than the family alone could
give. A hostile search finds four mappings that do reverse the effect, every one
of them at 53.8% positive or above, while every mapping that keeps it sits at
22% or below. The effect belongs to the minority-class regime — which is the
regime any real definition of sensitive puts the task in, but the paper should
say so rather than claim it holds everywhere.

**Does not establish.** That the effect is universal — Section C shows where it
stops. Nor that our mapping is *correct*. Nothing here compares any
label to the paper's Section III definition. Seventeen mappings drawn from the
same taxonomy share the taxonomy's blind spots — if the 1990s genre categories
systematically miss a kind of sensitivity, every mapping in this family misses it
too, and this analysis cannot see that.

**Still needs people.** Annotating a sample against Section III. This bounds how
much that could matter; it does not replace it.
