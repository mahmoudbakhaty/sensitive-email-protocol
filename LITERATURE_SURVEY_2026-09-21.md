# Where the field stands, and where this work sits in it

Survey date: 21 September 2026. Every figure is attributed to the source it was
read from. Figures marked [primary] were read out of the paper's own PDF or HTML
in this session; [secondary] means a search summary only, and must be checked
before it enters the thesis.

---

## 1. The four lines of work

### Line A - sensitive by TYPE (PII, named entities, patterns)

The mature line. The target is a surface form, so the task is well posed and the
numbers are high.

| Work | Data | Reported |
|---|---|---|
| Mehdy & Mehrpouyan 2021 (arXiv:2108.08483) | Tweets | 77.4% acc. disclosure detection; **99% acc. information-type** |
| Gambarelli et al. 2022, SPeDaC (arXiv:2208.06216) | Constructed benchmark | **RoBERTa 98.20%** binary; DeBERTa 95.81% on 5 classes; **77.63% on 61 classes** |
| Shahriar et al. 2024, DEXA | PII in unstructured text | comparative transformer study |
| Muralitharan & Arumugam 2024, NCAA | Documents | Privacy BERT-LSTM |
| Szawerna et al. 2024, CALD-pseudo | Swedish learner essays | PII detection |
| Rajgarhia et al. 2025 (arXiv:2510.07551) | Multilingual | +82% over fine-tuned NER, +17% over zero-shot LLMs (relative) |
| Elbarbary et al. 2025, IJSSE | Arabic, hybrid BERT-NER + rules | **F1 = 0.92** unstructured; 0.88 structured |

**Note the SPeDaC gradient**: 98.2% on 2 classes, 95.8% on 5, 77.6% on 61. The
headline number is a property of how coarse the label set is.

### Line B - sensitive by CONTEXT (our problem)

Smaller, younger, and the numbers fall sharply once the data is real.

| Work | Data | Reported |
|---|---|---|
| Neerbek 2020 (arXiv:2008.10863) | recursive NN for context | figures not in abstract |
| Ahmed et al. 2021, Internet of Things 16:100444 | text + images | 95% images, 80% text [secondary] |
| Kuzina et al. 2023, CASSED, ESWA 223:119924 | structured columns | **see Section 3 - the key datapoint** |
| Anand et al. 2023, COMSNETS | data-constrained | finance and health domains |
| Qawara & Alhindi 2026, Information 17(7):663 | social media, political/ethnic | DistilRoBERTa > ALBERT |
| **Telkamp & Hulsebos 2025 (arXiv:2512.04120)** | open data portals | *Towards Contextual Sensitive Data Detection*: recall **94% vs 63%** for commercial tools [primary] |

### Line C - sensitivity review (information retrieval)

The most methodologically careful line, and the closest to our concerns.

| Work | Contribution |
|---|---|
| McDonald et al. 2019, 2020 | how classifier accuracy and confidence affect human reviewers |
| Sayed et al. 2022, ECIR | intrinsic vs extrinsic evaluation compared directly |
| Branting et al. 2025, AI & Law 33(1) | decision support for government records |
| **McKechnie et al. 2026, SIGIR (arXiv:2606.27559)** | **SARA: 150 queries, 11,471 assessments on the SAME Berkeley Enron subset we use** |

SARA matters: it is built on the same 1,702 Berkeley-annotated emails. It adds
*relevance* judgements for retrieval, not independent sensitivity labels, so it
gives us no second opinion on our labels - but it is the nearest neighbouring
resource, and a reviewer will expect us to know it.

### Line D - evaluation integrity (the genre our paper belongs to)

The important discovery of this survey. **2026 has an established line of papers
whose contribution is auditing an existing benchmark, not beating it.**

| Work | Field | What they found |
|---|---|---|
| Kapoor & Narayanan 2023, Patterns | cross-discipline | leakage taxonomy; hundreds of affected studies |
| **Ivchenko 2026, CTSCAN (arXiv:2604.15561)** | chest CT | slices of one study split across train/test. **Dice 0.6665 -> 0.2066 under patient-disjoint evaluation: 0.4599 absolute, 69.00% relative** [primary] |
| Clinical AI protocol audit 2026 (PubMed 42601029) | DAIC-WOZ / E-DAIC | participant leakage across benchmark reissues |
| **Zainab et al. 2026 (arXiv:2608.16928)** | document sensitivity | **label leakage: classification markers left inside document bodies**; Strategic 16K, BERT 89.14% acc / 89.33% F1 [primary] |
| Near-OOD leakage fingerprint 2026 (arXiv:2607.19393) | OOD detection | contamination masquerading as detector quality |

**NeurIPS 2026 opened an Evaluations & Datasets track** whose call invites
submissions that "analyze strengths, limitations, or failure modes of existing
benchmarks" and "propose new evaluation protocols", and states they "need not
introduce novel models or surpass prior results". That is a venue description
written for a paper like ours.

---

## 2. What people have actually achieved

Ordered by how contested the ground truth is - which predicts the score better
than the method does.

| Task | Ground truth | Best reported |
|---|---|---|
| Information type of a disclosure | objective category | 99% acc. |
| Binary PII present / absent | surface pattern | 98.2% acc. |
| Document classification level | **the official marking on the document** | 89.3% F1 (Zainab) |
| Arabic PII, hybrid | surface pattern | 0.92 F1 |
| 61-way fine-grained PII | objective but fine | 77.6% acc. |
| Structured columns, **in-domain synthetic** | generated | **0.996 macro-F1** |
| Structured columns, **independent real tables** | real tables | **0.349 / 0.501 macro-F1** |
| **Context-dependent email, ours** | **contested human judgement** | **0.359 F1** |

---

## 3. The central finding of this survey: the CASSED case

All figures in this section are [primary] - read from the papers themselves in
this session.

### Step 1. The original paper evaluates in-domain and concludes robustness

Kuzina et al. built DeSSI themselves ("we created our own relational data
models"), 31,000+ columns, split **randomly** 60/20/20. Per-class F1 on the
DeSSI test part runs 0.98-1.00 across every label. They did test on real data -
an in-house NDA dataset - and reported a **weighted F1 of 0.976, a drop of
"around 0.02"**, concluding verbatim:

> "we consider that the small drop in performance only attests to the soundness
> and robustness of the proposed approach."

### Step 2. An independent re-evaluation finds the opposite

Ntwali, Ruck & Heckmann (arXiv:2506.22305, LLM-DPM 2025), Table 2, macro-F1:

| Model | DeSSI (synthetic) | Kaggle | OpenML | MIMIC-Demo-Ext |
|---|---|---|---|---|
| **CASSED** | **0.996** | **0.349** | **0.501** | 0.724 |
| **GPT-4o** | 0.766 | **0.902** | **0.964** | 0.829 |

CASSED "performed nearly perfectly" on the synthetic set and "dropped
drastically" on real tables - a **65% relative fall** on Kaggle.

### Step 3. Two design choices hid it, and both are on our protocol's list

- The only "real-world" test in the original paper was *in-house* data collected
  alongside the training data: near-domain, not independent. Genuinely
  out-of-domain data was never tried.
- The headline was **weighted** F1, dominated by the frequent "Other data"
  class. Ntwali et al. report **macro** F1. Same model, same task: 0.976
  weighted in-house versus 0.349 macro out-of-domain.

This is our paper's argument, demonstrated by independent parties against a
published paper in a strong journal. It is the best external validation the
thesis has.

### The honest counterweight - do not overstate this

**GPT-4o did not collapse.** It scored 0.902 and 0.964 on the same real tables.
So the CASSED case does **not** show that the task is intrinsically hard. It
shows that *a model fine-tuned on synthetic in-domain data does not transfer*,
and that an in-domain evaluation concealed it.

**But the evidence on LLMs is mixed, not settled.** Antypas et al. (WOAH 2025,
pp. 17-31) evaluate sensitive-content classification on social media and reach
the opposite conclusion: GPT-4o zero-shot gets binary macro-F1 **75.7**, while a
fine-tuned llama3-8b reaches **85.6**. They state that "the zero-shot models do
not reach the performance levels of their fine-tuned counterparts", with GPT-4o
the only partial exception. Few-shot prompting narrows the gap but does not
close it.

So one paper has an LLM rescuing a task and another has it losing to a
fine-tuned baseline, and the two tasks differ in exactly the way that matters:
CASSED's is structured columns where world knowledge about data types applies
directly, Antypas's is contested judgement about content. Ours is the second
kind. That is a reason to run the experiment rather than cite our way out of
it.

Our claim must stay exactly that narrow: **the evaluation design inflates the
number**, not "the task is unsolvable". And we owe the reader the obvious next
step - a strong instruction-tuned LLM evaluated under our protocol. We withdrew
a Llama run; that gap is now a named limitation, not an omission.

---

## 4. Where we sit

### What is genuinely ours

1. **Inter-annotator agreement on this task.** Zainab et al. have no annotators;
   their labels are the cables' own classification stamps. Line B papers rarely
   report agreement at all. Ours is 0.662 on 1,382 messages.
2. **The agreement-derived reference point.** One annotator scored as a
   classifier against the other reaches F1 = 0.744. This survey found it
   computed nowhere else in this literature.
3. **Thread-grouped splitting for email.** CTSCAN does patient-disjoint, the
   clinical audit participant-disjoint; nobody has done thread-disjoint for
   sensitivity in email.
4. **The library-version finding and the published fold fingerprint.** A
   grouped-CV result moving 0.024 F1 because scikit-learn 1.9 changed GroupKFold
   to stable sorting, with the fold hash released so a reader can verify the
   partition. Not standard practice anywhere this survey looked.

### What is NOT ours, and must be cited

1. **Leakage inflates sensitivity classification.** Zainab et al. (5 Aug 2026)
   got there first in writing, six weeks before our draft, claiming "the first
   fully reproducible sensitivity classification benchmark constructed under
   explicit leakage-controlled conditions". Narrower - WikiLeaks PlusD only -
   but the priority is theirs.
2. **The synthetic-to-real collapse.** CASSED produced it in 2023; Ntwali et al.
   measured and named it in 2025.
3. **The audit-an-existing-benchmark genre.** CTSCAN and the clinical protocol
   paper established the form in 2026.

### What is contested, and we must stop asserting it

**"Human ceiling" is the wrong name.** Richie, Grover & Tsui (BioNLP 2022),
*Inter-annotator agreement is not the ceiling of machine learning performance*,
show by simulation that a well-specified model can exceed IAA, especially when
annotators are noisy and differ in their classification functions. Boguslav &
Cohen (2017) gave real counterexamples earlier.

A second, sharper problem with our framing: our labels are the **consensus** of
the two annotators, so a model predicts the consensus, not either individual.
Pairwise agreement is not a bound on that.

The defensible version:

- 0.744 measures how contestable the judgement is. It is a reference point, not
  a hard ceiling.
- Reaching F1 = 1.000 would require being near-perfect on exactly the 172 cases
  two trained annotators settled differently. That argument survives intact.
- Our models reach 48% of 0.744, so the theoretical dispute does not touch our
  conclusions either way.

---

## 5. What to do about it

**Add to the paper (6 references):**

| Ref | Why |
|---|---|
| Zainab et al. 2026 | closest prior work; priority on the leakage claim |
| Ivchenko 2026 (CTSCAN) | establishes the genre; near-identical structure |
| Ntwali et al. 2025 | the measured synthetic-to-real collapse, with numbers |
| Richie et al. 2022 | the counter-argument to our ceiling framing |
| Telkamp & Hulsebos 2025 | nearest Line B work; defines sensitivity contextually as we do |
| Antypas et al. 2025 (WOAH) | a properly reported zero-shot LLM reference point, replacing the withdrawn Llama run |

**Rewrite** Section VII-C from "human ceiling" to "annotator agreement as a
reference point", engaging Richie et al. directly.

**Add to Related Work** a short subsection placing the paper in Line D, naming
CTSCAN and Zainab et al., and saying plainly what is ours and what is not.

**Add the CASSED case to the Discussion** as a worked example of the failure the
protocol is designed to catch - including the weighted-versus-macro point, which
our own trivial-floor convention already addresses.

**Add an explicit limitation:** no strong instruction-tuned LLM has been
evaluated under this protocol. Ntwali et al. show GPT-4o recovering most of the
gap on structured data; whether that transfers to contested contextual judgement
in email is open, and is the clearest next experiment.

**For the thesis:** Line D is the home. The contribution is not a classifier. It
is that this task has a contested ground truth, that its published numbers do
not survive leakage control, and that the protocol for checking is released and
checkable.
