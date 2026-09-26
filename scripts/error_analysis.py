# -*- coding: utf-8 -*-
"""Where the models are wrong, and whether "wrong" is the right word.

The paper had no error analysis. A reviewer asks for one, and for this paper
it is not decoration: the whole argument is that the difficulty lives in a
contested judgement rather than in the classifier, and that claim is testable
on the errors themselves.

The benchmark makes it testable because it carries two annotators. Under the
strict labels a message counts as sensitive only when BOTH called it so, which
means the 172 messages exactly one annotator called sensitive are scored as
NEGATIVES. If a model flags one of those, the scoring calls it a false
positive - but a trained human called it sensitive too.

So the question is whether the false positives concentrate there. If they sit
on the contested messages at their base rate, the errors are ordinary and the
contested set explains nothing. If they are enriched, then a measurable share
of what the metrics call error is the model siding with the annotator who was
outvoted, and the gap between the reported score and the annotator reference
of Section VII-C is smaller than the score alone suggests.

Enrichment is quoted with a thread-clustered bootstrap interval, because
messages in a thread are not independent - the same control the rest of the
paper uses.

    python scripts/error_analysis.py
"""
import io
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import io_paths                                              # noqa: E402
import strengthen as S                                       # noqa: E402

OUT = io_paths.result_out("RESULTS_ERROR_ANALYSIS.json")
N_BOOT, SEED = 2000, 42
MODELS = (("fine-tuned encoder", "transformer_grouped_strict"),
          ("logistic regression", "LogReg_grouped_strict"),
          ("linear SVM", "LinearSVM_grouped_strict"))

# Enron bodies carry real addresses and numbers. Examples are quoted for
# illustration only, so anything that looks like a contact detail is masked
# before it reaches a released file.
MASK = (
    # Enron's internal routing form, "Jane Q Smith/HOU/ECT@ECT", which is
    # where most of the personal names in these bodies actually sit.
    (re.compile(r"[A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,3}"
                r"/[A-Za-z]+(?:/[A-Za-z]+)*(?:@[\w.-]+)?"), "<person>"),
    # Header remnants that survive the body cut.
    (re.compile(r"(?im)^[ \t]*(From|To|Cc|Sent by):.*$"), r"\1: <person>"),
    (re.compile(r"(?m)^[ \t]*[A-Z][a-z]+ ?[-:,] "), "<person> - "),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
    (re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"), "<phone>"),
    (re.compile(r"\b\d{2,}[-\s]?\d{3,}\b"), "<number>"),
)


def redact(t, n=240):
    """Mask before collapsing the whitespace: two of the patterns are
    anchored to a line start that joining the text would destroy."""
    for pat, sub in MASK:
        t = pat.sub(sub, t)
    t = " ".join(t.split())
    return t[:n] + ("..." if len(t) > n else "")


def boot_ratio(num_mask, den_mask, base, groups, n=N_BOOT, seed=SEED):
    """Thread-clustered interval for share(num within den) / base."""
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(groups)))
    by = {t: np.where(groups == t)[0] for t in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), True)
        idx = np.concatenate([by[t] for t in pick])
        d = den_mask[idx].sum()
        if d == 0:
            continue
        out.append((num_mask[idx].sum() / d) / base)
    a = np.asarray(out)
    return [round(float(np.percentile(a, 2.5)), 3),
            round(float(np.percentile(a, 97.5)), 3)]


def main():
    P = json.load(io.open(io_paths.result_in("PREDICTIONS_FINAL.json",
                                             required=True), encoding="utf-8"))
    df = S.build(dedup=True)
    g = df.thread_key.values
    y = df.label.values
    ye = df.label_either.values
    contested = (ye == 1) & (y == 0)
    clean_neg = (ye == 0)
    words = np.array([len(t.split()) for t in df.text])

    base = contested.sum() / float((y == 0).sum())
    print("corpus %d | sensitive to both %d | contested %d | agreed harmless "
          "%d" % (len(y), int(y.sum()), int(contested.sum()),
                  int(clean_neg.sum())), flush=True)
    print("  a contested message is %.1f%% of all negatives" % (100 * base),
          flush=True)
    print(flush=True)

    out = {"note": "are the false positives the messages the annotators "
                   "disagreed about?",
           "n": int(len(y)), "contested": int(contested.sum()),
           "contested_share_of_negatives": round(float(base), 4),
           "draws": N_BOOT, "seed": SEED, "models": {}}

    print("  %-21s %5s %5s %9s %7s %s"
          % ("model", "FP", "FN", "in FP", "lift", "95% CI"), flush=True)
    for nice, key in MODELS:
        pred = np.array(P[key]["pred"])
        fp = (pred == 1) & (y == 0)
        fn = (pred == 0) & (y == 1)
        share = (fp & contested).sum() / float(max(1, fp.sum()))
        lift = share / base
        ci = boot_ratio(fp & contested, fp, base, g)
        rec = {"false_positives": int(fp.sum()),
               "false_negatives": int(fn.sum()),
               "contested_share_of_fp": round(float(share), 4),
               "lift": round(float(lift), 3), "lift_ci95": ci,
               "enriched": bool(ci[0] > 1.0),
               "flag_rate_contested": round(float(pred[contested].mean()), 4),
               "flag_rate_agreed_harmless": round(
                   float(pred[clean_neg].mean()), 4),
               "median_words_fp": int(np.median(words[fp])),
               "median_words_fn": int(np.median(words[fn])),
               "median_words_corpus": int(np.median(words))}
        out["models"][nice] = rec
        print("  %-21s %5d %5d %8.1f%% %7.2f  [%.2f, %.2f]%s"
              % (nice, fp.sum(), fn.sum(), 100 * share, lift, ci[0], ci[1],
                 "" if ci[0] > 1.0 else "  (includes 1)"), flush=True)

    # Examples, from the strongest model, for the paper to quote.
    pred = np.array(P[MODELS[0][1]]["pred"])
    sc = np.array(P[MODELS[0][1]]["score"])
    fp_c = np.where((pred == 1) & (y == 0) & contested)[0]
    fn_i = np.where((pred == 0) & (y == 1))[0]
    ex = {"false_positive_on_contested":
          [{"score": round(float(sc[i]), 4), "words": int(words[i]),
            "thread": str(df.thread_key.iloc[i])[:60],
            "text": redact(str(df.text.iloc[i]))}
           for i in fp_c[np.argsort(-sc[fp_c])][:3]],
          "false_negative":
          [{"score": round(float(sc[i]), 4), "words": int(words[i]),
            "thread": str(df.thread_key.iloc[i])[:60],
            "text": redact(str(df.text.iloc[i]))}
           for i in fn_i[np.argsort(sc[fn_i])][:3]]}
    out["examples"] = ex
    out["redaction"] = (
        "Example bodies are quoted from the public Enron archive, which this "
        "artifact rebuilds rather than redistributes. Masked: the internal "
        "routing form, From/To/Cc lines, e-mail addresses, telephone numbers "
        "and long digit strings. NOT masked: first names appearing inside a "
        "sentence, which no pattern catches without a name list and which "
        "over-masking would take real content with. The examples are "
        "illustrative and nothing in the paper depends on them.")

    lifts = [m["lift"] for m in out["models"].values()]
    allenr = all(m["enriched"] for m in out["models"].values())
    out["verdict"] = (
        "every model's false positives are enriched for the messages the two "
        "annotators split on, by %.2f to %.2f times their share of the "
        "negatives, and every interval excludes one. A measurable part of "
        "what the metrics score as error is the model agreeing with the "
        "annotator who was outvoted." % (min(lifts), max(lifts))
        if allenr else
        "the false positives are NOT reliably enriched for the contested "
        "messages (lifts %.2f to %.2f, at least one interval includes one), "
        "so the contested set does not explain the error."
        % (min(lifts), max(lifts)))
    print(flush=True)
    print("  " + out["verdict"], flush=True)
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print(flush=True)
    print("written %s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
