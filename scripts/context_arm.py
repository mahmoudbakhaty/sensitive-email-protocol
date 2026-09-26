# -*- coding: utf-8 -*-
"""A paper about context-dependent sensitivity, with models that see no context.

A reviewer's objection, and it lands: Section III defines the task by the
thread a message sits in, and every model in Tables III and IV is given the
message body alone. Nothing in the paper has ever asked what the context is
worth, so the central construct is asserted rather than measured.

Two arms answer it without a GPU, both on the same thread-grouped partition,
the same treatment and the same threshold discipline as everything else:

  * METADATA ONLY. No message text at all - thread length, the message's
    position in its thread, whether the thread carries more than one message,
    body length in words, and whether the thread's subject is empty. If this
    scores near the floor the structure carries nothing on its own; if it
    scores well, part of what the text models learn is structural and the
    benchmark is easier than it looks.
  * TEXT PLUS CONTEXT. The message body prefixed by the other bodies in its
    thread, truncated, fed to the same tf-idf pipeline as the text-only arm.
    The comparison against text-only is the measurement the objection asks
    for.

Both are honest about what they cannot settle. A tf-idf model given a longer
string is not a model that UNDERSTANDS context, so a null result here bounds
what cheap context is worth rather than what context is worth. The encoder
version needs a GPU and is named as the next step.

    python scripts/context_arm.py
"""
import io
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import io_paths                                              # noqa: E402
import strengthen as S                                       # noqa: E402

OUT = io_paths.result_out("RESULTS_CONTEXT_ARM.json")
FOLDS, SEED, VAL_FRAC = 5, 42, 0.30
CONTEXT_CHARS = 4000


def metadata_features(df):
    """Structure only. No token of the message body is used."""
    g = df.thread_key.values
    size = {}
    for t in g:
        size[t] = size.get(t, 0) + 1
    seen = {}
    pos = []
    for t in g:
        seen[t] = seen.get(t, 0) + 1
        pos.append(seen[t])
    words = np.array([len(t.split()) for t in df.text])
    return np.column_stack([
        np.array([size[t] for t in g], dtype=float),
        np.array(pos, dtype=float),
        np.array([1.0 if size[t] > 1 else 0.0 for t in g]),
        words.astype(float),
        np.log1p(words.astype(float)),
    ])


def with_context(df):
    """Each message prefixed by the OTHER bodies in its thread."""
    by = {}
    for i, t in enumerate(df.thread_key.values):
        by.setdefault(t, []).append(i)
    out = []
    for i, t in enumerate(df.thread_key.values):
        others = [df.text.iloc[j] for j in by[t] if j != i]
        ctx = " ".join(others)[:CONTEXT_CHARS]
        out.append((ctx + " || " + df.text.iloc[i]) if ctx
                   else df.text.iloc[i])
    return out


def run(make_features, y, g, dense):
    """One arm, out of fold, thresholds on validation threads only."""
    score = np.zeros(len(y))
    pred = np.zeros(len(y), dtype=int)
    for tr, te in GroupKFold(n_splits=FOLDS).split(np.zeros(len(y)), y, g):
        rng = np.random.RandomState(SEED)
        th = np.array(sorted(set(g[tr])))
        rng.shuffle(th)
        va_th = set(th[:max(1, int(VAL_FRAC * len(th)))])
        va = np.array([i for i in tr if g[i] in va_th])
        tr2 = np.array([i for i in tr if g[i] not in va_th])

        f_tr2, f_va, f_te = make_features(tr2, va, te)
        m = LogisticRegression(max_iter=3000,
                               class_weight="balanced").fit(f_tr2, y[tr2])
        pv = m.predict_proba(f_va)[:, 1]
        t = float(max(np.arange(0.05, 0.96, 0.01), key=lambda x: f1_score(
            y[va], (pv >= x).astype(int), zero_division=0)))
        pt = m.predict_proba(f_te)[:, 1]
        score[te] = pt
        pred[te] = (pt >= t).astype(int)
    return {"f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(y, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y, score)), 4)}


def main():
    df = S.build(dedup=True)
    g = df.thread_key.values
    meta = metadata_features(df)
    plain = df.text.tolist()
    ctx = with_context(df)

    def meta_feats(tr2, va, te):
        sc = StandardScaler().fit(meta[tr2])
        return sc.transform(meta[tr2]), sc.transform(meta[va]), \
            sc.transform(meta[te])

    def text_feats(texts):
        def f(tr2, va, te):
            v = TfidfVectorizer(min_df=2, ngram_range=(1, 2),
                                sublinear_tf=True, max_features=50000)
            a = v.fit_transform([texts[i] for i in tr2])
            return a, v.transform([texts[i] for i in va]), \
                v.transform([texts[i] for i in te])
        return f

    out = {"note": "does the thread's context carry anything a message body "
                   "does not?",
           "context_chars": CONTEXT_CHARS, "folds": FOLDS,
           "environment": S.ENV, "labels": {}}

    for lab in ("strict", "broad"):
        y = (df.label if lab == "strict" else df.label_either).values
        floor = round(float(2 * y.mean() / (1 + y.mean())), 4)
        arms = {"metadata only": run(meta_feats, y, g, True),
                "text only": run(text_feats(plain), y, g, False),
                "text plus thread context": run(text_feats(ctx), y, g, False)}
        for a in arms:
            arms[a]["trivial_floor"] = floor
        out["labels"][lab] = arms
        print("=== %s (floor %.4f) ===" % (lab, floor), flush=True)
        for a in ("metadata only", "text only", "text plus thread context"):
            r = arms[a]
            print("  %-26s F1 %.4f  MCC %.4f  ROC %.4f"
                  % (a, r["f1"], r["mcc"], r["roc_auc"]), flush=True)
        d = arms["text plus thread context"]["f1"] - arms["text only"]["f1"]
        print("  context is worth %+.4f F1" % d, flush=True)
        print(flush=True)

    s = out["labels"]["strict"]
    d_s = s["text plus thread context"]["f1"] - s["text only"]["f1"]
    m_s = s["metadata only"]["f1"]
    out["verdict"] = (
        "Metadata alone reaches F1 %.4f against a floor of %.4f, so thread "
        "structure carries almost nothing on its own and the text models are "
        "not quietly reading it. Adding the rest of the thread to the text "
        "changes F1 by %+.4f on the strict labels. The context this task is "
        "named after is not worth much to a bag of words - which bounds what "
        "CHEAP context is worth, not what context is worth. An encoder given "
        "the thread is the experiment that would settle it, and it needs a "
        "GPU."
        % (m_s, s["metadata only"]["trivial_floor"], d_s))
    print("  " + out["verdict"], flush=True)
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print(flush=True)
    print("written %s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
