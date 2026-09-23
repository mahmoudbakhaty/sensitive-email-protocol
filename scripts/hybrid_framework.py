# -*- coding: utf-8 -*-
"""A hybrid LLM-based framework for context-dependent sensitive data classification.

The registered thesis title promises a framework. What exists so far is a
protocol, a benchmark, and four models compared against each other under it.
The components have never been combined. This builds the combination.

Three signals, each seeing the same message and answering a different question:

  * a general-purpose instruction-tuned model, asked whether the message is
    sensitive, read as P(Yes) from the answer token - semantic judgement;
  * a fine-tuned encoder - task-specific representation;
  * tf-idf with logistic regression - lexical and distributional evidence.

An earlier fusion was cut from the paper on the supervisor's review, for being
miscalibrated and unvalidated. Both are addressed here rather than repeated.
Every component score is calibrated on held-out validation threads, the fusion
weights are fitted on those same validation threads, and the decision
threshold is chosen there too. The test fold sees none of it.

The protocol is unchanged: thread-disjoint folds, validation-only thresholds,
the trivial all-positive floor reported beside every score, MCC beside F1.

A note on what to expect. Every component already lands in a narrow band with
overlapping intervals, so the fusion may not beat them by much. That is a
result either way: if the signals are redundant the difficulty is in the task
rather than in the representation, and the protocol is what lets us say so.

Kaggle: GPU, internet on. Roughly two hours.
"""
import glob
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "scikit-learn==1.9.1"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "bitsandbytes"], check=False)

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.calibration import IsotonicRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

assert sklearn.__version__ == "1.9.1", sklearn.__version__
assert torch.cuda.is_available(), "no GPU: enable the accelerator"
DEV = "cuda"

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ENC_MODEL = "roberta-base"
DATA_DIR = "enron_with_categories"
OUT_JSON = "RESULTS_HYBRID.json"
SCORES_JSON = "SCORES_HYBRID.json"

MAX_LEN, BATCH, LR, EPOCHS = 384, 16, 2e-5, 6
FOLDS, SEED, VAL_FRAC = 5, 42, 0.30
MAX_CHARS = 4000
N_BOOT = 2000

SENS = {(4, 10), (3, 10), (2, 8), (3, 5), (3, 4), (1, 5), (1, 2)}
EMPTY = {(1, 7), (1, 8)}
WS = re.compile(r"\s+")
SUBJ = re.compile(r"^\s*((re|fw|fwd|aw)\s*:\s*)+", re.I)

SYSTEM = (
    "You judge whether a work email is sensitive in the context-dependent "
    "sense: its disclosure outside the company would cause harm because of "
    "what it means, not because it contains an identifier.\n\n"
    "Sensitive covers confidentiality and secrecy, legal advice and legal "
    "documents, political influence and contacts, deliberate management of "
    "company image, employment and compensation arrangements, and purely "
    "personal matters.\n\n"
    "Not sensitive covers routine business: logistics, scheduling, published "
    "news, public filings, newsletters and ordinary operational traffic. The "
    "mere presence of a name, address or phone number does not make a message "
    "sensitive.\n\n"
    "Answer with exactly one word: Yes or No.")


def fetch_data():
    if os.path.isdir(DATA_DIR):
        return
    import tarfile
    import urllib.request
    urllib.request.urlretrieve(
        "https://bailando.berkeley.edu/enron/enron_with_categories.tar.gz",
        "enron_with_categories.tar.gz")
    with tarfile.open("enron_with_categories.tar.gz") as tf:
        tf.extractall(".")


def build():
    rows, seen = [], set()
    for cf in sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.cats"))):
        cats = [tuple(map(int, l.strip().split(",")))
                for l in open(cf) if l.strip().count(",") == 2]
        if any((t, s) in EMPTY for t, s, _ in cats):
            continue
        raw = open(cf.replace(".cats", ".txt"), encoding="utf-8",
                   errors="replace").read()
        i = raw.find("X-FileName:")
        body = (raw[raw.find("\n", i) + 1:] if i >= 0 else raw).strip()
        if len(body.split()) < 30:
            continue
        h = hashlib.md5(WS.sub(" ", body.lower()).strip().encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        # A tab or space only: \s* crosses the newline and, on a message whose
        # Subject header is empty, captures the next header instead.
        m = re.search(r"^Subject:[ \t]*(.*)$", raw, re.M | re.I)
        subj = m.group(1).strip() if m else ""
        rows.append({"label": int(any((t, s) in SENS and f >= 2
                                      for t, s, f in cats)),
                     "label_either": int(any((t, s) in SENS
                                             for t, s, f in cats)),
                     "subject": subj,
                     "thread_key": WS.sub(" ", SUBJ.sub("", subj).strip().lower())
                                   or ("m:" + h[:10]),
                     "text": body})
    return pd.DataFrame(rows)


class DS(Dataset):
    def __init__(self, texts, labels):
        self.t, self.l = texts, labels

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return self.t[i], int(self.l[i])


def llm_scores(texts, subjects):
    """P(Yes) per message, once. Zero-shot, so it does not depend on the fold
    or on which label set is being predicted, and is reused throughout."""
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, quantization_config=cfg, device_map="auto")
        print("  LLM loaded in 4-bit", flush=True)
    except Exception as exc:
        print("  4-bit unavailable (%s); fp16" % type(exc).__name__, flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, dtype=torch.float16, device_map="auto")
    model.eval()

    def ids_for(word):
        out = set()
        for w in (word, " " + word, word.lower(), " " + word.lower()):
            t = tok.encode(w, add_special_tokens=False)
            if t:
                out.add(t[0])
        return sorted(out)

    yes, no = ids_for("Yes"), ids_for("No")
    assert yes and no, "answer tokens not in the vocabulary"

    out = np.zeros(len(texts))
    t0 = time.time()
    for i, (txt, sub) in enumerate(zip(texts, subjects)):
        body = WS.sub(" ", txt).strip()[:MAX_CHARS]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": "Subject: %s\n\nMessage:\n%s\n\nSensitive? "
                            "Answer Yes or No." % (sub or "(none)", body)}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt")
        enc = ({k: v.to(model.device) for k, v in enc.items()}
               if hasattr(enc, "keys") else {"input_ids": enc.to(model.device)})
        with torch.no_grad():
            logits = model(**enc).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        out[i] = float(torch.sigmoid(torch.logsumexp(lp[yes], 0)
                                     - torch.logsumexp(lp[no], 0)))
        if (i + 1) % 200 == 0:
            print("    scored %d/%d  (%.1f min)"
                  % (i + 1, len(texts), (time.time() - t0) / 60.0), flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def encoder_probs(X, y, tr2, va, te, tok):
    """Fine-tune on tr2, return probabilities for va and te."""
    def collate(batch):
        txt, l = zip(*batch)
        e = tok(list(txt), truncation=True, max_length=MAX_LEN,
                padding=True, return_tensors="pt")
        e["labels"] = torch.tensor(l)
        return e

    def probs(m, idx):
        dl = DataLoader(DS([X[i] for i in idx], y[idx]), batch_size=32,
                        collate_fn=collate)
        out = []
        with torch.no_grad():
            for b in dl:
                b = {k: v.to(DEV) for k, v in b.items()}
                b.pop("labels")
                out.extend(torch.softmax(m(**b).logits, -1)[:, 1].cpu().tolist())
        return np.array(out)

    torch.manual_seed(SEED)
    m = AutoModelForSequenceClassification.from_pretrained(
        ENC_MODEL, num_labels=2).to(DEV)
    m.train()
    dl = DataLoader(DS([X[i] for i in tr2], y[tr2]), batch_size=BATCH,
                    shuffle=True, collate_fn=collate)
    npos = max(int(y[tr2].sum()), 1)
    lossf = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, (len(tr2) - npos) / npos],
                            dtype=torch.float).to(DEV))
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    for _ in range(EPOCHS):
        for b in dl:
            b = {k: v.to(DEV) for k, v in b.items()}
            l = b.pop("labels")
            lossf(m(**b).logits, l).backward()
            opt.step()
            opt.zero_grad()
    m.eval()
    pv, pt = probs(m, va), probs(m, te)
    del m
    torch.cuda.empty_cache()
    return pv, pt


def metrics(y, pred, score, floor):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    d = dict(f1=round(float(f1_score(y, pred, zero_division=0)), 4),
             precision=round(float(precision_score(y, pred, zero_division=0)), 4),
             recall=round(float(recall_score(y, pred, zero_division=0)), 4),
             mcc=round(float(matthews_corrcoef(y, pred)), 4),
             tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
             trivial_floor=floor)
    if len(set(y)) > 1:
        d["roc_auc"] = round(float(roc_auc_score(y, score)), 4)
        d["pr_auc"] = round(float(average_precision_score(y, score)), 4)
    return d


def thread_bootstrap(y, pred, groups, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(groups)))
    by = {t: np.where(groups == t)[0] for t in uniq}
    f1s, mccs = [], []
    for _ in range(n):
        idx = np.concatenate([by[t] for t in rng.choice(uniq, len(uniq), True)])
        yy, pp = y[idx], pred[idx]
        if len(set(yy)) < 2:
            continue
        f1s.append(f1_score(yy, pp, zero_division=0))
        mccs.append(matthews_corrcoef(yy, pp))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4),
                   round(float(np.percentile(a, 97.5)), 4)]
    return {"f1_ci95": q(f1s), "mcc_ci95": q(mccs)}


def pick_threshold(y_va, s_va):
    grid = np.unique(np.round(np.linspace(0.01, 0.99, 99), 3))
    return float(max(grid, key=lambda t: f1_score(
        y_va, (s_va >= t).astype(int), zero_division=0)))


def run(df, lab, llm_all, OUT, SCORES):
    X = df.text.tolist()
    y = (df.label if lab == "strict" else df.label_either).values
    g = df.thread_key.values
    floor = round(float(2 * y.mean() / (1 + y.mean())), 4)
    tok = AutoTokenizer.from_pretrained(ENC_MODEL)
    Xv = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                         max_features=50000).fit_transform(X)

    names = ["llm", "encoder", "classical", "hybrid"]
    oof_s = {n: np.zeros(len(y)) for n in names}
    oof_p = {n: np.zeros(len(y), dtype=int) for n in names}
    weights = []

    for fold, (tr, te) in enumerate(
            GroupKFold(n_splits=FOLDS).split(X, y, g), 1):
        assert not (set(g[tr]) & set(g[te])), "thread leaked"
        rng = np.random.RandomState(SEED)
        th = np.array(sorted(set(g[tr])))
        rng.shuffle(th)
        va_th = set(th[:max(1, int(VAL_FRAC * len(th)))])
        va = np.array([i for i in tr if g[i] in va_th])
        tr2 = np.array([i for i in tr if g[i] not in va_th])

        t0 = time.time()
        lr = LogisticRegression(max_iter=2000,
                                class_weight="balanced").fit(Xv[tr2], y[tr2])
        c_va = lr.predict_proba(Xv[va])[:, 1]
        c_te = lr.predict_proba(Xv[te])[:, 1]
        e_va, e_te = encoder_probs(X, y, tr2, va, te, tok)
        l_va, l_te = llm_all[va], llm_all[te]

        # Calibrate every component on the validation threads. Without this
        # the LLM's mass sits at the extremes and dominates any linear
        # combination for reasons that have nothing to do with its accuracy.
        cal, sv, st = {}, {}, {}
        for nm, v, t in (("llm", l_va, l_te), ("encoder", e_va, e_te),
                         ("classical", c_va, c_te)):
            iso = IsotonicRegression(out_of_bounds="clip",
                                     y_min=0.0, y_max=1.0)
            iso.fit(v, y[va])
            cal[nm] = iso
            sv[nm], st[nm] = iso.predict(v), iso.predict(t)

        # Fuse on the validation threads only.
        Fva = np.column_stack([sv[n] for n in ("llm", "encoder", "classical")])
        Fte = np.column_stack([st[n] for n in ("llm", "encoder", "classical")])
        fuse = LogisticRegression(max_iter=1000,
                                  class_weight="balanced").fit(Fva, y[va])
        h_va = fuse.predict_proba(Fva)[:, 1]
        h_te = fuse.predict_proba(Fte)[:, 1]
        weights.append({"llm": round(float(fuse.coef_[0][0]), 4),
                        "encoder": round(float(fuse.coef_[0][1]), 4),
                        "classical": round(float(fuse.coef_[0][2]), 4)})

        for nm, v, t in (("llm", sv["llm"], st["llm"]),
                         ("encoder", sv["encoder"], st["encoder"]),
                         ("classical", sv["classical"], st["classical"]),
                         ("hybrid", h_va, h_te)):
            thr = pick_threshold(y[va], v)
            oof_s[nm][te] = t
            oof_p[nm][te] = (t >= thr).astype(int)

        print("  fold %d/%d done (%.1f min)  weights %s"
              % (fold, FOLDS, (time.time() - t0) / 60.0, weights[-1]),
              flush=True)

    for nm in names:
        r = metrics(y, oof_p[nm], oof_s[nm], floor)
        r.update(thread_bootstrap(y, oof_p[nm], g))
        OUT["%s_%s" % (nm, lab)] = r
        print("  %-10s %-7s F1 %.4f  MCC %.4f  ROC %.4f  CI [%.3f, %.3f]"
              % (nm, lab, r["f1"], r["mcc"], r.get("roc_auc", float("nan")),
                 r["f1_ci95"][0], r["f1_ci95"][1]), flush=True)
    OUT["fusion_weights_%s" % lab] = weights
    SCORES[lab] = {n: [round(float(v), 6) for v in oof_s[n]] for n in names}
    json.dump(OUT, io.open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    json.dump(SCORES, io.open(SCORES_JSON, "w", encoding="utf-8"), indent=2)


def main():
    print("=== hybrid LLM-based framework ===")
    print("sklearn %s | torch %s | %s"
          % (sklearn.__version__, torch.__version__,
             torch.cuda.get_device_name(0)))
    fetch_data()
    df = build()
    print("messages %d | threads %d | strict %d | broad %d"
          % (len(df), df.thread_key.nunique(), df.label.sum(),
             df.label_either.sum()), flush=True)

    print()
    print("scoring every message with the LLM once (zero-shot, fold- and "
          "label-independent)", flush=True)
    llm_all = llm_scores(df.text.tolist(), df.subject.tolist())
    print("  LLM scores: mean %.4f, at the extremes %.1f%%"
          % (llm_all.mean(),
             100.0 * np.mean((llm_all < 0.01) | (llm_all > 0.99))), flush=True)

    OUT = {"environment": {"python": sys.version.split()[0],
                           "scikit_learn": sklearn.__version__,
                           "torch": torch.__version__,
                           "gpu": torch.cuda.get_device_name(0)},
           "llm_model": LLM_MODEL, "encoder_model": ENC_MODEL,
           "dataset": {"messages": int(len(df)),
                       "threads": int(df.thread_key.nunique())},
           "note": "components calibrated on validation threads; fusion "
                   "weights and thresholds fitted there too; test folds "
                   "untouched"}
    SCORES = {"llm_raw": [round(float(v), 6) for v in llm_all]}

    for lab in ("strict", "broad"):
        print()
        print("=== %s labels ===" % lab, flush=True)
        run(df, lab, llm_all, OUT, SCORES)

    json.dump(OUT, io.open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    json.dump(SCORES, io.open(SCORES_JSON, "w", encoding="utf-8"), indent=2)
    print()
    print("written %s and %s" % (OUT_JSON, SCORES_JSON))


if __name__ == "__main__":
    main()
