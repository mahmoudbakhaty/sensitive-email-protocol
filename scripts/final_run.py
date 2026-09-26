# -*- coding: utf-8 -*-
"""The single run every number in the paper comes from.

Why this exists. Building the paper incrementally produced three separate
result files, and reconciling them failed twice:

  * scikit-learn 1.9.0 changed GroupKFold to stable sorting (PR #28464), so
    1.6.1/1.8.0 partition the corpus one way (fingerprint 3556420f17e7e4fa)
    and 1.9.1 another (50b3daba1a99ae32). Confidence intervals computed under
    one partition do not belong beside point estimates from the other.
  * pinning scikit-learn alone still left LogReg differing by ~0.008 F1
    between sessions, most likely because the main run's pip line also moved
    numpy and with it the BLAS path through lbfgs.

Chasing each source of drift separately does not converge. The fix is
structural and is the paper's own advice applied to itself: compute every
reported figure in one session, in one environment, and record what that
environment was. Nothing here is carried over from an earlier run.

What it produces, all from the same folds:
  dataset statistics and Cohen's kappa
  leakage audit: duplicates, thread overlap, strongest single term
  classical baselines, both label sets, stratified AND thread-grouped,
    with per-fold scores, thread-level bootstrap CIs and saved predictions
  RoBERTa, both label sets, thread-grouped, validation-only thresholds,
    with per-fold scores and saved out-of-fold probabilities
  paired transformer-vs-baseline tests on identical folds
  the fold fingerprint and every library version

Roughly 70 minutes on a T4. Results are written after every fold.
"""
import subprocess
import sys

# Pin FIRST, before anything imports sklearn. A bare `pip install
# scikit-learn` is what caused the drift in the first place.
REQUIRED_SKLEARN = "1.9.1"
subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "scikit-learn==%s" % REQUIRED_SKLEARN, "transformers"],
               capture_output=True)

import sklearn
if sklearn.__version__ != REQUIRED_SKLEARN:
    raise SystemExit(
        "scikit-learn %s loaded but %s required. pip cannot swap a module "
        "this kernel already imported. Fix: Run -> Restart session, then run "
        "THIS CELL FIRST." % (sklearn.__version__, REQUIRED_SKLEARN))

import glob
import hashlib
import json
import os
import re
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd
import scipy
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             matthews_corrcoef, confusion_matrix,
                             roc_auc_score, average_precision_score)
from scipy.stats import wilcoxon

DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert DEV == "cuda", ("No GPU. Session options -> Accelerator -> GPU T4 x2, "
                       "then Restart session and run this cell first.")

MODEL = "roberta-base"
MAX_LEN, BATCH, LR, EPOCHS, FOLDS, SEED, VAL_FRAC = 384, 16, 2e-5, 6, 5, 42, 0.30
N_BOOT = 2000
DATA_DIR = "enron_with_categories"
OUT_JSON = "RESULTS_FINAL.json"
PRED_JSON = "PREDICTIONS_FINAL.json"

SENS = {(4, 10), (3, 10), (2, 8), (3, 5), (3, 4), (1, 5), (1, 2)}
EMPTY = {(1, 7), (1, 8)}
WS = re.compile(r"\s+")
SUBJ = re.compile(r"^\s*((re|fw|fwd|aw)\s*:\s*)+", re.I)

ENV = {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
       "numpy": np.__version__, "scipy": scipy.__version__,
       "torch": torch.__version__, "transformers": transformers.__version__,
       "gpu": (torch.cuda.get_device_name(0) if DEV == "cuda"
               else "none")}
print("=== environment (every figure below comes from exactly this) ===")
for k, v in ENV.items():
    print("  %-14s %s" % (k, v))
print()


def fetch_data():
    if os.path.isdir(DATA_DIR):
        print("data already present")
        return
    urllib.request.urlretrieve(
        "https://bailando.berkeley.edu/enron/enron_with_categories.tar.gz",
        "enron_with_categories.tar.gz")
    with tarfile.open("enron_with_categories.tar.gz") as tf:
        tf.extractall(".")


def build():
    rows, seen = [], set()
    both = one = neither = 0
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
        agreed = any((t, s) in SENS and f >= 2 for t, s, f in cats)
        either = any((t, s) in SENS for t, s, f in cats)
        both += agreed
        one += (either and not agreed)
        neither += (not either)
        # A tab or space only: \s* crosses the newline, so on a message
        # whose Subject header is empty it captures the next header
        # instead. 43 messages here, 34 of which then shared one false
        # thread key.
        m = re.search(r"^Subject:[ \t]*(.*)$", raw, re.M | re.I)
        subj = m.group(1).strip() if m else ""
        rows.append({"label": int(agreed), "label_either": int(either),
                     "thread_key": WS.sub(" ", SUBJ.sub("", subj).strip().lower())
                                   or ("m:" + h[:10]),
                     "text": body, "word_count": len(body.split())})
    return pd.DataFrame(rows), (both, one, neither)


def cohens_kappa(both, one, neither):
    n = both + one + neither
    a, d = both, neither
    b = c = one / 2.0
    po = (a + d) / n
    pa, pb = (a + b) / n, (a + c) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe), po, pe


def metrics(y, p, score=None):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    d = dict(f1=round(f1_score(y, p, zero_division=0), 4),
             precision=round(precision_score(y, p, zero_division=0), 4),
             recall=round(recall_score(y, p, zero_division=0), 4),
             mcc=round(matthews_corrcoef(y, p), 4),
             tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))
    if score is not None and len(set(y)) > 1:
        d["roc_auc"] = round(float(roc_auc_score(y, score)), 4)
        d["pr_auc"] = round(float(average_precision_score(y, score)), 4)
    return d


def trivial_f1(y):
    b = float(np.mean(y))
    return round(2 * b / (1 + b), 4)


def fingerprint(n, y, g):
    folds = list(GroupKFold(n_splits=FOLDS).split(np.zeros(n), y, g))
    return hashlib.md5("|".join(",".join(map(str, te)) for _, te in folds)
                       .encode()).hexdigest()[:16]


def thread_bootstrap(y, pred, groups, n=N_BOOT, seed=SEED):
    """Percentile CI over threads, not messages.

    Messages in a thread are not independent draws - that is the paper's
    argument for grouped splitting - so resampling messages would give an
    interval narrower than the data supports.
    """
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
    return {"f1_ci95": q(f1s), "mcc_ci95": q(mccs), "resamples": len(f1s)}


class DS(Dataset):
    """Holds raw text; tokenisation happens per batch so RAM stays flat."""

    def __init__(self, texts, labels):
        self.t, self.y = texts, labels

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return self.t[i], int(self.y[i])


def classical(X, y, g, lab, OUT, PRED):
    Xv = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                         max_features=50000).fit_transform(X)
    for split, cv in (("stratified",
                       StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)),
                      ("grouped", GroupKFold(n_splits=FOLDS))):
        splits = list(cv.split(X, y, g) if split == "grouped"
                      else cv.split(X, y))
        for mk, name, meth in (
                (lambda: LogisticRegression(max_iter=2000,
                                            class_weight="balanced"),
                 "LogReg", "predict_proba"),
                (lambda: LinearSVC(C=0.5, class_weight="balanced",
                                      random_state=42),
                 "LinearSVM", "decision_function")):
            pred = np.zeros(len(y), dtype=int)
            score = np.zeros(len(y), dtype=float)
            ff, fm = [], []
            for tr, te in splits:
                if split == "grouped":
                    assert not (set(g[tr]) & set(g[te])), "thread leaked"
                m = mk().fit(Xv[tr], y[tr])
                p = m.predict(Xv[te])
                s = getattr(m, meth)(Xv[te])
                if np.ndim(s) == 2:
                    s = s[:, 1]
                pred[te], score[te] = p, s
                ff.append(round(float(f1_score(y[te], p, zero_division=0)), 4))
                fm.append(round(float(matthews_corrcoef(y[te], p)), 4))
            k = "%s_%s_%s" % (name, split, lab)
            r = metrics(y, pred, score)
            r.update(folds_f1=ff, folds_mcc=fm,
                     fold_mean_f1=round(float(np.mean(ff)), 4),
                     fold_std_f1=round(float(np.std(ff)), 4),
                     fold_mean_mcc=round(float(np.mean(fm)), 4),
                     fold_std_mcc=round(float(np.std(fm)), 4))
            if split == "grouped":
                r.update(thread_bootstrap(y, pred, g))
            OUT[k] = r
            PRED[k] = {"pred": pred.tolist(),
                       "score": [round(float(v), 6) for v in score]}
            print("%-10s %-11s %-7s F1 %.4f  MCC %.4f  ROC %.4f%s"
                  % (name, split, lab, r["f1"], r["mcc"], r["roc_auc"],
                     "  CI [%.3f, %.3f]" % tuple(r["f1_ci95"])
                     if "f1_ci95" in r else ""), flush=True)
    json.dump(OUT, open(OUT_JSON, "w"), indent=2)


def transformer(X, y, g, lab, OUT, PRED):
    tok = AutoTokenizer.from_pretrained(MODEL)

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

    oof_bin, oof_prob = np.zeros(len(X), dtype=int), np.zeros(len(X))
    ff, fa, ft = [], [], []
    t0 = time.time()
    for fold, (tr, te) in enumerate(
            GroupKFold(n_splits=FOLDS).split(X, y, g), 1):
        assert not (set(g[tr]) & set(g[te])), "thread leaked"
        rng = np.random.RandomState(SEED)
        th = np.array(sorted(set(g[tr])))
        rng.shuffle(th)
        va_th = set(th[:max(1, int(VAL_FRAC * len(th)))])
        va = np.array([i for i in tr if g[i] in va_th])
        tr2 = np.array([i for i in tr if g[i] not in va_th])

        torch.manual_seed(SEED)
        m = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=2).to(DEV)
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

        pv = probs(m, va)
        t = float(max(np.arange(0.05, 0.96, 0.01),
                      key=lambda x: f1_score(y[va], (pv >= x).astype(int),
                                             zero_division=0)))
        pt = probs(m, te)
        oof_prob[te] = pt
        oof_bin[te] = (pt >= t).astype(int)
        ff.append(round(float(f1_score(y[te], oof_bin[te], zero_division=0)), 4))
        fa.append(round(float(roc_auc_score(y[te], pt)), 4)
                  if len(set(y[te])) > 1 else None)
        ft.append(round(t, 2))
        print("  [%s] fold %d  val_pos %3d  thr %.2f  F1 %.4f  ROC %.4f  "
              "(%.1f min)" % (lab, fold, int(y[va].sum()), t, ff[-1],
                              fa[-1] or float("nan"), (time.time() - t0) / 60),
              flush=True)
        OUT["transformer_%s_progress" % lab] = {"f1": ff, "auc": fa}
        json.dump(OUT, open(OUT_JSON, "w"), indent=2)
        del m
        torch.cuda.empty_cache()

    r = metrics(y, oof_bin, oof_prob)
    r.update(model=MODEL, folds_f1=ff, folds_auc=fa, thresholds=ft,
             fold_mean_f1=round(float(np.mean(ff)), 4),
             fold_std_f1=round(float(np.std(ff)), 4))
    r.update(thread_bootstrap(y, oof_bin, g))
    OUT["transformer_grouped_%s" % lab] = r
    PRED["transformer_grouped_%s" % lab] = {
        "pred": oof_bin.tolist(),
        "score": [round(float(v), 6) for v in oof_prob]}
    json.dump(OUT, open(OUT_JSON, "w"), indent=2)
    return r


def paired(OUT):
    """Transformer vs each baseline on identical folds, same session."""
    for lab in ("strict", "broad"):
        tf = OUT["transformer_grouped_%s" % lab]["folds_f1"]
        for base in ("LogReg", "LinearSVM"):
            bf = OUT["%s_grouped_%s" % (base, lab)]["folds_f1"]
            d = [a - b for a, b in zip(tf, bf)]
            try:
                p = round(float(wilcoxon(tf, bf).pvalue), 4)
            except ValueError:
                p = None
            OUT["paired_transformer_vs_%s_%s" % (base, lab)] = {
                "transformer_folds": tf, "baseline_folds": bf,
                "differences": [round(x, 4) for x in d],
                "wins": int(sum(1 for x in d if x > 0)), "n_folds": len(d),
                "mean_difference": round(float(np.mean(d)), 4),
                "wilcoxon_p": p, "min_attainable_p_at_n5": 0.0625}
            print("  transformer vs %-10s %-7s wins %d/5  mean %+.4f  p=%s"
                  % (base, lab, OUT["paired_transformer_vs_%s_%s"
                                    % (base, lab)]["wins"],
                     float(np.mean(d)), p), flush=True)


def main():
    fetch_data()
    df, agree = build()
    X = df.text.tolist()
    Y = {"strict": df.label.values, "broad": df.label_either.values}
    g = df.thread_key.values
    k, po, pe = cohens_kappa(*agree)

    OUT = {"environment": ENV,
           "fold_fingerprint": fingerprint(len(df), Y["strict"], g),
           "note": "every figure in this file comes from this one session"}
    PRED = {}
    OUT["dataset"] = {"messages": int(len(df)),
                      "sensitive_strict": int(Y["strict"].sum()),
                      "sensitive_broad": int(Y["broad"].sum()),
                      "threads": int(df.thread_key.nunique()),
                      "median_words": int(df.word_count.median())}
    OUT["annotation"] = {"both": agree[0], "one": agree[1], "neither": agree[2],
                         "observed_agreement": round(po, 4),
                         "chance_agreement": round(pe, 4),
                         "cohens_kappa": round(k, 4)}
    OUT["trivial_floor"] = {"strict": trivial_f1(Y["strict"]),
                            "broad": trivial_f1(Y["broad"])}
    print("messages %d | strict %d | broad %d | threads %d | kappa %.4f"
          % (len(df), Y["strict"].sum(), Y["broad"].sum(),
             df.thread_key.nunique(), k))
    print("fold fingerprint %s" % OUT["fold_fingerprint"])
    print("trivial floor: strict %.4f | broad %.4f"
          % (OUT["trivial_floor"]["strict"], OUT["trivial_floor"]["broad"]))
    print()

    norm = [WS.sub(" ", t.lower()).strip() for t in X]
    tc = pd.Series(g).value_counts()
    multi = tc[tc > 1]
    cvec = CountVectorizer(min_df=5, binary=True, max_features=20000)
    B = cvec.fit_transform(X).tocsc()
    vocab = np.array(cvec.get_feature_names_out())
    bw, bf = "", 0.0
    for j in range(B.shape[1]):
        col = np.asarray(B[:, j].todense()).ravel()
        if col.sum() < 10:
            continue
        f = f1_score(Y["strict"], col, zero_division=0)
        if f > bf:
            bw, bf = vocab[j], f
    OUT["leakage_audit"] = {
        "exact_duplicates": len(norm) - len(set(norm)),
        "threads_with_multiple_messages": int(len(multi)),
        "threads_with_mixed_labels": int(sum(
            1 for t in multi.index
            if df.loc[df.thread_key == t, "label"].nunique() > 1)),
        "vocabulary_terms": int(B.shape[1]),
        "best_single_term": str(bw),
        "best_single_term_f1": round(float(bf), 4)}
    a = OUT["leakage_audit"]
    print("audit: duplicates %d | multi-message threads %d | mixed %d"
          % (a["exact_duplicates"], a["threads_with_multiple_messages"],
             a["threads_with_mixed_labels"]))
    print("audit: strongest of %d terms '%s' F1 %.4f (trivial %.4f)"
          % (a["vocabulary_terms"], bw, bf, OUT["trivial_floor"]["strict"]))
    print()

    for lab in ("strict", "broad"):
        classical(X, Y[lab], g, lab, OUT, PRED)
    print()

    for lab in ("strict", "broad"):
        print("=== RoBERTa, %s labels ===" % lab, flush=True)
        transformer(X, Y[lab], g, lab, OUT, PRED)
    print()

    print("=== paired comparisons, identical folds ===")
    paired(OUT)

    json.dump(OUT, open(OUT_JSON, "w"), indent=2)
    json.dump(PRED, open(PRED_JSON, "w"))

    print()
    print("================= FINAL, ONE SESSION =================")
    for lab in ("strict", "broad"):
        print("--- %s labels (trivial floor %.4f) ---"
              % (lab.upper(), OUT["trivial_floor"][lab]))
        rows = []
        for key, nm in ((f"LinearSVM_grouped_{lab}", "LinearSVM"),
                        (f"LogReg_grouped_{lab}", "LogReg"),
                        (f"transformer_grouped_{lab}", MODEL)):
            v = OUT[key]
            rows.append({"model": nm, "F1": v["f1"], "P": v["precision"],
                         "R": v["recall"], "MCC": v["mcc"],
                         "ROC": v.get("roc_auc"), "PR": v.get("pr_auc"),
                         "F1_mean": v["fold_mean_f1"],
                         "F1_sd": v["fold_std_f1"],
                         "F1_CI": v.get("f1_ci95")})
        print(pd.DataFrame(rows).to_string(index=False))
        print("  grouping cost (stratified -> grouped, F1):")
        for base in ("LogReg", "LinearSVM"):
            s = OUT["%s_stratified_%s" % (base, lab)]["f1"]
            gr = OUT["%s_grouped_%s" % (base, lab)]["f1"]
            print("    %-10s %.4f -> %.4f  (%+.4f)" % (base, s, gr, gr - s))
        print()
    print("environment: sklearn %s, numpy %s, torch %s"
          % (ENV["scikit_learn"], ENV["numpy"], ENV["torch"]))
    print("fold fingerprint: %s" % OUT["fold_fingerprint"])
    print("saved -> %s and %s" % (OUT_JSON, PRED_JSON))
    print("======================================================")


main()
