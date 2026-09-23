# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# KEPT AS IT RAN. DO NOT "FIX" THE SUBJECT REGEX BELOW.
#
# This module built the dataset for Section VIII (the cross-lingual run) and
# for Table VI (the library-version finding). It carries the Subject regex
# from before the correction described in the README:
#
#     re.search(r"^Subject:\s*(.*)$", ...)
#
# \s* crosses the newline, so on a message with an empty Subject header it
# captures the next header instead. Forty-three messages are affected and
# thirty-four of them share one false thread key, which is why this module
# yields 1069 thread keys where the corrected builder yields 1103.
#
# Every number in results/RESULTS_arabic_v2.json was produced over those 1069
# keys. Correcting the regex here would make the released code stop matching
# the released numbers. The Arabic section is internally consistent - its
# English and Arabic arms share this grouping, so the comparison between them
# holds - and its figures are not comparable with Tables III to V, which use
# the corrected builder. Re-running it needs a GPU.
# ---------------------------------------------------------------------------
"""Run 2. Same protocol as run 1, four changes, all of them answers to what
run 1 revealed:

  1. THRESHOLD-FREE METRICS. Run 1 tuned a threshold on ~36 validation
     positives and the chosen value swung 0.25-0.94 across folds; fold F1
     ranged 0.235-0.461 as a result. ROC-AUC and PR-AUC measure how well the
     model RANKS messages and never touch a threshold, so they are immune to
     that noise. Run 1 discarded the raw probabilities, which is why this
     needs a re-run rather than a recomputation.
  2. BIGGER VALIDATION SLICE, 18% -> 30% of training threads, so the
     threshold that IS reported rests on ~60 positives instead of ~36.
  3. BOTH LABEL DEFINITIONS. Strict = both annotators agreed (250 positives).
     Broad = either annotator marked it (422). Reported side by side, never
     swapped - the broad set is a second row in the table, not a replacement.
  4. TRIVIAL BASELINE printed beside every number. With an 18.1% positive
     rate, "call everything sensitive" scores F1 0.307. Any result must be
     read against that, and run 1 showed LinearSVM falls BELOW it.

SMOKE=1 runs a miniature version that exercises every path in minutes on CPU.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

SMOKE = os.environ.get("SMOKE") == "1"

if not SMOKE:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "transformers", "scikit-learn", "pandas"],
                   capture_output=True)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             matthews_corrcoef, confusion_matrix,
                             roc_auc_score, average_precision_score)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "roberta-base"
MAX_LEN = 128 if SMOKE else 384
BATCH = 16
LR = 2e-5
EPOCHS = 1 if SMOKE else 6
FOLDS = 2 if SMOKE else 5
SEED = 42
VAL_FRAC = 0.30
DATA_DIR = os.environ.get("DATA_DIR", "enron_with_categories")
OUT_JSON = "RESULTS_v2.json"

SENS = {(4, 10): "secrecy/confidentiality", (3, 10): "legal advice",
        (2, 8): "legal documents", (3, 5): "political influence",
        (3, 4): "company image - influencing",
        (1, 5): "employment arrangements", (1, 2): "purely personal"}
EMPTY = {(1, 7), (1, 8)}
WS = re.compile(r"\s+")
SUBJ = re.compile(r"^\s*((re|fw|fwd|aw)\s*:\s*)+", re.I)


def fetch_data():
    """Pure-Python download and extract - no reliance on wget or tar."""
    if os.path.isdir(DATA_DIR):
        print("data already present:", DATA_DIR)
        return
    import tarfile
    import urllib.request
    url = ("https://bailando.berkeley.edu/enron/"
           "enron_with_categories.tar.gz")
    print("downloading", url)
    urllib.request.urlretrieve(url, "enron_with_categories.tar.gz")
    with tarfile.open("enron_with_categories.tar.gz") as tf:
        tf.extractall(".")
    print("extracted to", DATA_DIR)


def build():
    """Returns the dataframe plus the raw agreement counts for kappa."""
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
        if agreed:
            both += 1
        elif either:
            one += 1
        else:
            neither += 1
        m = re.search(r"^Subject:\s*(.*)$", raw, re.M | re.I)
        subj = m.group(1).strip() if m else ""
        rows.append({"label": int(agreed), "label_either": int(either),
                     "thread_key": WS.sub(" ", SUBJ.sub("", subj).strip().lower())
                                   or ("m:" + h[:10]),
                     "text": body, "word_count": len(body.split())})
    return pd.DataFrame(rows), (both, one, neither)


def cohens_kappa(both, one, neither):
    """Kappa from the agreement counts.

    The .cats files record how many of the two annotators chose each category,
    so a message is: both-sensitive, exactly-one-sensitive, or neither. The
    single-annotator cases are the disagreements and split evenly between the
    two off-diagonal cells.
    """
    n = both + one + neither
    a, d = both, neither
    b = c = one / 2.0
    po = (a + d) / n
    pa = (a + b) / n
    pb = (a + c) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe), po, pe


def metrics(y, p, score=None):
    """Threshold metrics; ROC-AUC and PR-AUC too when raw scores are given."""
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
    """F1 of 'call everything sensitive' - the floor every number sits above."""
    base = float(np.mean(y))
    return round(2 * base / (1 + base), 4), round(base, 4)


class DS(Dataset):
    """Holds raw text; tokenisation happens per batch so RAM stays flat."""

    def __init__(self, texts, labels):
        self.t, self.y = texts, labels

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return self.t[i], int(self.y[i])


def main():
    fetch_data()
    df, agree_counts = build()
    if SMOKE:
        df = df.groupby("label", group_keys=False).head(60).reset_index(drop=True)
    X = df.text.tolist()
    Y = {"strict": df.label.values, "broad": df.label_either.values}
    g = df.thread_key.values

    OUT = {}
    k, po, pe = cohens_kappa(*agree_counts)
    tf1, brate = trivial_f1(Y["strict"])
    OUT["dataset"] = {"messages": int(len(df)),
                      "sensitive_strict": int(Y["strict"].sum()),
                      "sensitive_broad": int(Y["broad"].sum()),
                      "sensitive_pct": round(100 * brate, 1),
                      "threads": int(df.thread_key.nunique()),
                      "median_words": int(df.word_count.median())}
    OUT["trivial_all_positive"] = {"strict_f1": tf1,
                                   "broad_f1": trivial_f1(Y["broad"])[0]}
    OUT["annotation"] = {"both_sensitive": agree_counts[0],
                         "one_sensitive": agree_counts[1],
                         "neither": agree_counts[2],
                         "observed_agreement": round(po, 4),
                         "chance_agreement": round(pe, 4),
                         "cohens_kappa": round(k, 4)}
    print("messages %d | strict %d (%.1f%%) | broad %d | threads %d"
          % (len(df), Y["strict"].sum(), 100 * brate, Y["broad"].sum(),
             df.thread_key.nunique()))
    print("Cohen's kappa %.3f  (observed agreement %.3f)" % (k, po))
    print("trivial all-positive F1: strict %.3f | broad %.3f"
          % (tf1, OUT["trivial_all_positive"]["broad_f1"]))

    # ---------- leakage audit (on the strict labels) ----------
    y = Y["strict"]
    norm = [WS.sub(" ", t.lower()).strip() for t in X]
    dups = len(norm) - len(set(norm))
    tc = pd.Series(g).value_counts()
    multi = tc[tc > 1]
    mixed = sum(1 for th in multi.index
                if df.loc[df.thread_key == th, "label"].nunique() > 1)
    cvec = CountVectorizer(min_df=5, binary=True, max_features=20000)
    B = cvec.fit_transform(X).tocsc()
    vocab = np.array(cvec.get_feature_names_out())
    best_word, best_f1 = "", 0.0
    for j in range(B.shape[1]):
        col = np.asarray(B[:, j].todense()).ravel()
        if col.sum() < 10:
            continue
        f = f1_score(y, col, zero_division=0)
        if f > best_f1:
            best_word, best_f1 = vocab[j], f
    OUT["leakage_audit"] = {"exact_duplicates": int(dups),
                            "threads_with_multiple_messages": int(len(multi)),
                            "threads_with_mixed_labels": int(mixed),
                            "best_single_word": str(best_word),
                            "best_single_word_f1": round(float(best_f1), 4),
                            "trivial_f1_for_comparison": tf1}
    print("audit: duplicates %d | multi-message threads %d | mixed-label %d"
          % (dups, len(multi), mixed))
    print("audit: strongest single word '%s' F1 %.3f  (trivial %.3f)"
          % (best_word, best_f1, tf1))

    # ---------- classical baselines: both label sets, both split styles ----------
    Xv = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                         max_features=50000).fit_transform(X)
    for lab_name, yt in Y.items():
        for split_name, cv, groups in (
                ("stratified",
                 StratifiedKFold(FOLDS, shuffle=True, random_state=SEED), None),
                ("thread_grouped", GroupKFold(n_splits=FOLDS), g)):
            for mdl, name, meth in (
                    (LogisticRegression(max_iter=2000, class_weight="balanced"),
                     "LogReg", "predict_proba"),
                    (LinearSVC(C=0.5, class_weight="balanced"),
                     "LinearSVM", "decision_function")):
                p = cross_val_predict(mdl, Xv, yt, cv=cv, groups=groups)
                s = cross_val_predict(mdl, Xv, yt, cv=cv, groups=groups,
                                      method=meth)
                if s.ndim == 2:
                    s = s[:, 1]
                key = "%s_%s_%s" % (name, split_name, lab_name)
                OUT[key] = metrics(yt, p, s)
                print("%-10s %-15s %-7s F1 %.3f  ROC-AUC %.3f  PR-AUC %.3f"
                      % (name, split_name, lab_name, OUT[key]["f1"],
                         OUT[key]["roc_auc"], OUT[key]["pr_auc"]))
    json.dump(OUT, open(OUT_JSON, "w"), indent=2)

    # ---------- transformer ----------
    tok = AutoTokenizer.from_pretrained(MODEL)

    def collate(batch):
        txt, lab = zip(*batch)
        e = tok(list(txt), truncation=True, max_length=MAX_LEN,
                padding=True, return_tensors="pt")
        e["labels"] = torch.tensor(lab)
        return e

    def probs(model, idx, yt):
        dl = DataLoader(DS([X[i] for i in idx], yt[idx]), batch_size=32,
                        collate_fn=collate)
        out = []
        with torch.no_grad():
            for b in dl:
                b = {k2: v.to(DEV) for k2, v in b.items()}
                b.pop("labels")
                out.extend(torch.softmax(model(**b).logits, -1)[:, 1].cpu().tolist())
        return np.array(out)

    def run_transformer(yt, tag):
        """Thread-grouped CV. Keeps the RAW probabilities so AUC is possible."""
        oof_bin = np.zeros(len(df))
        oof_prob = np.zeros(len(df))
        fold_f1, fold_auc, fold_thr = [], [], []
        t0 = time.time()
        for fold, (tr, te) in enumerate(
                GroupKFold(n_splits=FOLDS).split(X, yt, g), 1):
            assert not (set(g[tr]) & set(g[te])), "thread leaked train/test"
            rng = np.random.RandomState(SEED)
            th = np.array(sorted(set(g[tr])))
            rng.shuffle(th)
            va_th = set(th[:max(1, int(VAL_FRAC * len(th)))])
            va = np.array([i for i in tr if g[i] in va_th])
            tr2 = np.array([i for i in tr if g[i] not in va_th])
            if len(va) == 0 or yt[tr2].sum() == 0:
                va, tr2 = tr[:max(1, len(tr) // 4)], tr[max(1, len(tr) // 4):]

            torch.manual_seed(SEED)
            m = AutoModelForSequenceClassification.from_pretrained(
                MODEL, num_labels=2).to(DEV)
            m.train()
            dl = DataLoader(DS([X[i] for i in tr2], yt[tr2]), batch_size=BATCH,
                            shuffle=True, collate_fn=collate)
            npos = max(int(yt[tr2].sum()), 1)
            lossf = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, (len(tr2) - npos) / npos],
                                    dtype=torch.float).to(DEV))
            opt = torch.optim.AdamW(m.parameters(), lr=LR)
            for _ in range(EPOCHS):
                for b in dl:
                    b = {k2: v.to(DEV) for k2, v in b.items()}
                    lab = b.pop("labels")
                    lossf(m(**b).logits, lab).backward()
                    opt.step()
                    opt.zero_grad()
            m.eval()

            pv = probs(m, va, yt)
            best_t = max(np.arange(0.05, 0.96, 0.01),
                         key=lambda t: f1_score(yt[va], (pv >= t).astype(int),
                                                zero_division=0))
            pt = probs(m, te, yt)
            oof_prob[te] = pt
            oof_bin[te] = (pt >= best_t).astype(int)
            f = f1_score(yt[te], (pt >= best_t).astype(int), zero_division=0)
            auc = (roc_auc_score(yt[te], pt) if len(set(yt[te])) > 1
                   else float("nan"))
            fold_f1.append(round(float(f), 4))
            fold_auc.append(round(float(auc), 4))
            fold_thr.append(round(float(best_t), 2))
            print("  [%s] fold %d  val_pos %d  thr %.2f  F1 %.3f  ROC-AUC %.3f"
                  "  (%.1f min)"
                  % (tag, fold, int(yt[va].sum()), best_t, f, auc,
                     (time.time() - t0) / 60), flush=True)
            OUT["transformer_%s_progress" % tag] = {"folds_f1": fold_f1,
                                                    "folds_auc": fold_auc}
            json.dump(OUT, open(OUT_JSON, "w"), indent=2)
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()

        r = metrics(yt, oof_bin.astype(int), oof_prob)
        r["fold_mean_f1"] = round(float(np.mean(fold_f1)), 4)
        r["fold_std_f1"] = round(float(np.std(fold_f1)), 4)
        r["fold_mean_auc"] = round(float(np.mean(fold_auc)), 4)
        r["fold_std_auc"] = round(float(np.std(fold_auc)), 4)
        r["folds_f1"] = fold_f1
        r["folds_auc"] = fold_auc
        r["thresholds"] = fold_thr
        r["model"] = MODEL
        OUT["transformer_%s" % tag] = r
        json.dump(OUT, open(OUT_JSON, "w"), indent=2)
        return r

    for tag in ("strict", "broad"):
        print("")
        print("=== transformer, %s labels (%d positives) ==="
              % (tag, int(Y[tag].sum())), flush=True)
        run_transformer(Y[tag], tag)

    # ---------- report ----------
    print("")
    print("================ FINAL RESULTS (run 2) ================")
    d = OUT["dataset"]
    print("dataset: %d messages, %d threads, median %d words"
          % (d["messages"], d["threads"], d["median_words"]))
    print("strict %d positives (%.1f%%) | broad %d positives"
          % (d["sensitive_strict"], d["sensitive_pct"], d["sensitive_broad"]))
    print("Cohen's kappa (2 annotators): %.3f" % OUT["annotation"]["cohens_kappa"])
    print("")
    for lab_name in ("strict", "broad"):
        triv = OUT["trivial_all_positive"]["%s_f1" % lab_name]
        print("--- %s labels (trivial all-positive F1 = %.3f) ---"
              % (lab_name.upper(), triv))
        tbl = []
        for key, nm in (
                ("LogReg_thread_grouped_%s" % lab_name, "LogReg (grouped)"),
                ("LinearSVM_thread_grouped_%s" % lab_name, "LinearSVM (grouped)"),
                ("LogReg_stratified_%s" % lab_name, "LogReg (ungrouped)"),
                ("LinearSVM_stratified_%s" % lab_name, "LinearSVM (ungrouped)")):
            if key in OUT:
                v = OUT[key]
                tbl.append({"model": nm, "F1": v["f1"], "MCC": v["mcc"],
                            "ROC_AUC": v.get("roc_auc"),
                            "PR_AUC": v.get("pr_auc")})
        tk = "transformer_%s" % lab_name
        if tk in OUT:
            v = OUT[tk]
            tbl.append({"model": "%s (grouped)" % MODEL, "F1": v["f1"],
                        "MCC": v["mcc"], "ROC_AUC": v.get("roc_auc"),
                        "PR_AUC": v.get("pr_auc")})
            print(pd.DataFrame(tbl).sort_values("ROC_AUC", ascending=False)
                  .to_string(index=False))
            print("  transformer folds F1  : %s" % v["folds_f1"])
            print("  transformer folds AUC : %s" % v["folds_auc"])
            print("  thresholds chosen     : %s" % v["thresholds"])
            print("  fold-mean F1  %.3f +- %.3f | fold-mean AUC %.3f +- %.3f"
                  % (v["fold_mean_f1"], v["fold_std_f1"],
                     v["fold_mean_auc"], v["fold_std_auc"]))
            print("  confusion: TP %d FP %d FN %d TN %d"
                  % (v["tp"], v["fp"], v["fn"], v["tn"]))
        print("")
    print("strongest single word: '%s' F1 %.3f  (trivial %.3f)"
          % (OUT["leakage_audit"]["best_single_word"],
             OUT["leakage_audit"]["best_single_word_f1"],
             OUT["trivial_all_positive"]["strict_f1"]))
    print("saved -> %s" % OUT_JSON)
    print("=======================================================")


if __name__ == "__main__":
    main()
