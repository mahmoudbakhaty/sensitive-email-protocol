# -*- coding: utf-8 -*-
"""RoBERTa over several seeded partitions, because one partition is not a result.

The paper tells its readers not to report a single deterministic partition, and
then reports one for the encoder. That was defensible until the benchmark bug
was fixed: repairing a thread key that affected 43 messages moved the encoder's
strict-label F1 by 0.051, against a fold standard deviation of 0.077. A number
that mobile should not be quoted alone.

This runs the identical training configuration over several seeded
StratifiedGroupKFold partitions and reports the mean, the standard deviation
and the range, exactly as the classical results already are. Nothing else about
the pipeline changes: same model, same hyperparameters, same validation-only
threshold selection, same thread-disjoint constraint.

Kaggle: GPU, internet on. Roughly 40 minutes per seed per label set.
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

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

assert sklearn.__version__ == "1.9.1", sklearn.__version__
assert torch.cuda.is_available(), "no GPU: enable the accelerator"
DEV = "cuda"

MODEL = "roberta-base"
DATA_DIR = "enron_with_categories"
OUT_JSON = "RESULTS_ROBERTA_SEEDS.json"
MAX_LEN, BATCH, LR, EPOCHS, FOLDS, VAL_FRAC = 384, 16, 2e-5, 6, 5, 0.30
SEEDS = [42, 43, 44, 45, 46]

SENS = {(4, 10), (3, 10), (2, 8), (3, 5), (3, 4), (1, 5), (1, 2)}
EMPTY = {(1, 7), (1, 8)}
WS = re.compile(r"\s+")
SUBJ = re.compile(r"^\s*((re|fw|fwd|aw)\s*:\s*)+", re.I)


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
        # A tab or space only: \s* crosses the newline, so on a message
        # whose Subject header is empty it captures the next header
        # instead. 43 messages here, 34 of which then shared one false
        # thread key.
        m = re.search(r"^Subject:[ \t]*(.*)$", raw, re.M | re.I)
        subj = m.group(1).strip() if m else ""
        rows.append({"label": int(any((t, s) in SENS and f >= 2
                                      for t, s, f in cats)),
                     "label_either": int(any((t, s) in SENS
                                             for t, s, f in cats)),
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


def one_partition(X, y, g, seed, tok):
    """One seeded partition, five folds, pooled out-of-fold predictions."""
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
    cv = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, g):
        assert not (set(g[tr]) & set(g[te])), "thread leaked"
        rng = np.random.RandomState(seed)
        th = np.array(sorted(set(g[tr])))
        rng.shuffle(th)
        va_th = set(th[:max(1, int(VAL_FRAC * len(th)))])
        va = np.array([i for i in tr if g[i] in va_th])
        tr2 = np.array([i for i in tr if g[i] not in va_th])

        torch.manual_seed(seed)
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
        del m
        torch.cuda.empty_cache()

    return {"f1": round(float(f1_score(y, oof_bin, zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(y, oof_bin)), 4),
            "roc_auc": round(float(roc_auc_score(y, oof_prob)), 4)}


def main():
    fetch_data()
    df = build()
    X = df.text.tolist()
    g = df.thread_key.values
    print("messages %d | threads %d" % (len(df), df.thread_key.nunique()))
    print("seeds: %s" % SEEDS, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)

    out = {"environment": {"python": sys.version.split()[0],
                           "scikit_learn": sklearn.__version__,
                           "torch": torch.__version__,
                           "gpu": torch.cuda.get_device_name(0)},
           "model": MODEL, "seeds": SEEDS,
           "note": "StratifiedGroupKFold per seed; same configuration as the "
                   "single-partition run, only the partition varies"}

    for lab in ("strict", "broad"):
        y = (df.label if lab == "strict" else df.label_either).values
        runs = []
        for s in SEEDS:
            t0 = time.time()
            r = one_partition(X, y, g, s, tok)
            r["seed"] = s
            runs.append(r)
            print("  %-7s seed %d  F1 %.4f  MCC %.4f  ROC %.4f  (%.1f min)"
                  % (lab, s, r["f1"], r["mcc"], r["roc_auc"],
                     (time.time() - t0) / 60.0), flush=True)
            json.dump(out, io.open(OUT_JSON, "w", encoding="utf-8"), indent=2)
        f = np.array([r["f1"] for r in runs])
        m = np.array([r["mcc"] for r in runs])
        a = np.array([r["roc_auc"] for r in runs])
        out[lab] = {"runs": runs,
                    "f1_mean": round(float(f.mean()), 4),
                    "f1_sd": round(float(f.std()), 4),
                    "f1_range": [round(float(f.min()), 4),
                                 round(float(f.max()), 4)],
                    "mcc_mean": round(float(m.mean()), 4),
                    "roc_auc_mean": round(float(a.mean()), 4)}
        print("  %-7s MEAN F1 %.4f +/- %.4f  range [%.4f, %.4f]  MCC %.4f  "
              "ROC %.4f" % (lab, f.mean(), f.std(), f.min(), f.max(),
                            m.mean(), a.mean()), flush=True)
        json.dump(out, io.open(OUT_JSON, "w", encoding="utf-8"), indent=2)

    json.dump(out, io.open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print("written %s" % OUT_JSON)


if __name__ == "__main__":
    main()
