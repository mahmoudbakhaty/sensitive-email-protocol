# -*- coding: utf-8 -*-
"""An instruction-tuned LLM evaluated under the same protocol as everything else.

This closes the gap a reviewer will find first. Ntwali et al. (arXiv:2506.22305)
show GPT-4o recovering most of the synthetic-to-real collapse that sank a
fine-tuned model on structured data, so "did you try an LLM?" is not an idle
question. Either answer is worth reporting: if the LLM also lands near the
fine-tuned models, the difficulty belongs to the task rather than to our choice
of model; if it pulls ahead, the benchmark discriminates and we say so.

The protocol is the point, so it is enforced here exactly as elsewhere:

  * thread-disjoint folds, from the same GroupKFold the other runs use;
  * few-shot examples drawn ONLY from the training threads of the current
    fold - an in-context example from the test thread is leakage, and it is the
    form of leakage that prompted systems invite;
  * the decision threshold chosen on a validation split of the training
    threads, never on the test fold;
  * the trivial all-positive floor reported beside every score.

Kaggle: GPU T4 x2 or P100, internet on. Runtime roughly 40-70 minutes.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

# Pin before anything imports sklearn, for the reason documented in
# FINDING_sklearn_groupkfold.md: 1.9.0 changed GroupKFold to stable sorting.
# Guarded so the file can be imported and its fold logic tested off-Kaggle;
# main() still refuses to produce numbers on an unpinned or CPU-only machine.
if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "scikit-learn==1.9.1"], check=True)
    # 4-bit weights: a 7B in fp16 leaves under 400 MiB free on a T4 and the
    # first long prompt exhausts it. Quantisation keeps the model size and
    # buys the headroom instead of trading down to a smaller model.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "bitsandbytes"], check=False)

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold


def require_environment():
    import torch
    assert sklearn.__version__ == "1.9.1", (
        "the scikit-learn pin did not take: got %s. Restart the session and "
        "run this cell first." % sklearn.__version__)
    assert torch.cuda.is_available(), "no GPU: enable the accelerator"
    return torch

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "enron_with_categories"
OUT_JSON = "RESULTS_LLM.json"
FOLDS, SEED, VAL_FRAC = 5, 42, 0.30
N_SHOT = 6
MAX_CHARS = 4000      # the message being judged, head-truncated
SHOT_CHARS = 900      # in-context examples, kept short so the prompt stays
                      # inside a T4's KV cache and the run inside its window
N_BOOT = 2000

SENS = {(4, 10), (3, 10), (2, 8), (3, 5), (3, 4), (1, 5), (1, 2)}
EMPTY = {(1, 7), (1, 8)}
WS = re.compile(r"\s+")
SUBJ = re.compile(r"^\s*((re|fw|fwd|aw)\s*:\s*)+", re.I)

def environment(torch):
    return {"python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__, "numpy": np.__version__,
            "torch": torch.__version__, "model": MODEL,
            "gpu": torch.cuda.get_device_name(0)}

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
    """Identical to final_run.py, so the row set is the same one."""
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
                     "subject": subj,
                     "thread_key": WS.sub(" ", SUBJ.sub("", subj).strip().lower())
                                   or ("m:" + h[:10]),
                     "text": body})
    return pd.DataFrame(rows)


def shorten(t, n=MAX_CHARS):
    t = WS.sub(" ", t).strip()
    return t[:n]


def user_msg(subj, text, n=MAX_CHARS):
    return "Subject: %s\n\nMessage:\n%s\n\nSensitive? Answer Yes or No." % (
        subj or "(none)", shorten(text, n))


def metrics(y, p, score):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    d = dict(f1=round(float(f1_score(y, p, zero_division=0)), 4),
             precision=round(float(precision_score(y, p, zero_division=0)), 4),
             recall=round(float(recall_score(y, p, zero_division=0)), 4),
             mcc=round(float(matthews_corrcoef(y, p)), 4),
             tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))
    if len(set(y)) > 1:
        d["roc_auc"] = round(float(roc_auc_score(y, score)), 4)
        d["pr_auc"] = round(float(average_precision_score(y, score)), 4)
    return d


def trivial_f1(y):
    b = float(np.mean(y))
    return round(2 * b / (1 + b), 4)


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
    return {"f1_ci95": q(f1s), "mcc_ci95": q(mccs), "resamples": len(f1s)}


class Judge(object):
    """Reads P(Yes) off the first generated token.

    A generated word would give a bare decision and no way to move the
    operating point. The probability mass on the two answer tokens gives a
    continuous score, so the LLM is thresholded on validation data exactly like
    every other model here.
    """

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = self._load(torch, AutoModelForCausalLM)
        self.model.eval()
        self.calls = 0
        self.yes = self._variants(["Yes", " Yes", "yes", " yes"])
        self.no = self._variants(["No", " No", "no", " no"])
        assert self.yes and self.no, "answer tokens not found in the vocabulary"

    def _load(self, torch, AutoModelForCausalLM):
        """4-bit if bitsandbytes is available, otherwise fp16 with headroom."""
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
            cfg = BitsAndBytesConfig(load_in_4bit=True,
                                     bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16,
                                     bnb_4bit_use_double_quant=True)
            m = AutoModelForCausalLM.from_pretrained(
                MODEL, quantization_config=cfg, device_map="auto")
            print("  loaded in 4-bit", flush=True)
            return m
        except Exception as exc:
            print("  4-bit unavailable (%s); falling back to fp16 with "
                  "reserved headroom" % type(exc).__name__, flush=True)
            n = torch.cuda.device_count()
            # Leave 3 GiB per device for activations and the KV cache.
            free = int(torch.cuda.get_device_properties(0).total_memory
                       / (1024 ** 3)) - 3
            mm = {i: "%dGiB" % max(free, 4) for i in range(n)}
            m = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=torch.float16, device_map="auto", max_memory=mm)
            print("  loaded in fp16, max_memory=%s" % mm, flush=True)
            return m

    def _variants(self, words):
        ids = set()
        for w in words:
            t = self.tok.encode(w, add_special_tokens=False)
            if t:
                ids.add(t[0])
        return sorted(ids)

    def score(self, shots, subj, text):
        msgs = [{"role": "system", "content": SYSTEM}]
        for s_subj, s_text, s_lab in shots:
            msgs.append({"role": "user",
                         "content": user_msg(s_subj, s_text, SHOT_CHARS)})
            msgs.append({"role": "assistant",
                         "content": "Yes" if s_lab else "No"})
        msgs.append({"role": "user", "content": user_msg(subj, text)})
        # apply_chat_template returns a bare tensor on some transformers
        # versions and a BatchEncoding on others. Passing the second straight
        # into the model raises "indices must be Tensor, not BatchEncoding",
        # so normalise to keyword arguments and let both shapes through.
        out = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt")
        dev = self.model.device
        if hasattr(out, "keys"):
            enc = {k: v.to(dev) for k, v in out.items()}
        else:
            enc = {"input_ids": out.to(dev)}
        torch = self.torch
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        y = torch.logsumexp(lp[self.yes], dim=0)
        n = torch.logsumexp(lp[self.no], dim=0)
        v = float(torch.sigmoid(y - n))
        self.calls = getattr(self, "calls", 0) + 1
        if self.calls % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return v


def pick_shots(df, idx, rng):
    """N_SHOT examples, half positive, drawn only from the given rows."""
    sub = df.iloc[idx]
    pos = sub.index[sub["_y"] == 1].to_numpy()
    neg = sub.index[sub["_y"] == 0].to_numpy()
    k = N_SHOT // 2
    take = list(rng.choice(pos, min(k, len(pos)), replace=False))
    take += list(rng.choice(neg, min(N_SHOT - len(take), len(neg)),
                            replace=False))
    rng.shuffle(take)
    return [(df.at[i, "subject"], df.at[i, "text"], int(df.at[i, "_y"]))
            for i in take]


def run(df, lab, judge, shots_on, OUT):
    y = df["_y"].values
    g = df.thread_key.values
    folds = list(GroupKFold(n_splits=FOLDS).split(np.zeros(len(df)), y, g))
    score = np.zeros(len(df), dtype=float)
    thresholds = []
    rng = np.random.RandomState(SEED)

    for fi, (tr, te) in enumerate(folds, 1):
        assert not (set(g[tr]) & set(g[te])), "thread leaked"
        shots = pick_shots(df, tr, rng) if shots_on else []
        for i in te:
            score[i] = judge.score(shots, df.at[i, "subject"], df.at[i, "text"])
        print("  fold %d/%d scored (%d messages)" % (fi, FOLDS, len(te)),
              flush=True)

    # Threshold on held-out training threads only, fold by fold.
    pred = np.zeros(len(df), dtype=int)
    for tr, te in folds:
        tr_threads = np.array(sorted(set(g[tr])))
        rs = np.random.RandomState(SEED)
        rs.shuffle(tr_threads)
        cut = max(1, int(len(tr_threads) * VAL_FRAC))
        val_threads = set(tr_threads[:cut])
        va = np.array([i for i in tr if g[i] in val_threads])
        if len(va) == 0 or len(set(y[va])) < 2:
            best = 0.5   # nothing to select on; fall back to the neutral point
        else:
            grid = np.unique(np.round(score[va], 3))
            best = max(grid, key=lambda t: f1_score(
                y[va], (score[va] >= t).astype(int), zero_division=0))
        thresholds.append(round(float(best), 4))
        pred[te] = (score[te] >= best).astype(int)

    key = "%s_%s_%s" % ("Qwen2.5-7B", "fewshot" if shots_on else "zeroshot", lab)
    r = metrics(y, pred, score)
    r.update(thread_bootstrap(y, pred, g))
    r.update(model=MODEL, shots=N_SHOT if shots_on else 0,
             thresholds=thresholds, trivial_floor=trivial_f1(y),
             folds_f1=[round(float(f1_score(y[te], pred[te], zero_division=0)), 4)
                       for _, te in folds])
    OUT[key] = r
    json.dump(OUT, open(OUT_JSON, "w"), indent=2)
    print("%-34s F1 %.4f  MCC %.4f  ROC %.4f  CI [%.3f, %.3f]  floor %.4f"
          % (key, r["f1"], r["mcc"], r.get("roc_auc", float("nan")),
             r["f1_ci95"][0], r["f1_ci95"][1], r["trivial_floor"]), flush=True)


def main():
    torch = require_environment()
    ENV = environment(torch)
    print("=== environment ===")
    for k, v in ENV.items():
        print("  %-14s %s" % (k, v))
    print()

    fetch_data()
    df = build()
    print("messages %d | threads %d | strict %d | broad %d"
          % (len(df), df.thread_key.nunique(), df.label.sum(),
             df.label_either.sum()))
    print()

    judge = Judge()
    OUT = {"environment": ENV,
           "note": "thread-disjoint folds; few-shot examples from training "
                   "threads only; thresholds from validation threads only"}

    for lab in ("strict", "broad"):
        df["_y"] = df.label if lab == "strict" else df.label_either
        for shots_on in (False, True):
            print("=== %s, %s ===" % (lab, "few-shot" if shots_on
                                      else "zero-shot"), flush=True)
            run(df, lab, judge, shots_on, OUT)
            print()

    print("written %s" % OUT_JSON)


if __name__ == "__main__":
    main()
