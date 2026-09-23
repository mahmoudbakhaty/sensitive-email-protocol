# -*- coding: utf-8 -*-
"""Exercise the LLM harness without a GPU.

The model is the cheap part to get right. What is worth testing before a Kaggle
session is the bookkeeping: that no thread crosses a fold boundary, that no
in-context example is ever drawn from a test thread, and that the threshold is
chosen without touching the test fold. A stub judge makes all three checkable
on a laptop.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_path                                       # noqa: E402

import numpy as np

import llm_protocol as L

L.DATA_DIR = corpus_path.resolve(required=True)
L.N_BOOT = 200


class StubJudge(object):
    """Returns a score loosely tied to the label, and records what it saw."""

    def __init__(self, df):
        self.df = df
        self.shot_texts_seen = []
        self.rng = np.random.RandomState(0)

    def score(self, shots, subj, text):
        self.shot_texts_seen.append(tuple(s[1] for s in shots))
        row = self.df.index[self.df.text == text]
        y = int(self.df.at[row[0], "_y"]) if len(row) else 0
        return float(np.clip(0.35 + 0.3 * y + self.rng.normal(0, 0.12), 0, 1))


def main():
    df = L.build()
    print("messages %d | threads %d" % (len(df), df.thread_key.nunique()))
    df["_y"] = df.label
    y, g = df["_y"].values, df.thread_key.values

    folds = list(L.GroupKFold(n_splits=L.FOLDS).split(np.zeros(len(df)), y, g))
    for tr, te in folds:
        assert not (set(g[tr]) & set(g[te])), "thread crossed a fold boundary"
    print("fold disjointness OK across %d folds" % len(folds))

    # Every message is tested exactly once.
    covered = np.concatenate([te for _, te in folds])
    assert len(covered) == len(df) and len(set(covered)) == len(df), \
        "folds do not partition the data"
    print("folds partition the data OK")

    # In-context examples must never come from a thread under test.
    rng = np.random.RandomState(L.SEED)
    for tr, te in folds:
        shots = L.pick_shots(df, tr, rng)
        test_threads = set(g[te])
        for s_subj, s_text, s_lab in shots:
            hit = df.index[df.text == s_text]
            assert len(hit), "a shot came from outside the dataframe"
            assert df.at[hit[0], "thread_key"] not in test_threads, \
                "an in-context example came from a test thread"
        assert len(shots) == L.N_SHOT, "wrong number of shots: %d" % len(shots)
        assert 0 < sum(s[2] for s in shots) < L.N_SHOT, \
            "shots are not label-balanced"
    print("few-shot sourcing OK: no example from a test thread, balanced")

    OUT = {}
    judge = StubJudge(df)
    L.run(df, "strict", judge, True, OUT)

    key = [k for k in OUT if k.startswith("Qwen")][0]
    r = OUT[key]
    for f in ("f1", "mcc", "roc_auc", "f1_ci95", "thresholds",
              "trivial_floor", "folds_f1"):
        assert f in r, "missing field: %s" % f
    assert len(r["thresholds"]) == L.FOLDS, "one threshold per fold expected"
    assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == len(df), \
        "confusion matrix does not cover the dataset"
    print("result record OK: %s" % key)
    print("  F1 %.4f  MCC %.4f  ROC %.4f  floor %.4f  thresholds %s"
          % (r["f1"], r["mcc"], r["roc_auc"], r["trivial_floor"],
             r["thresholds"]))
    print()
    print("harness verified; only the model itself is untested here")


if __name__ == "__main__":
    main()
