# -*- coding: utf-8 -*-
"""Check the framework's bookkeeping without a GPU.

The parts that can go wrong here are not the models. They are whether any
validation thread reaches the test fold, whether the calibrators and the
fusion see only validation data, and whether the thresholds are chosen there
too. A stub encoder and a stub LLM make all of that checkable on a laptop.

The first Kaggle run of the LLM script died in a line this kind of test could
not reach, so the model call itself is covered separately by
test_judge_path.py. This covers everything around it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_path                                       # noqa: E402

import os
import sys
import types

import numpy as np

# Stand in for the GPU pieces before the module is imported.
fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(
    is_available=lambda: True, get_device_name=lambda i: "stub",
    empty_cache=lambda: None)
fake_torch.__version__ = "stub"


def _fail(*a, **k):
    raise AssertionError("the stub test must not reach real torch")


fake_torch.manual_seed = lambda s: None
fake_torch.no_grad = lambda: types.SimpleNamespace(
    __enter__=lambda s: None, __exit__=lambda s, *a: False)

import hybrid_framework as H  # noqa: E402

H.DATA_DIR = corpus_path.resolve(required=True)
H.N_BOOT = 200

SEEN = {"encoder_train": [], "encoder_eval": []}


def stub_encoder(X, y, tr2, va, te, tok):
    """Records what it was trained on and asked about, then returns a signal
    loosely tied to the label so the downstream arithmetic is exercised."""
    SEEN["encoder_train"].append(set(tr2))
    SEEN["encoder_eval"].append((set(va), set(te)))
    rng = np.random.RandomState(0)
    mk = lambda idx: np.clip(0.3 + 0.35 * y[idx] + rng.normal(0, 0.1, len(idx)),
                             0.001, 0.999)
    return mk(va), mk(te)


def main():
    H.encoder_probs = stub_encoder
    df = H.build()
    print("messages %d | threads %d" % (len(df), df.thread_key.nunique()))

    rng = np.random.RandomState(1)
    y_all = df.label.values
    # A saturated stub, like the real LLM: mass at the extremes.
    llm_all = np.where(rng.rand(len(df)) < 0.5 + 0.3 * y_all,
                       rng.uniform(0.97, 0.999, len(df)),
                       rng.uniform(0.001, 0.03, len(df)))
    print("stub LLM at the extremes: %.1f%%"
          % (100.0 * np.mean((llm_all < 0.01) | (llm_all > 0.99))))

    OUT, SCORES = {}, {}
    H.run(df, "strict", llm_all, OUT, SCORES)

    # run() used to write RESULTS_HYBRID.json unconditionally, so this stub -
    # whose encoder is handed the label and therefore scores F1 0.94 on a
    # benchmark where nothing honest passes 0.41 - produced a file that read
    # like a real result. Nothing in the paper came from it. Keep it that way.
    for f in (H.OUT_JSON, H.SCORES_JSON):
        assert not os.path.exists(f),             "the stub run wrote %s; a result file must come from main()" % f

    g = df.thread_key.values
    print()
    for i, (tr2, (va, te)) in enumerate(zip(SEEN["encoder_train"],
                                            SEEN["encoder_eval"]), 1):
        assert not (tr2 & va), "fold %d: validation rows used for training" % i
        assert not (tr2 & te), "fold %d: test rows used for training" % i
        assert not (va & te), "fold %d: validation and test overlap" % i
        tr_th = {g[j] for j in tr2}
        va_th = {g[j] for j in va}
        te_th = {g[j] for j in te}
        assert not (tr_th & te_th), "fold %d: a thread spans train and test" % i
        assert not (va_th & te_th), "fold %d: a thread spans val and test" % i
        assert not (tr_th & va_th), "fold %d: a thread spans train and val" % i
    print("no row and no thread crosses train, validation or test, %d folds"
          % len(SEEN["encoder_train"]))

    for k in ("llm_strict", "encoder_strict", "classical_strict",
              "hybrid_strict"):
        assert k in OUT, "missing result: %s" % k
        r = OUT[k]
        for f in ("f1", "mcc", "roc_auc", "f1_ci95", "trivial_floor"):
            assert f in r, "%s missing %s" % (k, f)
        assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == len(df), \
            "%s: confusion matrix does not cover the dataset" % k
    print("four result records, each covering every message")

    w = OUT["fusion_weights_strict"]
    assert len(w) == H.FOLDS, "one weight vector per fold expected"
    assert all(set(d) == {"llm", "encoder", "classical"} for d in w)
    print("fusion weights recorded per fold: %s" % w[0])

    assert set(SCORES["strict"]) == {"llm", "encoder", "classical", "hybrid"}
    assert all(len(v) == len(df) for v in SCORES["strict"].values())
    print("per-message scores kept for all four, %d each" % len(df))

    print()
    print("framework bookkeeping verified; only the two models are stubbed")


if __name__ == "__main__":
    sys.exit(main())
