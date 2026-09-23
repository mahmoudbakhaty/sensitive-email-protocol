# -*- coding: utf-8 -*-
"""The LLM cache must reload, and must refuse a corpus it does not belong to.

A cache that silently accepts the wrong scores is worse than no cache: it
would attach one corpus's LLM judgements to another's labels and report the
result as a measurement.
"""
import io
import json
import os
import sys
import types

import numpy as np

fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(
    is_available=lambda: True, get_device_name=lambda i: "stub",
    empty_cache=lambda: None)
fake_torch.__version__ = "stub"
fake_torch.manual_seed = lambda s: None
fake_torch.no_grad = lambda: types.SimpleNamespace(
    __enter__=lambda s: None, __exit__=lambda s, *a: False)

import hybrid_framework as H  # noqa: E402

H.DATA_DIR = (r"C:\Users\lenovo\AppData\Local\Temp\claude\C--Users-lenovo"
              r"\fccd8d1f-1d07-4097-95b6-2b0cf5cf3dd3\scratchpad"
              r"\enroncat\enron_with_categories")
H.SCORES_JSON = "TEST_llm_cache.json"

df = H.build()
fp = H.corpus_fingerprint(df)
rng = np.random.RandomState(7)
scores = rng.rand(len(df))

try:
    assert H.load_llm_cache(fp, len(df)) is None, "no file -> no cache"

    json.dump({"llm_raw": [round(float(v), 6) for v in scores],
               "corpus_fingerprint": fp},
              io.open(H.SCORES_JSON, "w", encoding="utf-8"))

    got = H.load_llm_cache(fp, len(df))
    assert got is not None, "a matching cache must load"
    assert len(got) == len(df)
    assert np.allclose(got, np.round(scores, 6)), "scores must survive"
    print("reload: %d scores, identical to 6 dp" % len(got))

    assert H.load_llm_cache("deadbeefdeadbeef", len(df)) is None, \
        "a different corpus must be refused"
    print("a different fingerprint is refused")

    assert H.load_llm_cache(fp, len(df) + 1) is None, \
        "a different message count must be refused"
    print("a different message count is refused")

    json.dump({"llm_raw": [0.1, 0.2], "corpus_fingerprint": fp},
              io.open(H.SCORES_JSON, "w", encoding="utf-8"))
    assert H.load_llm_cache(fp, len(df)) is None, \
        "a truncated cache must be refused"
    print("a truncated cache is refused")

    io.open(H.SCORES_JSON, "w", encoding="utf-8").write("{not json")
    assert H.load_llm_cache(fp, len(df)) is None, \
        "a corrupt cache must be refused, not raised"
    print("a corrupt cache is refused without raising")

    # the real one must still be absent: this test drives helpers, not a run
    assert not os.path.exists(H.OUT_JSON), "no result file from a stub test"
    print()
    print("LLM cache verified: it reloads, and refuses everything else")
finally:
    if os.path.exists(H.SCORES_JSON):
        os.remove(H.SCORES_JSON)
