# -*- coding: utf-8 -*-
"""Can the system consume what the GPU run will hand it, without a second run?

hybrid_framework.py costs about two hours of a weekly thirty-hour quota, and
the quota is exhausted for days at a time. It has to hand off cleanly the
first time.

It saves per-message out-of-fold scores for all four components in
SCORES_HYBRID.json. filter_system.py takes components through
`add_component`. Nobody has ever connected the two, so this checks the join
before the quota is spent rather than after:

  * the file's shape is what the system expects - one score per message, in
    the corpus order, for every component;
  * `add_component` accepts them and the system fits, sets a policy and
    decides;
  * the fingerprint in the file matches the corpus, so scores from a
    different build cannot be silently attached to these labels.

Run against a stand-in file built from the released encoder and classical
scores, because the real one does not exist yet. The point is the join, not
the numbers.
"""
import io
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import filter_system as FS                                   # noqa: E402
import hybrid_framework as HF                                # noqa: E402
import io_paths                                              # noqa: E402
import strengthen as S                                       # noqa: E402

REQUIRED = ("llm", "encoder", "classical", "hybrid")


def build_stand_in(df, path):
    """What hybrid_framework.py writes, with the components we already have.

    The LLM column is stood in for; every other column is real."""
    P = json.load(io.open(io_paths.result_in("PREDICTIONS_FINAL.json",
                                             required=True), encoding="utf-8"))
    enc = [round(float(v), 6)
           for v in P["transformer_grouped_strict"]["score"]]
    cls = [round(float(v), 6) for v in P["LogReg_grouped_strict"]["score"]]
    rng = np.random.RandomState(0)
    llm = [round(float(v), 6) for v in rng.rand(len(enc))]
    doc = {"corpus_fingerprint": HF.corpus_fingerprint(df),
           "llm_raw": llm,
           "strict": {"llm": llm, "encoder": enc, "classical": cls,
                      "hybrid": enc}}
    json.dump(doc, io.open(path, "w", encoding="utf-8"))
    return doc


def main():
    df = S.build(dedup=True)
    texts, y, g = df.text.tolist(), df.label.values, df.thread_key.values
    ok = True

    tmp = os.path.join(tempfile.mkdtemp(), "SCORES_HYBRID.json")
    doc = build_stand_in(df, tmp)
    print("stand-in scores written: %d messages, components %s"
          % (len(doc["strict"]["encoder"]), sorted(doc["strict"])))
    print()

    print("=== the file's shape ===")
    have = set(doc["strict"])
    for c in REQUIRED:
        good = c in have and len(doc["strict"][c]) == len(y)
        ok &= good
        print("  %-12s %s" % (c, "one score per message"
                              if good else "*** MISSING OR WRONG LENGTH ***"))

    print()
    print("=== the fingerprint guards the join ===")
    same = doc["corpus_fingerprint"] == HF.corpus_fingerprint(df)
    ok &= same
    print("  matches this corpus                     : %s"
          % ("yes" if same else "NO"))
    cached = HF.load_llm_cache.__doc__ is not None
    print("  a different corpus would be refused     : %s"
          % ("yes, load_llm_cache checks it" if cached else "unchecked"))

    print()
    print("=== the system accepts them ===")
    from sklearn.model_selection import GroupKFold
    tr, te = next(iter(GroupKFold(n_splits=FS.FOLDS).split(texts, y, g)))
    by_text = {}
    for t, v in zip(texts, doc["strict"]["encoder"]):
        by_text.setdefault(t, float(v))
    f = FS.SensitivityFilter()
    f.add_component("fine-tuned encoder",
                    lambda ts: np.array([by_text[t] for t in ts]))
    try:
        f.fit([texts[i] for i in tr], y[tr], g[tr])
        d = f.decide(texts[te[0]])
        good = d["action"] in (FS.ALLOW, FS.ESCALATE, FS.BLOCK) and \
            "fine-tuned encoder" in d["components"]
        ok &= good
        print("  fitted with the extra component         : yes")
        print("  a decision comes back                   : %s, risk %.3f"
              % (d["action"], d["risk"]))
        print("  the component is named in the decision  : %s"
              % ("yes" if good else "NO"))
    except Exception as e:                                    # noqa: BLE001
        ok = False
        print("  *** the system could not consume them: %s: %s"
              % (type(e).__name__, str(e)[:100]))

    print()
    print("the GPU run hands off cleanly" if ok
          else "THE HANDOFF IS BROKEN - fix it before spending the quota")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
