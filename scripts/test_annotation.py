# -*- coding: utf-8 -*-
"""Does the annotation package work before anyone spends an hour on it?

An instrument handed over untested is a way of wasting someone's afternoon.
Three things are checked, and the first is the one that matters:

  * THE SAMPLE IS BLIND. The instrument and the item file must not carry our
    label, the stratum, or anything else that would let an annotator infer it.
    Checked against the key rather than by reading the code.
  * The derived-label rule in the HTML matches the rule the scorer applies.
    They are written in two languages and could drift; if they do, the export
    says one thing and the score computes another.
  * The scorer runs on a completed file and its verdict can come back against
    the mapping - across several simulated annotators, with RESULTS_DIR
    redirected so a simulation never leaves a record that reads like a real
    one. A checker that can only say "fine" is not a checker - the
    failure paths are exercised with simulated annotators who agree, who
    over-label and who are blind to half the definition.

Nothing here annotates anything. It exercises the machinery.
"""
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_annotation_sample as M                           # noqa: E402

ANN = os.path.join(REPO, "annotation")


def simulate(key, mode, seed=0, noise=0.05):
    """A stand-in annotator answering the three questions the tool asks.

    THE SYSTEMATIC PART IS A FUNCTION OF THE MESSAGE, NOT A FRESH COIN. A first
    version drew randomly per judgement, so a repeated message got a different
    answer and an annotator with a consistently different standard looked
    wildly inconsistent - the scorer then reported "too noisy to conclude" and
    was right to. A real disagreement about where the line sits is stable when
    the same message comes round again; only noise is not. The repeat block
    exists to separate those two, so the simulation has to contain both.

    `noise` is applied per judgement on top, and is the only thing a repeat
    pair can disagree about."""
    rng = np.random.RandomState(seed)

    def stable(base, p):
        """Deterministic in the message: the same text always gets the same
        systematic judgement, however many times it is shown."""
        h = int(hashlib.sha256(("%d|%s" % (seed, base)).encode()).hexdigest(),
                16)
        return (h % 10000) / 10000.0 < p

    out = {}
    for aid, k in key.items():
        ours = k["our_label"]
        base = k.get("repeat_of") or aid
        if mode == "agrees":
            want = ours
        elif mode == "over_labels":
            # our positives are mostly wrong: the annotator finds no harm
            want = ours if (ours == 0 or stable(base, 0.25)) else 0
        elif mode == "misses":
            # the annotator finds sensitivity among our negatives
            want = 1 if (ours == 1 or stable(base, 0.35)) else 0
        else:
            raise ValueError(mode)
        if rng.rand() < noise:
            want = 1 - want
        if want:
            a = {"harm": "yes", "quotable": "no", "identifier": "no",
                 "derived": "SENSITIVE", "reason": "simulated"}
        else:
            a = {"harm": "no", "identifier": "no",
                 "derived": "not sensitive", "reason": "simulated"}
        out[aid] = a
    return out


def html_rule_matches_scorer():
    """The HTML derives the label in JavaScript; the scorer reads the result.

    Both encode Section III. Rather than trust that, pull the branches out of
    the template and check they are the same four rules."""
    t = M.TEMPLATE
    want = [
        ('identifier === "yes"', "out of scope"),
        ('harm === "no"', "not sensitive"),
        ('harm === "unsure"', "unsure"),
        ('quotable === "yes"', "not sensitive"),
        ('quotable === "no"', "SENSITIVE"),
    ]
    ok = True
    for cond, verdict in want:
        seg = t[t.find("function derive"):t.find("function render")]
        i = seg.find(cond)
        good = i >= 0 and verdict in seg[i:i + 220]
        ok &= good
        print("    %-26s -> %-14s %s"
              % (cond, verdict, "yes" if good else "*** NOT FOUND ***"))
    return ok


def main():
    ok = True
    print("=== the sample is blind ===")
    for f in ("sample.json", "key.json", "annotate.html"):
        if not os.path.isfile(os.path.join(ANN, f)):
            print("  %s missing - run make_annotation_sample.py first" % f)
            return 1
    key = json.load(io.open(os.path.join(ANN, "key.json"),
                            encoding="utf-8"))["key"]
    samp = io.open(os.path.join(ANN, "sample.json"), encoding="utf-8").read()
    html = io.open(os.path.join(ANN, "annotate.html"), encoding="utf-8").read()
    items = json.loads(samp)["items"]

    fields = set(items[0].keys())
    clean = fields == {"aid", "n", "text"}
    ok &= clean
    print("  item fields are exactly aid/n/text        : %s"
          % ("yes" if clean else "NO, got %s" % sorted(fields)))

    # the strong form: could any annotator recover the label from what is
    # shipped? Only if the key's aids map to something present in the files.
    leaked = [w for w in ("our_label", "stratum", "\"agree\":")
              if w in samp or w in html]
    ok &= not leaked
    print("  no label-bearing field in either file     : %s"
          % ("yes" if not leaked else "NO: %s" % leaked))

    order = [key[it["aid"]]["our_label"] for it in items]
    runs = sum(1 for a, b in zip(order, order[1:]) if a != b)
    shuffled = abs(runs - (len(order) - 1) / 2.0) < 3 * np.sqrt(len(order)) / 2
    ok &= shuffled
    print("  the hidden order is shuffled, not blocked : %s (%d alternations "
          "in %d)" % ("yes" if shuffled else "NO", runs, len(order) - 1))

    firsts = {k: v for k, v in key.items() if not v.get("repeat_of")}
    one_per_thread = (len({v["thread_key"] for v in firsts.values()})
                      == len(firsts))
    ok &= one_per_thread
    print("  at most one message per thread            : %s"
          % ("yes" if one_per_thread else "NO"))

    reps = {k: v for k, v in key.items() if v.get("repeat_of")}
    pos = {it["aid"]: it["n"] for it in items}
    gaps = [abs(pos[a] - pos[v["repeat_of"]]) for a, v in reps.items()
            if a in pos and v["repeat_of"] in pos]
    far = bool(gaps) and min(gaps) > len(items) // 5
    ok &= far
    print("  %d repeats, and none close to its original: %s (nearest %d "
          "apart)" % (len(reps), "yes" if far else "NO",
                      min(gaps) if gaps else -1))
    same_text = all(next(i["text"] for i in items if i["aid"] == a)
                    == next(i["text"] for i in items
                            if i["aid"] == v["repeat_of"])
                    for a, v in reps.items())
    ok &= same_text
    print("  each repeat is the same message            : %s"
          % ("yes" if same_text else "NO"))

    print()
    print("=== the tool's rule is the scorer's rule ===")
    ok &= html_rule_matches_scorer()

    print()
    print("=== the scorer runs, and can return against us ===")
    seed = json.load(io.open(os.path.join(ANN, "key.json"),
                             encoding="utf-8"))["seed"]
    # What each simulated annotator SHOULD produce. "agrees" must not be
    # reported as a finding: a clean annotator disagreeing only at their own
    # noise rate is the case where the honest answer is "nothing shown".
    # ACROSS SIMULATION SEEDS, NOT ONE. An audit found the scorer aborting
    # with "too noisy to conclude" on roughly three seeds in five, because the
    # abort threshold tripped at the exact self-disagreement count a
    # 90%-consistent annotator produces. Testing one seed hid it completely:
    # seed 0 happened to give zero self-disagreements. A verdict that depends
    # on which way the annotator's noise fell is not a verdict.
    expect = {"agrees": "clears the annotator's own error rate",
              "over_labels": "OVER-LABELS",
              "misses": "BLIND"}
    SIM_SEEDS = [0, 1, 2, 3, 5, 8]
    for mode, want in expect.items():
        verdicts = []
        for sim in SIM_SEEDS:
            tmp = os.path.join(tempfile.mkdtemp(), "annotations.json")
            json.dump({"seed": seed,
                       "annotations": simulate(key, mode, seed=sim)},
                      io.open(tmp, "w", encoding="utf-8"))
        # RESULTS_DIR is redirected into the temp dir. Without it the scorer
        # writes RESULTS_SECTION_III_ANNOTATION.json where every real result
        # file lives, asserting a finding about the thesis - "THE MAPPING IS
        # BLIND beyond noise" - from a simulated annotator, with nothing in the
        # file to say it was simulated. One was sitting there until this ran.
            env = dict(os.environ, RESULTS_DIR=os.path.dirname(tmp))
            r = subprocess.run([sys.executable,
                                os.path.join(HERE, "score_annotation.py"),
                                tmp], capture_output=True, text=True, env=env)
            verdict, take = "", False
            for line in r.stdout.splitlines():
                if line.startswith("=== verdict"):
                    take = True
                    continue
                if take and line.strip():
                    verdict = line.strip()
                    break
            verdicts.append((sim, r.returncode, verdict))
        hits = [v for v in verdicts
                if v[1] == 0 and want.lower() in v[2].lower()]
        good = len(hits) == len(SIM_SEEDS)
        ok &= good
        print("  %-12s %d/%d seeds -> %s"
              % (mode, len(hits), len(SIM_SEEDS),
                 (verdicts[0][2] or "NO VERDICT")[:70]))
        if not good:
            print("       expected every seed to contain %r" % want)
            for sim, rc, v in verdicts:
                if not (rc == 0 and want.lower() in v.lower()):
                    print("       seed %d: %s"
                          % (sim, (v or "exit %d" % rc)[:88]))

    print()
    print("the annotation package is ready to hand over" if ok
          else "THE ANNOTATION PACKAGE IS BROKEN - do not hand it over")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
