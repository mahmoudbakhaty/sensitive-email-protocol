# -*- coding: utf-8 -*-
"""Can thread reconstruction be improved, and does it change the leakage cost?

The benchmark groups messages by normalised subject line. The paper records
that this both over-merges and under-merges, and the audit showed the
over-merging concretely: 'ticket' holds 7 messages and 'energy issues' 15,
subjects generic enough to be reused by unrelated conversations. The headers
that would settle it are absent - 3 messages of 1,382 carry In-Reply-To and
none carries References - so the subject line is all there is to work with.

It is not all there is to work with in the message, though. A reply shares
correspondents with its parent and follows it closely in time. Adding those two
constraints splits a generic subject into the separate conversations that
actually used it.

The question is not whether the new grouping looks tidier. It is whether the
measured cost of thread leakage changes when the grouping gets better, because
the paper reports that cost and calls it a lower bound. Both groupings are run
through the identical pipeline.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_paths                                          # noqa: E402

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_path                                       # noqa: E402

import glob
import hashlib
import io
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import strengthen as S

# Resolved, not hardcoded: $ENRON_DIR, then the usual places, then the
# author's original path. scripts/verify_corpus.py puts it where this
# finds it. See corpus_path.py.
BASE = corpus_path.resolve()
OUT = io_paths.result_out("RESULTS_threading_v2.json")
SENS = {(4, 10), (3, 10), (2, 8), (3, 5), (3, 4), (1, 5), (1, 2)}
EMPTY = {(1, 7), (1, 8)}
WS = re.compile(r"\s+")
SUBJ = re.compile(r"^\s*((re|fw|fwd|aw)\s*:\s*)+", re.I)
ADDR = re.compile(r"[\w.\-+]+@[\w.\-]+")

SEEDS = list(range(42, 62))
WINDOW_DAYS = 30          # a reply arriving a month later is a new conversation
MIN_SHARE = 0.34          # at least a third of the correspondents in common


def header(raw, name):
    m = re.search(r"^%s:[ \t]*(.*)$" % re.escape(name), raw, re.M | re.I)
    return m.group(1).strip() if m else ""


def parse_date(s):
    s = re.sub(r"\s*\([A-Z]{2,5}\)\s*$", "", s).strip()
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%d %b %Y %H:%M:%S %z"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def build():
    rows, seen = [], set()
    for cf in sorted(glob.glob(os.path.join(BASE, "*", "*.cats"))):
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
        subj = header(raw, "Subject")
        who = set(ADDR.findall(header(raw, "From"))) | \
              set(ADDR.findall(header(raw, "To"))) | \
              set(ADDR.findall(header(raw, "Cc")))
        rows.append({
            "label": int(any((t, s) in SENS and f >= 2 for t, s, f in cats)),
            "label_either": int(any((t, s) in SENS for t, s, f in cats)),
            "subject_key": WS.sub(" ", SUBJ.sub("", subj).strip().lower())
                           or ("m:" + h[:10]),
            "when": parse_date(header(raw, "Date")),
            "who": who,
            "text": body})
    return pd.DataFrame(rows)


def refine(df):
    """Split each subject group into runs that share people and a time window.

    Single-link over a subject group: two messages join when they share
    correspondents and fall within the window. Nothing merges across different
    subjects, so this can only split - a strictly finer grouping than the
    published one, which is the direction that makes the leakage control
    stricter rather than weaker.
    """
    keys = [None] * len(df)
    for subj, idx in df.groupby("subject_key").groups.items():
        idx = list(idx)
        if len(idx) == 1:
            keys[idx[0]] = subj
            continue
        order = sorted(idx, key=lambda i: (df.at[i, "when"] or datetime.min))
        parent = {i: i for i in order}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a_pos, a in enumerate(order):
            for b in order[a_pos + 1:]:
                ta, tb = df.at[a, "when"], df.at[b, "when"]
                if ta and tb and abs(tb - ta) > timedelta(days=WINDOW_DAYS):
                    break
                wa, wb = df.at[a, "who"], df.at[b, "who"]
                if not wa or not wb:
                    continue
                share = len(wa & wb) / float(min(len(wa), len(wb)))
                if share >= MIN_SHARE:
                    parent[find(b)] = find(a)
        for i in order:
            keys[i] = "%s#%d" % (subj, order.index(find(i)))
    return keys


def evaluate(X, y, g, Xv):
    r2 = [S.run_cv(X, y, g, "stratified", seed=s, Xv=Xv)[0]["f1"] for s in SEEDS]
    r3 = [S.run_cv(X, y, g, "grouped", seed=s, Xv=Xv)[0]["f1"] for s in SEEDS]
    return (round(float(np.mean(r2)), 4), round(float(np.mean(r3)), 4),
            round(float(np.mean(r3)) - float(np.mean(r2)), 4),
            round(float(np.std(r3)), 4))


def main():
    df = build()
    df["refined_key"] = refine(df)
    # The paper quotes the largest group before and after refinement. It was
    # right, and it was not in this record - the only way to check it was to
    # rebuild the refinement. Recorded now.
    import collections as _c
    largest = {"subject": max(_c.Counter(df.subject_key).values()),
               "refined": max(_c.Counter(df.refined_key).values())}
    print("messages                    %d" % len(df))
    print("largest group  subject %d -> refined %d"
          % (largest["subject"], largest["refined"]))
    print("subject-line threads        %d" % df.subject_key.nunique())
    print("refined threads             %d" % df.refined_key.nunique())
    split = df.groupby("subject_key").refined_key.nunique()
    print("subject groups that split   %d" % int((split > 1).sum()))
    print("largest subject group       %d messages" %
          df.subject_key.value_counts().iloc[0])
    print("largest refined group       %d messages" %
          df.refined_key.value_counts().iloc[0])
    print(flush=True)

    X = df.text.tolist()
    Xv = S.vectorise(X)
    out = {
        "largest_group": largest,"environment": S.ENV, "seeds": len(SEEDS),
           "window_days": WINDOW_DAYS, "min_share": MIN_SHARE,
           "subject_threads": int(df.subject_key.nunique()),
           "refined_threads": int(df.refined_key.nunique()),
           "subject_groups_split": int((split > 1).sum())}
    for lab in ("strict", "broad"):
        y = (df.label if lab == "strict" else df.label_either).values
        rec = {}
        for name, col in (("subject", "subject_key"),
                          ("refined", "refined_key")):
            r2, r3, cost, sd = evaluate(X, y, df[col].values, Xv)
            rec[name] = {"ungrouped": r2, "grouped": r3,
                         "leakage_cost": cost, "grouped_sd": sd}
            print("  %-7s %-8s ungrouped %.4f  grouped %.4f  cost %+.4f"
                  % (lab, name, r2, r3, cost), flush=True)
        rec["cost_change"] = round(rec["refined"]["leakage_cost"] -
                                   rec["subject"]["leakage_cost"], 4)
        print("  %-7s change in measured leakage cost: %+.4f"
              % (lab, rec["cost_change"]))
        print(flush=True)
        out[lab] = rec

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=2)
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
