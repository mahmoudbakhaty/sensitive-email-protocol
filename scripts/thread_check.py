# -*- coding: utf-8 -*-
"""How good is subject-line threading on this corpus?

The paper reconstructs threads from normalised subject lines and records, as a
threat to validity, that this both over- and under-merges. That is an assertion
about our own method and it can be checked: RFC 2822 gives In-Reply-To and
References headers, and where a message carries one the parent is known.

The literature says the Enron corpus has very few messages with In-Reply-To,
which is why subject-line threading is used at all. If that holds here, the
limitation stands as stated and we can say how few. If enough messages carry
the header, the agreement between the two methods is measurable and the paper
should report a number instead of a caveat.
"""
import glob
import hashlib
import os
import re
from collections import defaultdict

BASE = (r"C:\Users\lenovo\AppData\Local\Temp\claude\C--Users-lenovo"
        r"\fccd8d1f-1d07-4097-95b6-2b0cf5cf3dd3\scratchpad"
        r"\enroncat\enron_with_categories")

SENS = {(4, 10), (3, 10), (2, 8), (3, 5), (3, 4), (1, 5), (1, 2)}
EMPTY = {(1, 7), (1, 8)}
WS = re.compile(r"\s+")
SUBJ = re.compile(r"^\s*((re|fw|fwd|aw)\s*:\s*)+", re.I)


def header(raw, name):
    m = re.search(r"^%s:\s*(.*)$" % re.escape(name), raw, re.M | re.I)
    return m.group(1).strip() if m else ""


def main():
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
        rows.append({
            "msg_id": header(raw, "Message-ID"),
            "in_reply_to": header(raw, "In-Reply-To"),
            "references": header(raw, "References"),
            "xto": header(raw, "X-To"),
            "subject_thread": WS.sub(" ", SUBJ.sub("", subj).strip().lower())
                              or ("m:" + h[:10]),
        })

    n = len(rows)
    with_irt = sum(1 for r in rows if r["in_reply_to"])
    with_ref = sum(1 for r in rows if r["references"])
    with_id = sum(1 for r in rows if r["msg_id"])
    print("messages                      %d" % n)
    print("carry a Message-ID            %d  (%.1f%%)" % (with_id, 100.0 * with_id / n))
    print("carry In-Reply-To             %d  (%.1f%%)" % (with_irt, 100.0 * with_irt / n))
    print("carry References              %d  (%.1f%%)" % (with_ref, 100.0 * with_ref / n))
    print()

    subj_threads = defaultdict(list)
    for r in rows:
        subj_threads[r["subject_thread"]].append(r)
    sizes = sorted((len(v) for v in subj_threads.values()), reverse=True)
    print("subject-line threads          %d" % len(subj_threads))
    print("largest threads               %s" % sizes[:8])
    print("messages in threads of 1      %d" % sum(1 for s in sizes if s == 1))

    # Over-merging is the risk that matters: a generic subject pulls unrelated
    # messages together, which makes the grouping control stricter than the
    # conversation structure warrants rather than looser.
    generic = [t for t, v in subj_threads.items()
               if len(v) > 3 and len(t.split()) <= 2]
    print()
    print("threads of >3 messages with a subject of <=2 words: %d" % len(generic))
    for t in sorted(generic, key=lambda x: -len(subj_threads[x]))[:8]:
        senders = len({r["xto"] for r in subj_threads[t]})
        print("   %-28s %d messages, %d distinct X-To" %
              ("'" + t[:26] + "'", len(subj_threads[t]), senders))

    if with_irt:
        by_id = {r["msg_id"]: r for r in rows if r["msg_id"]}
        checkable = agree = 0
        for r in rows:
            parent = by_id.get(r["in_reply_to"])
            if parent is None:
                continue
            checkable += 1
            agree += (parent["subject_thread"] == r["subject_thread"])
        print()
        print("reply pairs with both messages in the corpus: %d" % checkable)
        if checkable:
            print("subject threading agrees with the header on %d of them (%.1f%%)"
                  % (agree, 100.0 * agree / checkable))


if __name__ == "__main__":
    main()
