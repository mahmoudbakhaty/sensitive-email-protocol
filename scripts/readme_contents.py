# -*- coding: utf-8 -*-
"""Regenerate the README's contents from its own headings.

The list was written by hand once and was already two sections out of date the
next time a section was added, which is how it will always go. It is generated
now, and any heading not placed in a group appears under "Also" rather than
silently vanishing - a missing entry is the failure this file exists to
prevent.

test_paths.py runs this and fails if the file on disk did not already match.
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

BEGIN, END = "<!-- contents -->", "<!-- /contents -->"

# The grouping is the argument: what the release is, whether you can run it,
# what the protocol says, what the system does, what the measurements found.
GROUPS = [
    ("Start here",
     ["Headline numbers", "Reproduce it", "What is here"]),
    ("Can you run it, and does it reproduce",
     ["The data is not redistributed, it is rebuilt",
      "Did you get the same corpus we did?",
      "Can you actually run this?",
      "Do these scripts reproduce these numbers?",
      "The pasteable notebook, and which build produced what",
      "Check the fingerprint before comparing anything",
      "Reference verification"]),
    ("The protocol",
     ["The four controls", "Known limitations",
      "The two control files, and why their numbers are higher"]),
    ("The system, and what it can promise",
     ["The filter as a system, and what it can promise",
      "The system blocks nothing, and that is the answer",
      "The design works; 5.5% is the task, not the policy",
      "What a model is worth when it is allowed to decline",
      "Does it decline what the annotators argued over?",
      "The GPU run, and what it has to hand over"]),
    ("What the measurements found",
     ["Why the block promise breaks, and the calibrator that fixes it",
      "What more labels buy, and which labels to buy"]),
]


def anchor(h):
    a = re.sub(r"[^a-z0-9 -]", "", h.lower())
    return "#" + a.replace(" ", "-")


def main():
    p = os.path.join(REPO, "README.md")
    s = io.open(p, encoding="utf-8").read()
    heads = [h for h in re.findall(r"^## (.+)$", s, re.M) if h != "Contents"]
    placed = {h for _g, hs in GROUPS for h in hs}
    extra = [h for h in heads if h not in placed]

    lines = ["## Contents", ""]
    for g, hs in GROUPS:
        present = [h for h in hs if h in heads]
        if not present:
            continue
        lines += ["**%s**" % g, ""]
        lines += ["- [%s](%s)" % (h, anchor(h)) for h in present]
        lines.append("")
    if extra:
        lines += ["**Also**", ""]
        lines += ["- [%s](%s)" % (h, anchor(h)) for h in extra]
        lines.append("")
    toc = "\n".join(lines)

    if BEGIN not in s or END not in s:
        raise SystemExit("markers %s / %s not found in README.md"
                         % (BEGIN, END))
    new = s[:s.index(BEGIN)] + BEGIN + "\n" + toc + \
        s[s.index(END):]
    io.open(p, "w", encoding="utf-8").write(new)

    listed = sum(len([h for h in hs if h in heads]) for _g, hs in GROUPS)
    print("contents regenerated: %d sections, %d grouped, %d under Also"
          % (len(heads), listed, len(extra)))
    for h in extra:
        print("  ungrouped: %s" % h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
