# -*- coding: utf-8 -*-
"""The pasteable notebook must be the script it claims to be.

`KAGGLE_FINAL.txt` is what the README tells a reader to paste into Kaggle, and
the README describes it as `= scripts/final_run.py`. It had drifted: the
Subject-header correction - the defect that moved 43 messages, merged 34 of
them into one false thread and changed the benchmark from 1069 threads to
1103 - was applied to the script and not to the notebook. Anyone following the
reproduction instructions would have rebuilt the buggy benchmark and got
numbers that do not match the paper.

A stale copy of a file is invisible until someone diffs it, so this diffs it.

`KAGGLE_ARABIC_V2.txt` is deliberately NOT held equal to `arabic_v2.py`: it is
the record of the run that produced `results/RESULTS_arabic_v2.json`, over the
pre-correction 1069 thread keys. It is checked for its provenance header
instead, so nobody quietly "fixes" it and breaks the tie between the released
code and the released numbers.
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Spelled with the escape, not a literal tab. A tab character in source is one
# editor setting away from becoming spaces, and `[ ]*` would quietly bring back
# a variant of the bug this line exists to fix.
FIXED_REGEX = r'r"^Subject:[ \t]*(.*)$"'
BUGGY_REGEX = r'r"^Subject:\s*(.*)$"'

# Scripts that must carry the corrected Subject regex, because the results
# they produced were computed with it.
MUST_BE_FIXED = ["final_run.py", "strengthen.py", "llm_protocol.py",
                 "roberta_seeds.py", "hybrid_framework.py"]
# Kept as they ran, with the pre-correction regex, and labelled as such.
MUST_BE_LABELLED = [("colab_v2.py", HERE),
                    ("KAGGLE_ARABIC_V2.txt", REPO)]


def read(p):
    return io.open(p, encoding="utf-8").read().replace("\r", "")


def check_notebook_matches_script():
    a = read(os.path.join(REPO, "KAGGLE_FINAL.txt"))
    b = read(os.path.join(HERE, "final_run.py"))
    ok = a == b
    print("  KAGGLE_FINAL.txt == scripts/final_run.py : %s"
          % ("yes" if ok else "NO - the notebook has drifted from the script"))
    if not ok:
        import difflib
        for line in list(difflib.unified_diff(
                a.splitlines(), b.splitlines(),
                "KAGGLE_FINAL.txt", "scripts/final_run.py", n=1,
                lineterm=""))[:20]:
            print("     " + line[:110])
    return ok


def check_subject_regex():
    ok = True
    for f in MUST_BE_FIXED:
        p = os.path.join(HERE, f)
        if not os.path.isfile(p):
            print("  %-24s MISSING" % f)
            ok = False
            continue
        s = read(p)
        good = FIXED_REGEX in s and BUGGY_REGEX not in s
        ok &= good
        print("  %-24s %s" % (f, "corrected Subject regex" if good else
                              "*** CARRIES THE PRE-CORRECTION REGEX ***"))
    return ok


def check_kept_as_run_are_labelled():
    ok = True
    for f, d in MUST_BE_LABELLED:
        p = os.path.join(d, f)
        if not os.path.isfile(p):
            print("  %-24s MISSING - three released scripts import it" % f)
            ok = False
            continue
        s = read(p)
        good = "KEPT AS IT RAN" in s and "1069" in s
        ok &= good
        print("  %-24s %s" % (f, "labelled as a pre-correction record"
                              if good else "*** NOT LABELLED ***"))
    return ok


def main():
    ok = True
    print("=== the pasteable notebook is the script ===")
    ok &= check_notebook_matches_script()
    print()
    print("=== scripts behind the published tables carry the fix ===")
    ok &= check_subject_regex()
    print()
    print("=== files kept as they ran say so ===")
    ok &= check_kept_as_run_are_labelled()
    print()
    print("notebooks and scripts agree" if ok else
          "NOTEBOOKS AND SCRIPTS DISAGREE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
