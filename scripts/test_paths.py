# -*- coding: utf-8 -*-
"""No released script may name one machine.

Twenty absolute paths into the author's home directory were spread across
seventeen released scripts before this was written. Six pointed at the corpus,
fourteen at result files, and none of them worked for anyone who cloned the
repository - which made the release's central promise, that the benchmark is
rebuilt rather than redistributed, untrue for every reader.

The kind of defect that arrives one line at a time needs a check that runs, not
a note asking people to be careful. This fails if any script under scripts/
carries a user-specific absolute path, and it checks that the two resolvers
prefer the repository's own copies over anything on the machine.
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import corpus_path                                            # noqa: E402
import io_paths                                               # noqa: E402

# A drive-letter or UNC path anywhere in a released script. corpus_path.py is
# the one exception: it records where the published runs were produced, as an
# explicitly last-resort fallback.
ABSOLUTE = re.compile(r'r?["\'][A-Za-z]:[\\/]|r?["\']\\\\\\\\')
EXEMPT = {"corpus_path.py"}


def check_no_machine_paths():
    bad = []
    for f in sorted(os.listdir(HERE)):
        if not f.endswith(".py") or f in EXEMPT:
            continue
        s = io.open(os.path.join(HERE, f), encoding="utf-8").read()
        for i, line in enumerate(s.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if ABSOLUTE.search(line):
                bad.append("%s:%d  %s" % (f, i, line.strip()[:78]))
    if bad:
        print("scripts naming an absolute path:")
        for b in bad:
            print("   " + b)
    else:
        print("no released script names an absolute path"
              " (corpus_path.py exempt, by design)")
    return not bad


def check_results_resolve_to_repo():
    """A reader's copy must win over anything in a home directory."""
    ok = True
    want = os.path.join(REPO, "results")
    for n in ("RESULTS_FINAL.json", "RESULTS_LLM.json",
              "RESULTS_ladder_v2.json", "PREDICTIONS_FINAL.json"):
        got = io_paths.result_in(n)
        good = os.path.dirname(got).lower().startswith(want.lower())
        ok &= good
        print("  %-26s -> %s" % (n, "the repository's copy" if good else
                                 "NOT the repository's copy: " + got))
    return ok


def check_corpus_resolution_order():
    """$ENRON_DIR must come first, and the author's path must come last."""
    old = os.environ.get("ENRON_DIR")
    try:
        os.environ["ENRON_DIR"] = os.path.join(REPO, "no_such_corpus")
        c = corpus_path.candidates()
        first_ok = c[0].endswith("no_such_corpus")
        last_ok = c[-1] == corpus_path.ORIGINAL
        print("  $ENRON_DIR is consulted first        : %s" % first_ok)
        print("  the author's path is consulted last  : %s" % last_ok)
        return first_ok and last_ok
    finally:
        if old is None:
            os.environ.pop("ENRON_DIR", None)
        else:
            os.environ["ENRON_DIR"] = old


def check_no_duplicate_files():
    """The same file in two places drifts. Twice today it already had.

    KAGGLE_FINAL.txt was a stale copy of final_run.py and still carried the
    benchmark bug; changes_gen.py existed at the root and under scripts/, and
    the second copy could not even import because paper_refs_v2.py sits beside
    the first."""
    root = {f for f in os.listdir(REPO) if f.endswith(".py")}
    here = {f for f in os.listdir(HERE) if f.endswith(".py")}
    dup = sorted(root & here)
    print("  python files in both the root and scripts/ : %s"
          % (", ".join(dup) if dup else "none"))
    return not dup


def check_readme_headline_is_current():
    """The README's front table must match the records.

    Every figure in it was typed. After the Subject-header correction the
    records changed and the table did not, so the first thing a reader saw was
    1,069 threads instead of 1,103 and the best strict method at 0.359 instead
    of 0.410. It is generated now; this checks it was regenerated."""
    import subprocess
    readme = os.path.join(REPO, "README.md")
    before = io.open(readme, encoding="utf-8").read()
    r = subprocess.run([sys.executable,
                        os.path.join(HERE, "readme_headline.py")],
                       capture_output=True, text=True)
    after = io.open(readme, encoding="utf-8").read()
    if r.returncode != 0:
        print("  regenerating the headline table failed: %s"
              % r.stderr.strip()[:120])
        return False
    same = before == after
    print("  README headline table matches the records : %s"
          % ("yes" if same else "NO - it was stale and has been regenerated"))
    return same


def check_readme_contents_is_current():
    """The contents list must cover every heading.

    Written by hand once, it was two sections out of date the next time one
    was added. It is generated; this checks it was regenerated."""
    import subprocess
    readme = os.path.join(REPO, "README.md")
    before = io.open(readme, encoding="utf-8").read()
    r = subprocess.run([sys.executable,
                        os.path.join(HERE, "readme_contents.py")],
                       capture_output=True, text=True)
    after = io.open(readme, encoding="utf-8").read()
    if r.returncode != 0:
        print("  regenerating the contents failed: %s"
              % r.stderr.strip()[:120])
        return False
    same = before == after
    print("  README contents covers every section      : %s"
          % ("yes" if same else "NO - it was stale and has been regenerated"))
    return same


def check_final_md_is_current():
    """RESULTS_FINAL.md must match the record it describes.

    It was hand-written, and by 25 September it had drifted onto the withdrawn
    1,069-thread build: fingerprint 50b3daba1a99ae32, and five of the ten
    record F1s in it appeared nowhere in the JSON the same session wrote. The
    README points a reader at it as the source of Tables I-VI. It is generated
    now; this checks it was regenerated."""
    import subprocess
    p = os.path.join(REPO, "results", "RESULTS_FINAL.md")
    before = io.open(p, encoding="utf-8").read()
    r = subprocess.run([sys.executable, os.path.join(HERE, "final_md.py")],
                       capture_output=True, text=True)
    after = io.open(p, encoding="utf-8").read()
    if r.returncode != 0:
        print("  regenerating RESULTS_FINAL.md failed: %s"
              % r.stderr.strip()[:120])
        return False
    same = before == after
    print("  RESULTS_FINAL.md matches its record       : %s"
          % ("yes" if same else "NO - it was stale and has been regenerated"))
    return same


def check_tie_diagnostic_figures():
    """The strict-vs-inclusive box must match the measurement it reports.

    RESULTS_EXTERNAL_CALIBRATION.md and README.md both carry the Enron-Spam
    contrast between counting `neg > t` and `neg >= t`. Both were typed, and
    both drifted onto a superseded run: they said 0.8367 delivered and 76.2%
    on the cut while the released record said 0.6676 and 59.5% for the same
    cell. The figures come from RESULTS_TIE_DIAGNOSTIC.json now; this checks
    the two documents still print what that record holds."""
    rec = os.path.join(REPO, "results", "RESULTS_TIE_DIAGNOSTIC.json")
    if not os.path.isfile(rec):
        print("  tie-diagnostic figures                    : "
              "NO - run scripts/tie_diagnostic.py")
        return False
    r = json.load(io.open(rec, encoding="utf-8"))
    want = ["%.4f" % r["delivered_strict"],
            "%.4f" % r["delivered_inclusive"],
            "%.1f%%" % (100 * r["on_cut_share"]),
            "{:,}".format(r["on_cut"]),
            "{:,}".format(r["negatives"])]
    ok = True
    for name in ("README.md", os.path.join(
            "results", "RESULTS_EXTERNAL_CALIBRATION.md")):
        txt = io.open(os.path.join(REPO, name), encoding="utf-8").read()
        # The superseded figures may still be named as history, but only in a
        # sentence that says so; they must not stand as the current values.
        missing = [w for w in want if w not in txt]
        if missing:
            ok = False
            print("  %-41s: NO - missing %s" % (name, ", ".join(missing)))
    print("  tie-diagnostic figures                    : %s"
          % ("yes" if ok else "NO"))
    return ok


def main():
    ok = True
    print("=== the README front table ===")
    ok &= check_readme_headline_is_current()
    ok &= check_readme_contents_is_current()
    ok &= check_final_md_is_current()
    ok &= check_tie_diagnostic_figures()
    print()
    print("=== one copy of each file ===")
    ok &= check_no_duplicate_files()
    print()
    print("=== absolute paths ===")
    ok &= check_no_machine_paths()
    print()
    print("=== result files resolve to the repository ===")
    ok &= check_results_resolve_to_repo()
    print()
    print("=== corpus resolution order ===")
    ok &= check_corpus_resolution_order()
    print()
    print("paths are portable" if ok else "PATHS ARE NOT PORTABLE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
