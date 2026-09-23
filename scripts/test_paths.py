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


def main():
    ok = True
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
