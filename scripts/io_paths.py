# -*- coding: utf-8 -*-
"""Where result files are read from and written to, resolved rather than
hardcoded.

Fourteen absolute paths into one machine's Downloads folder were spread across
eleven released scripts. The ones that only wrote there failed on anyone
else's machine with a permission or missing-directory error. The ones that
read were worse: `score_consistency.py` and `fig_ladder.py` are pointed at by
the README as the checks a reader can run, and they read files by a path no
reader has - while a copy of each sits in this repository's `results/`.

  result_in(name)   read:  $RESULTS_DIR, then <repo>/results, then
                           ~/Downloads.
  result_out(name)  write: $RESULTS_DIR if set, else ~/Downloads.

No author-specific fallback: in a clone <repo>/results always exists and comes
first, and on the machine that produced the runs ~/Downloads is that same
directory.

Writes keep their old destination by default so the author's build steps are
unchanged; a reader who wants everything in one place sets RESULTS_DIR.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _dirs():
    env = os.environ.get("RESULTS_DIR")
    out = [env] if env else []
    out += [os.path.join(REPO, "results"),
            os.path.join(REPO, "results", "superseded"),
            os.path.join(os.path.expanduser("~"), "Downloads")]
    return out


def result_in(name, required=False):
    """The first existing copy of `name`.

    Returns the repository's path when nothing exists, so an error message
    names somewhere sensible."""
    for d in _dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    if required:
        raise SystemExit(
            "%s was not found.\nLooked in:\n  %s\n\n"
            "Set RESULTS_DIR to the directory holding it."
            % (name, "\n  ".join(_dirs())))
    return os.path.join(REPO, "results", name)


def result_out(name):
    """Where a script should write. $RESULTS_DIR, else ~/Downloads."""
    d = os.environ.get("RESULTS_DIR") or os.path.join(
        os.path.expanduser("~"), "Downloads")
    os.path.isdir(d) or os.makedirs(d)
    return os.path.join(d, name)
