# -*- coding: utf-8 -*-
"""Where the Enron archive lives, resolved rather than hardcoded.

Six released scripts carried an absolute path into a temp directory on the
author's machine. They ran here and nowhere else, which makes the release's
central promise - that the benchmark is rebuilt from a public archive rather
than redistributed - untrue for everyone who is not the author.

Resolution order, first hit wins:

  1. $ENRON_DIR, if set. The explicit answer, and what the Kaggle notebooks
     and CI would set.
  2. ./enron_with_categories under the current directory.
  3. enron_with_categories beside the repository root, which is where
     verify_corpus.py extracts it.
  4. The author's original path, kept last so the historical runs still
     reproduce on the machine that produced them.

Nothing here downloads anything: scripts/verify_corpus.py does that, checks
the archive against its recorded sha256, and extracts it where step 3 finds
it.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LEAF = "enron_with_categories"

# Where the published runs were produced. Last resort, never the first.
ORIGINAL = (r"C:\Users\lenovo\AppData\Local\Temp\claude\C--Users-lenovo"
            r"\fccd8d1f-1d07-4097-95b6-2b0cf5cf3dd3\scratchpad\enroncat"
            r"\enron_with_categories")


def candidates():
    env = os.environ.get("ENRON_DIR")
    out = [env] if env else []
    out += [os.path.join(os.getcwd(), LEAF),
            os.path.join(REPO, LEAF),
            os.path.join(REPO, "scripts", "_corpus_check", LEAF),
            ORIGINAL]
    return [c for c in out if c]


def looks_like_corpus(p):
    """A directory with at least one .cats file two levels down."""
    if not p or not os.path.isdir(p):
        return False
    for name in os.listdir(p):
        d = os.path.join(p, name)
        if os.path.isdir(d) and any(f.endswith(".cats") for f in os.listdir(d)):
            return True
    return False


def resolve(required=False):
    """The first candidate that actually holds the corpus.

    Returns the first candidate unchanged when none does, so a caller that
    only wants a path for a message still gets one; pass required=True to
    fail loudly instead."""
    for c in candidates():
        if looks_like_corpus(c):
            return c
    if required:
        raise SystemExit(
            "the Enron archive was not found.\n"
            "Looked in:\n  " + "\n  ".join(candidates()) + "\n\n"
            "Fix it either way:\n"
            "  python scripts/verify_corpus.py      # downloads, checks the "
            "sha256, extracts\n"
            "  set ENRON_DIR=<path to enron_with_categories>")
    return candidates()[0]
