# -*- coding: utf-8 -*-
"""Is every randomised estimator in this release seeded?

A rerun of classical_seeds.py moved a ROC-AUC from 0.7231 to 0.7232. The cause
was LinearSVC: liblinear's coordinate descent is randomised, and the estimator
was constructed without random_state in ten scripts, including the two that
produce Tables III and IV.

The drift is small - one in ten thousand, and it appears in one configuration
out of the dozens here - which is exactly why it needs a check rather than a
memory. A paper whose argument includes "publish the fold assignment so two
runs can be compared" cannot ship scripts whose own output moves between runs.

This reads the syntax tree rather than grepping, because random_state usually
sits on a continuation line and a line-oriented search reports it as missing.
It fails on any construction of a known-randomised estimator without an
explicit seed.

    python scripts/test_determinism.py
"""
import ast
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Estimators whose fit is randomised unless seeded. LogisticRegression with
# the default lbfgs solver is deterministic and is not listed; it would be if
# the solver were changed to one of the stochastic ones.
RANDOMISED = {
    "LinearSVC": "liblinear's coordinate descent",
    "SGDClassifier": "stochastic gradient descent",
    "RandomForestClassifier": "bootstrap sampling and feature subsetting",
    "ExtraTreesClassifier": "random splits",
    "MLPClassifier": "weight initialisation",
    "KMeans": "centroid initialisation",
}
# A file may legitimately construct one without a seed if it never fits it.
EXEMPT = set()


def offenders(path):
    try:
        tree = ast.parse(io.open(path, encoding="utf-8").read())
    except SyntaxError:
        return [("(does not parse)", 0)]
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (fn.id if isinstance(fn, ast.Name)
                else fn.attr if isinstance(fn, ast.Attribute) else None)
        if name not in RANDOMISED:
            continue
        if any(k.arg == "random_state" for k in node.keywords):
            continue
        bad.append((name, node.lineno))
    return bad


def main():
    files = sorted(f for f in os.listdir(HERE) if f.endswith(".py"))
    checked, bad = 0, []
    for f in files:
        if f in EXEMPT:
            continue
        checked += 1
        for name, line in offenders(os.path.join(HERE, f)):
            bad.append((f, line, name))

    print("%d scripts checked for unseeded randomised estimators" % checked)
    print("estimators treated as randomised: %s"
          % ", ".join(sorted(RANDOMISED)))
    print()
    if not bad:
        print("every randomised estimator is constructed with a seed")
        return 0
    for f, line, name in bad:
        print("  %-26s :%-5d %s (%s)"
              % (f, line, name, RANDOMISED[name]))
    print()
    print("%d unseeded construction(s). Each is a figure that can move "
          "between runs of the same script." % len(bad))
    return 1


if __name__ == "__main__":
    sys.exit(main())
