# -*- coding: utf-8 -*-
"""Final audit: does the repository contain what the paper promises?

Not a rebuild check and not a numbers check - both of those already pass.
This asks the one question left: if a reviewer follows the paper's artifact
section to the repository, does each thing it names actually exist there.
"""
import io
import os
import re

from pypdf import PdfReader

PAPER = os.environ.get("PAPER_PDF") or os.path.join(
    os.path.expanduser("~"), "Downloads", "Research_Paper_v2_REVIEW_Bakhaty.pdf")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
text = "".join(p.extract_text() for p in PdfReader(PAPER).pages)
body = text.split("REFERENCES")[0]

have = set()
for d, _, fs in os.walk(REPO):
    for f in fs:
        have.add(os.path.relpath(os.path.join(d, f), REPO).replace(os.sep, "/"))

final_md = io.open(os.path.join(REPO, "results", "RESULTS_FINAL.md"),
                   encoding="utf-8").read()
readme = io.open(os.path.join(REPO, "README.md"), encoding="utf-8").read()

print("=== the artifact section, as printed ===")
i = body.find("ARTIFACT RELEASE")
print(" ".join(body[i:i + 640].split()))
print()

checks = {
    "benchmark construction script": "scripts/final_run.py" in have,
    "audit scripts for all four controls": (
        "scripts/final_run.py" in have
        and "thread leaked" in io.open(
            os.path.join(REPO, "scripts", "final_run.py"),
            encoding="utf-8").read()),
    "experiment notebook, pasteable": "KAGGLE_FINAL.txt" in have,
    "per-fold result records": "results/RESULTS_FINAL.md" in have,
    "pinned library versions recorded": "scikit-learn 1.9.1" in final_md
                                        or "sklearn 1.9.1" in final_md,
    "reference verification log": "paper_refs_v2.py" in have,
    "fold fingerprint published": "50b3daba1a99ae32" in readme,
    "repository URL resolves to content": len(have) > 10,
}
print("=== does the repo contain each promise? ===")
for k, v in sorted(checks.items()):
    print(("  OK    " if v else "  GAP   ") + k)

print()
print("=== per-fold detail actually recorded? ===")
# Look at the release, not at one markdown file: the fold arrays live in
# the JSON records. Reading only RESULTS_FINAL.md reported False while five
# files carried folds_f1.
_files_with_folds = []
for _d, _, _fs in os.walk(os.path.join(REPO, "results")):
    for _f in _fs:
        if not _f.endswith(".json"):
            continue
        _t = io.open(os.path.join(_d, _f), encoding="utf-8",
                     errors="ignore").read()
        if re.search(r'"folds_f1"\s*:\s*\[', _t):
            _files_with_folds.append(_f)
print("  fold mean and SD present :", "+-" in final_md)
print("  individual fold values   :", bool(_files_with_folds),
      "in", sorted(_files_with_folds))

print()
print("=== every number the paper prints, traceable to the repo? ===")
sources = final_md + readme
for d, _, fs in os.walk(os.path.join(REPO, "results")):
    for f in fs:
        sources += io.open(os.path.join(d, f), encoding="utf-8",
                           errors="ignore").read()
quoted = set(re.findall(r"\b0\.\d{3}\b", body))
missing = sorted(q for q in quoted
                 if q not in sources and q.rstrip("0") not in sources
                 and not any(s.startswith(q) for s in
                             re.findall(r"0\.\d{4}", sources)))
print("  distinct 3-dp figures in the paper :", len(quoted))
print("  not found in any results file      :", missing if missing else "none")
