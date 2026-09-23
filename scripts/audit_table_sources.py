# -*- coding: utf-8 -*-
"""Every table row must match the record its caption points at.

Table X's caption said its English figures were "FROM THE RUN OF TABLE III".
They were not, and had not been since Tables III and IV were rebuilt under one
treatment: Table X prints LinearSVM 0.264 where Table III prints 0.381, a gap
of 0.117 F1. The figures themselves were right - they are the cross-lingual
run's own English arm - but the caption sent a reader to a table they do not
come from, which is the same defect as a mis-attributed citation.

The earlier check only asked whether each printed number appears somewhere in
the release. That passes a number sitting in the wrong row, under the wrong
caption, attributed to the wrong run. This one binds a row to a record.
"""
import io
import json
import os
import re

from pypdf import PdfReader

PDF = r"C:\Users\lenovo\Downloads\Research_Paper_v2_Bakhaty.pdf"
RES = r"C:\Users\lenovo\Downloads\artifact\results"


def load(n):
    return json.load(io.open(os.path.join(RES, n), encoding="utf-8"))


F = load("RESULTS_FINAL.json")
U = load("RESULTS_UNIFORM.json")
A = load("RESULTS_arabic_v2.json")
L = load("RESULTS_LLM.json")

# label -> (record, [fields in the order the table prints them])
ROWS = [
    ("III", "LinearSVM", U["LinearSVM_uniform_strict"],
     ["f1", "precision", "recall", "mcc", "roc_auc"]),
    ("III", "LogReg", U["LogReg_uniform_strict"],
     ["f1", "precision", "recall", "mcc", "roc_auc"]),
    ("III", "Charn-grams", U["CharNgram_uniform_strict"],
     ["f1", "precision", "recall", "mcc", "roc_auc"]),
    ("III", "RoBERTa", F["transformer_grouped_strict"],
     ["f1", "precision", "recall", "mcc", "roc_auc"]),
    ("IV", "LinearSVM", U["LinearSVM_uniform_broad"],
     ["f1", "precision", "recall", "mcc", "roc_auc"]),
    ("IV", "LogReg", U["LogReg_uniform_broad"],
     ["f1", "precision", "recall", "mcc", "roc_auc"]),
    ("IV", "Charn-grams", U["CharNgram_uniform_broad"],
     ["f1", "precision", "recall", "mcc", "roc_auc"]),
    ("IV", "RoBERTa", F["transformer_grouped_broad"],
     ["f1", "precision", "recall", "mcc", "roc_auc"]),
    # Table X prints the cross-lingual run's OWN arms, English included.
    ("X", "LinearSVMEnglish", A["LinearSVM_english"],
     ["f1", "mcc", "roc_auc", "pr_auc"]),
    ("X", "LogRegEnglish", A["LogReg_english"],
     ["f1", "mcc", "roc_auc", "pr_auc"]),
    ("X", "RoBERTaEnglish", A["roberta_english"],
     ["f1", "mcc", "roc_auc", "pr_auc"]),
    ("X", "AraBERTARprep.", A["arabert_arabic_preprocessed"],
     ["f1", "mcc", "roc_auc", "pr_auc"]),
    ("X", "LogRegARprep.", A["LogReg_arabic_preprocessed"],
     ["f1", "mcc", "roc_auc", "pr_auc"]),
]

# Table IX prints Labels, k, F1, 95% CI, MCC, ROC at FOUR decimals, not the
# five-column three-decimal layout of Tables III and IV. Checked on its own
# terms rather than forced into the other shape.
ROWS_4DP = [
    ("IX", "strict0", L["Qwen2.5-7B_zeroshot_strict"], ["f1"]),
    ("IX", "strict6", L["Qwen2.5-7B_fewshot_strict"], ["f1"]),
    ("IX", "broad0", L["Qwen2.5-7B_zeroshot_broad"], ["f1"]),
    ("IX", "broad6", L["Qwen2.5-7B_fewshot_broad"], ["f1"]),
]

txt = " ".join(p.extract_text() or "" for p in PdfReader(PDF).pages)
flat = re.sub(r"\s+", "", txt)

bad = 0
for table, label, rec, fields in ROWS:
    seq = "".join("%.3f" % rec[f] for f in fields)
    ok = (label + seq) in flat
    print("Table %-4s %-18s %-30s %s"
          % (table, label, seq, "row matches its record"
             if ok else "*** DOES NOT MATCH ***"))
    bad += (not ok)

for table, label, rec, fields in ROWS_4DP:
    seq = "".join("%.4f" % rec[f] for f in fields)
    ok = (label + seq) in flat
    print("Table %-4s %-18s %-30s %s"
          % (table, label, seq, "row matches its record"
             if ok else "*** DOES NOT MATCH ***"))
    bad += (not ok)

# the MCC and ROC columns of Table IX, checked separately because the CI sits
# between them in the printed row
for _, label, rec, _ in ROWS_4DP:
    seq = "%.4f%.4f" % (rec["mcc"], rec["roc_auc"])
    ok = seq in flat
    print("Table IX   %-18s %-30s %s"
          % (label + " MCC/ROC", seq, "matches its record"
             if ok else "*** DOES NOT MATCH ***"))
    bad += (not ok)

print()
print("rows not matching their record:", bad)

# and the caption must not claim a source the numbers do not have
CAPTION_CLAIMS = [
    ("ENGLISHFIGURESAREFROMTHERUNOFTABLEIII", False,
     "Table X's English rows are the cross-lingual run's own, not Table III's"),
    ("THEENGLISHROWSARETHEREFORENOTTHETABLEIIIFIGURES", True,
     "Table X says where its English rows come from"),
]
print()
for probe, want, why in CAPTION_CLAIMS:
    got = probe in flat
    print("  %-6s %s" % ("ok" if got == want else "WRONG", why))
    bad += (got != want)

raise SystemExit(1 if bad else 0)
