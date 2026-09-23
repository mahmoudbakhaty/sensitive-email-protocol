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
import sys

from pypdf import PdfReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_paths                                              # noqa: E402

# $PAPER_PDF, else the built paper wherever the other result files live.
PDF = os.environ.get("PAPER_PDF") or os.path.join(
    os.path.expanduser("~"), "Downloads", "Research_Paper_v2_Bakhaty.pdf")


def load(n):
    return json.load(io.open(io_paths.result_in(n, required=True),
                             encoding="utf-8"))


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

P = load("RESULTS_perm_thread.json")
LAD = load("RESULTS_ladder_v2.json")
AG = load("RESULTS_agreement_metrics.json")

# Single values that must appear, each with the record it comes from. Table II
# printed 173 multi-message threads and 55 mixed ones long after the
# Subject-header correction moved them to 171 and 54: the record updated, two
# hand-typed literals in the paper did not.
SINGLES = [
    ("I  messages", "%d" % F["dataset"]["messages"]),
    ("I  threads", "%d" % F["dataset"]["threads"]),
    ("I  median words", "%d" % F["dataset"]["median_words"]),
    ("I  strict positives", "%d" % F["dataset"]["sensitive_strict"]),
    ("I  broad positives", "%d" % F["dataset"]["sensitive_broad"]),
    ("I  observed agreement", (F["annotation"]["observed_agreement"], 3)),
    ("I  chance agreement", (F["annotation"]["chance_agreement"], 3)),
    ("I  kappa", (F["annotation"]["cohens_kappa"], 3)),
    ("II exact duplicates", "messages%d"
     % F["leakage_audit"]["exact_duplicates"]),
    ("II multi-message threads", "messages%d"
     % F["leakage_audit"]["threads_with_multiple_messages"]),
    ("II mixed-label threads", "labels%d"
     % F["leakage_audit"]["threads_with_mixed_labels"]),
    ("II strongest word", "%.3f" % F["leakage_audit"]["best_single_term_f1"]),
    ("VIII strict observed", "%.4f" % P["strict"]["observed_f1"]),
    ("VIII strict null mean", "%.4f" % P["strict"]["message_level"]["null_mean"]),
    ("VIII strict null max", "%.4f" % P["strict"]["message_level"]["null_max"]),
    ("VIII broad observed", "%.4f" % P["broad"]["observed_f1"]),
    ("XI arabic raw vocab", "8,%d"
     % (A["audit_arabic_raw"]["vocabulary_terms"] % 1000)),
    ("XI arabic prep vocab", "8,%d"
     % (A["audit_arabic_preprocessed"]["vocabulary_terms"] % 1000)),
]

# Table VII: every rung of the ladder, F1 then MCC as printed.
for lab in ("strict", "broad"):
    for r in LAD[lab]["rungs"]:
        SINGLES.append(("VII %s %s" % (lab, r["rung"].split()[0]),
                        "%.4f" % r["f1"]))
        SINGLES.append(("VII %s %s MCC" % (lab, r["rung"].split()[0]),
                        "%.4f" % r["mcc"]))

# Table V: the stratified/grouped pairs, from the run that produced them.
for lab in ("strict", "broad"):
    for m in ("LogReg", "LinearSVM"):
        SINGLES.append(("V %s %s strat." % (m, lab),
                        "%.3f" % F["%s_stratified_%s" % (m, lab)]["f1"]))
        SINGLES.append(("V %s %s group." % (m, lab),
                        "%.3f" % F["%s_grouped_%s" % (m, lab)]["f1"]))

txt = " ".join(p.extract_text() or "" for p in PdfReader(PDF).pages)
flat = re.sub(r"\s+", "", txt)

def fmt(value, dp):
    """Every rendering of a record value the paper could legitimately print.

    The record holds 0.8755 and the paper prints 0.876, which is the correct
    round to three places. Python's %.3f gives 0.875, because the binary float
    is a hair below the half. Both are accepted; a check must not call correct
    rounding a defect. Rounding must be taken from the RAW value - taking it
    from an already-rounded string loses the digit that decides it."""
    from decimal import Decimal, ROUND_HALF_UP
    exact = Decimal(repr(float(value)))
    q = Decimal(1).scaleb(-dp)
    return {"%.*f" % (dp, value), str(exact.quantize(q, ROUND_HALF_UP))}


bad = 0
for label, probe in SINGLES:
    forms = fmt(*probe) if isinstance(probe, tuple) else {probe}
    ok = any(f in flat for f in forms)
    print("Table %-24s %-16s %s"
          % (label, "/".join(sorted(forms)), "in the paper" if ok
             else "*** NOT IN THE PAPER ***"))
    bad += (not ok)
print()
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
