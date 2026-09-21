# -*- coding: utf-8 -*-
"""What a library upgrade alone does to every figure in the ladder.

The paper reports that moving from scikit-learn 1.8.0 to 1.9.1 shifted a
grouped cross-validation score by 0.024 F1, because 1.9.0 changed GroupKFold to
stable sorting (PR #28464) and the two versions therefore partition the corpus
differently. That is one observation about one number.

Running the whole strengthening suite under both versions turns it into a
measurement: every rung of the ladder, both label sets, same code, same data,
same seed, one thing different. Rungs R0 to R2 use a random split and should
barely move; R3 is the grouped one and is where the change has to show. A rung
that moves when it has no business moving is itself worth knowing about.
"""
import io
import json
import sys

L = r"C:\Users\lenovo\Downloads\RESULTS_strengthen_sklearn180.json"
R = r"C:\Users\lenovo\Downloads\RESULTS_strengthen_sklearn191.json"
OUT = r"C:\Users\lenovo\Downloads\RESULTS_version_effect.md"


def load(p):
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except IOError:
        sys.exit("missing: %s\nRun strengthen.py under that version first." % p)


def main():
    a, b = load(L), load(R)
    va = a["environment"]["scikit_learn"]
    vb = b["environment"]["scikit_learn"]

    lines = ["# What the scikit-learn version alone changes", "",
             "Same code, same corpus, same seed. Only the library differs:",
             "**%s** against **%s**." % (va, vb), "",
             "GroupKFold gained stable sorting in 1.9.0 (PR #28464), so the two "
             "versions assign threads to folds differently. Rungs R0-R2 use a "
             "random message split and are not grouped, so they are the "
             "control: they should not move.", ""]

    for lab in ("strict", "broad"):
        k = "ladder_" + lab
        if k not in a or k not in b:
            continue
        lines += ["## Leakage ladder, %s labels" % lab, "",
                  "| Rung | Control | F1 (%s) | F1 (%s) | change |"
                  % (va, vb), "|---|---|---|---|---|"]
        for ra, rb in zip(a[k], b[k]):
            assert ra["rung"] == rb["rung"], "rungs out of order"
            d = rb["f1"] - ra["f1"]
            flag = " **" if abs(d) >= 0.005 else " "
            lines.append("| %s | %s | %.4f | %.4f |%s%+.4f%s |"
                         % (ra["rung"], ra["control"], ra["f1"], rb["f1"],
                            flag, d, flag.rstrip()))
        lines.append("")

        for what, key, field in (
                ("Repeated cross-validation, pooled F1",
                 "repeated_cv_" + lab, "pooled_f1_mean"),
                ("Permutation null, p-value",
                 "permutation_" + lab, "p_value"),
                ("Permutation null, observed F1",
                 "permutation_" + lab, "observed_f1")):
            if key in a and key in b:
                x, y = a[key][field], b[key][field]
                lines.append("- **%s**: %.4f -> %.4f (%+.4f)"
                             % (what, x, y, y - x))
        lines.append("")

    if "agreement" in a and "agreement" in b:
        lines += ["## Annotator agreement reference", "",
                  "This is computed from the annotation counts and never "
                  "touches a splitter, so it must be identical. If it is not, "
                  "something other than the splitter changed.", ""]
        for f in ("agreement_f1", "cohens_kappa"):
            x, y = a["agreement"][f], b["agreement"][f]
            same = "identical" if abs(x - y) < 1e-9 else "**DIFFERS**"
            lines.append("- %s: %.4f vs %.4f - %s" % (f, x, y, same))
        lines.append("")

    io.open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print()
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
