# -*- coding: utf-8 -*-
"""Where every figure quoted from another paper was read from.

The reference list itself lives in paper_refs_v2.py; this file exists for the
numbers, not the citations. A figure taken from someone else's paper cannot be
regenerated from this repository, so the only thing that makes it checkable is
a record of which page it was read off. Each entry is
(value, source read in the session of 21 September 2026).

Entries were read from the paper's own arXiv page, ACL Anthology page or
publisher PDF. None came from a search summary - two of these figures were
initially taken from one and were wrong, which is why this file exists.

Reference numbers are NOT written here. They were, and four of the six had
drifted by one after an entry was inserted into the list, so a reader chasing
a quoted number was sent to the wrong reference. They are now looked up from
paper_refs_v2.py by the identifier that cannot drift - an arXiv id, a DOI or
an ACL anthology id - and the lookup fails the import if any identifier stops
resolving to exactly one entry.
"""
import paper_refs_v2 as _R


def _ref(identifier):
    """The IEEE number of the entry carrying this identifier.

    Raises rather than returning a wrong number: a citation pointing at the
    wrong entry is the defect this file exists to prevent."""
    hits = [i for i, r in enumerate(_R.REFS, 1) if identifier in r]
    if len(hits) != 1:
        raise AssertionError(
            "%r matches %d references, expected exactly 1: %s"
            % (identifier, len(hits), hits))
    return hits[0]


# The works these figures come from, by an identifier that cannot drift.
SOURCES = {
    "zainab": "arXiv:2608.16928",
    "ctscan": "arXiv:2604.15561",
    "ntwali": "arXiv:2506.22305",
    "cassed": "10.1016/j.eswa.2023.119924",
    "telkamp": "arXiv:2512.04120",
    "antypas": "2025.woah-1.2",
}
REF_NUMBERS = {k: _ref(v) for k, v in SOURCES.items()}

QUOTED_FIGURES = {
    # Zainab et al. - reference [REF_NUMBERS["zainab"]]
    "zainab_bert_acc": (89.14, "arXiv:2608.16928 abstract"),
    "zainab_bert_f1": (89.33, "arXiv:2608.16928 abstract"),

    # Ivchenko, CTSCAN - reference [REF_NUMBERS["ctscan"]]
    "ctscan_dice_slice_mixed": (0.6665, "arXiv:2604.15561 abstract"),
    "ctscan_dice_case_disjoint": (0.2066, "arXiv:2604.15561 abstract"),
    "ctscan_relative_drop_pct": (69.00, "arXiv:2604.15561 abstract"),

    # Ntwali et al., Table 2 - reference [REF_NUMBERS["ntwali"]].
    # These are a re-evaluation of CASSED, not CASSED's own reported figures;
    # the original paper reports a weighted F1 on its own in-house real-world
    # set, which is the discrepancy the paper discusses.
    "cassed_dessi_macro_f1": (0.996, "arXiv:2506.22305 Table 2"),
    "cassed_kaggle_macro_f1": (0.349, "arXiv:2506.22305 Table 2"),
    "cassed_openml_macro_f1": (0.501, "arXiv:2506.22305 Table 2"),
    "gpt4o_dessi_macro_f1": (0.766, "arXiv:2506.22305 Table 2"),
    "gpt4o_kaggle_macro_f1": (0.902, "arXiv:2506.22305 Table 2"),
    "gpt4o_openml_macro_f1": (0.964, "arXiv:2506.22305 Table 2"),

    # Kuzina et al., CASSED - reference [REF_NUMBERS["cassed"]]
    "cassed_inhouse_weighted_f1": (0.976, "ESWA 223:119924, Section 5.3"),

    # Telkamp and Hulsebos - reference [REF_NUMBERS["telkamp"]]
    "telkamp_recall": (94, "arXiv:2512.04120 abstract"),
    "telkamp_commercial_recall": (63, "arXiv:2512.04120 abstract"),

    # Antypas et al., WOAH 2025 - reference [REF_NUMBERS["antypas"]]
    "antypas_gpt4o_zeroshot_binary": (75.7, "2025.woah-1.2, results table"),
    "antypas_best_finetuned_binary": (85.6, "2025.woah-1.2, Section 5.1"),
}


if __name__ == "__main__":
    print("figures quoted from other papers, with the reference each belongs "
          "to\n")
    for k, v in sorted(SOURCES.items()):
        print("  [%2d]  %-12s %s" % (REF_NUMBERS[k], k, v))
    print()
    for k, (val, src) in sorted(QUOTED_FIGURES.items()):
        print("  %-30s %-8s %s" % (k, val, src))
