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
"""

QUOTED_FIGURES = {
    # Zainab et al., arXiv:2608.16928 - reference [37]
    "zainab_bert_acc": (89.14, "arXiv:2608.16928 abstract"),
    "zainab_bert_f1": (89.33, "arXiv:2608.16928 abstract"),

    # Ivchenko, CTSCAN, arXiv:2604.15561 - reference [38]
    "ctscan_dice_slice_mixed": (0.6665, "arXiv:2604.15561 abstract"),
    "ctscan_dice_case_disjoint": (0.2066, "arXiv:2604.15561 abstract"),
    "ctscan_relative_drop_pct": (69.00, "arXiv:2604.15561 abstract"),

    # Ntwali et al., arXiv:2506.22305, Table 2 - reference [39].
    # These are a re-evaluation of CASSED [12], not CASSED's own reported
    # figures; the original paper reports a weighted F1 on its own in-house
    # real-world set, which is the discrepancy the paper discusses.
    "cassed_dessi_macro_f1": (0.996, "arXiv:2506.22305 Table 2"),
    "cassed_kaggle_macro_f1": (0.349, "arXiv:2506.22305 Table 2"),
    "cassed_openml_macro_f1": (0.501, "arXiv:2506.22305 Table 2"),
    "gpt4o_dessi_macro_f1": (0.766, "arXiv:2506.22305 Table 2"),
    "gpt4o_kaggle_macro_f1": (0.902, "arXiv:2506.22305 Table 2"),
    "gpt4o_openml_macro_f1": (0.964, "arXiv:2506.22305 Table 2"),

    # Kuzina et al., CASSED, ESWA 223:119924 - reference [12]
    "cassed_inhouse_weighted_f1": (0.976, "ESWA 223:119924, Section 5.3"),

    # Telkamp and Hulsebos, arXiv:2512.04120 - reference [41]
    "telkamp_recall": (94, "arXiv:2512.04120 abstract"),
    "telkamp_commercial_recall": (63, "arXiv:2512.04120 abstract"),

    # Antypas et al., WOAH 2025, pp. 17-31 - reference [42]
    "antypas_gpt4o_zeroshot_binary": (75.7, "2025.woah-1.2, results table"),
    "antypas_best_finetuned_binary": (85.6, "2025.woah-1.2, Section 5.1"),
}
