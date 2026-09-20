# -*- coding: utf-8 -*-
"""Reference list v2 - every entry verified 2026-09-15 against Crossref,
arXiv, ACL Anthology or IEEE Xplore. Ordered by first citation (IEEE).

The supervisor's review flagged that a dozen entries were title-only:
  "In a rigor-branded paper this reads as unverified or machine-inserted...
   One dead citation poisons the whole thing."

Verification outcome:
  * 32/32 entries resolve to a real publication. Nothing was invented.
  * 17 entries had NO author list. All 17 now carry full author lists.
  * 1 entry was MIS-ATTRIBUTED: [16] was credited to McDonald, Macdonald &
    Ounis; it is by Sayed, Mallekav & Oard. This was the worst defect found -
    confidently wrong is worse than incomplete.
  * 1 entry had the wrong year: [18] is 2025 (vol. 33, no. 1), not 2023.
    Crossref shows online-first December 2023, print March 2025.
  * 1 title was truncated: [32] ends "...on Reddit".
  * Entries also gained missing volume/issue/page data where Crossref had it.

Sources used for verification (per entry, in VERIFIED below):
  crossref = api.crossref.org/works/<doi>   arxiv = arxiv.org/abs/<id>
  acl = aclanthology.org                    ieee = ieeexplore.ieee.org
"""

REFS = [
    # [1] Section I - the commercial state of practice
    "Microsoft, “Microsoft Purview Data Loss Prevention documentation,” "
    "Microsoft Learn, 2026. [Online]. Available: "
    "https://learn.microsoft.com/purview/dlp-learn-about-dlp",

    # [2]-[8] Section II-A, explicit sensitive information (PII/NER)
    "L. Mainetti and A. Elia, “Detecting personally identifiable "
    "information through natural language processing: A step forward,” "
    "Applied System Innovation, vol. 8, no. 2, art. 55, 2025, "
    "doi:10.3390/asi8020055.",

    "M. H. Shahriar, A. V. D. M. Kayem, D. Reich and C. Meinel, "
    "“Identifying personal identifiable information (PII) in unstructured "
    "text: A comparative study on transformers,” in Database and Expert "
    "Systems Applications (DEXA), LNCS, Springer, 2024, pp. 174-181, "
    "doi:10.1007/978-3-031-68312-1_14.",

    "J. Muralitharan and C. Arumugam, “Privacy BERT-LSTM: A novel NLP "
    "algorithm for sensitive information detection in textual documents,” "
    "Neural Computing and Applications, vol. 36, no. 25, pp. 15439-15454, "
    "2024, doi:10.1007/s00521-024-09707-w.",

    "A. K. M. N. Mehdy and H. Mehrpouyan, “A multi-input multi-output "
    "transformer-based hybrid neural network for multi-class privacy "
    "disclosure detection,” arXiv:2108.08483, 2021.",

    "M. I. Szawerna, S. Dobnik, R. Muñoz Sánchez, "
    "T. Lindström Tiedemann and E. Volodina, “Detecting personal "
    "identifiable information in Swedish learner essays,” in Proc. "
    "Workshop on Computational Approaches to Language Data Pseudonymization "
    "(CALD-pseudo), ACL, 2024, pp. 54-63.",

    "H. Rajgarhia, S. Gupta, A. Shaik, G. P. Kumar, Y. Santhoshraj, "
    "S. N. T. Nishitha and A. Mukherji, “An evaluation study of hybrid "
    "methods for multilingual PII detection,” arXiv:2510.07551, 2025.",

    "O. Elbarbary, M. Rasslan, A. El Bolock and C. Sabty, “Hybrid AI for "
    "Arabic sensitive data detection: Enhancing privacy compliance in "
    "Egypt,” International Journal of Safety and Security Engineering, "
    "vol. 15, no. 6, pp. 1103-1109, 2025, doi:10.18280/ijsse.150602.",

    # [9]-[14] Section II-B, context-dependent sensitive data
    "H. Ahmed, I. Traore, S. Saad and M. Mamun, “Automated detection of "
    "unstructured context-dependent sensitive information using deep "
    "learning,” Internet of Things, vol. 16, art. 100444, 2021, "
    "doi:10.1016/j.iot.2021.100444.",

    "H. M. Qawara and H. Alhindi, “Detecting context-dependent sensitive "
    "data in unstructured text,” Information, vol. 17, no. 7, art. 663, "
    "2026, doi:10.3390/info17070663.",

    "V. Kužina, A.-M. Petrić, M. Barišić and A. Jović, "
    "“CASSED: Context-based approach for structured sensitive data "
    "detection,” Expert Systems with Applications, vol. 223, art. 119924, "
    "2023, doi:10.1016/j.eswa.2023.119924.",

    "J. Neerbek, “Sensitive information detection: Recursive neural "
    "networks for encoding context,” arXiv:2008.10863, 2020.",

    "S. Anand, M. Shukla and S. Lodha, “Detecting sensitive information "
    "from unstructured text in a data-constrained environment,” in Proc. "
    "15th Int. Conf. COMmunication Systems & NETworkS (COMSNETS), IEEE, 2023.",

    "G. Gambarelli, A. Gangemi and R. Tripodi, “Is your model sensitive? "
    "SPeDaC: A new benchmark for detecting and classifying sensitive personal "
    "data,” arXiv:2208.06216, 2022.",

    # [15]-[18] Section II-C, sensitivity review
    "G. McDonald, C. Macdonald and I. Ounis, “How the accuracy and "
    "confidence of sensitivity classification affects digital sensitivity "
    "review,” ACM Trans. Information Systems, vol. 39, no. 1, art. 4, "
    "2020, doi:10.1145/3417334.",

    # CORRECTED: v1 credited this to McDonald, Macdonald and Ounis.
    "M. F. Sayed, N. Mallekav and D. W. Oard, “Comparing intrinsic and "
    "extrinsic evaluation of sensitivity classification,” in Advances in "
    "Information Retrieval (ECIR), LNCS, Springer, 2022, pp. 215-222, "
    "doi:10.1007/978-3-030-99739-7_25.",

    "G. McDonald, C. Macdonald and I. Ounis, “The FACTS of "
    "technology-assisted sensitivity review,” arXiv:1907.02956, 2019.",

    # CORRECTED: v1 gave the year as 2023 (online-first).
    "K. Branting, B. Brown, C. Giannella, J. Van Guilder, J. Harrold, "
    "S. Howell and J. R. Baron, “Decision support for detecting sensitive "
    "text in government records,” Artificial Intelligence and Law, "
    "vol. 33, no. 1, pp. 171-197, 2025, doi:10.1007/s10506-023-09383-6.",

    # [19]-[23] Section II-D, LLMs and contextual privacy
    "N. Mireshghallah, H. Kim, X. Zhou, Y. Tsvetkov, Y. Choi, M. Sap and "
    "T. Berg-Kirkpatrick, “Can LLMs keep a secret? Testing privacy "
    "implications of language models via contextual integrity theory,” "
    "in Proc. Int. Conf. Learning Representations (ICLR), 2024.",

    "H. Li, W. Hu, H. Jing, Y. Chen, Q. Hu, S. Han, T. Chu, P. Hu and "
    "Y. Song, “PrivaCI-Bench: Evaluating privacy with contextual "
    "integrity and legal compliance,” arXiv:2502.17041, 2025.",

    "M. Miranda, E. S. Ruzzetti, A. Santilli, F. M. Zanzotto, "
    "S. Bratières and E. Rodolà, “Preserving privacy in large "
    "language models: A survey on current threats and solutions,” "
    "arXiv:2408.05212, 2024.",

    "K. Chen, X. Zhou, Y. Lin, S. Feng, L. Shen and P. Wu, “A survey on "
    "privacy risks and protection in large language models,” J. King Saud "
    "Univ. - Computer and Information Sciences, vol. 37, no. 7, art. 163, "
    "2025, doi:10.1007/s44443-025-00177-1.",

    "P. Desai, L. Tang, Y. Meng and Z. Xi, “SafeGPT: Preventing data "
    "leakage and unethical outputs in enterprise LLM use,” "
    "arXiv:2601.06366, 2026.",

    # [24]-[26] Section II-E, evaluation methodology
    "S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis "
    "in machine-learning-based science,” Patterns, vol. 4, no. 9, "
    "art. 100804, 2023, doi:10.1016/j.patter.2023.100804.",

    "D. Chicco and G. Jurman, “The advantages of the Matthews correlation "
    "coefficient (MCC) over F1 score and accuracy in binary classification "
    "evaluation,” BMC Genomics, vol. 21, art. 6, 2020, "
    "doi:10.1186/s12864-019-6413-7.",

    "M. Szep, D. Rueckert, R. von Eisenhart-Rothe and F. Hinterwimmer, "
    "“Fine-tuning large language models with limited data: A survey and "
    "practical guide,” arXiv:2411.09539, 2024.",

    # [27] Section III, model
    "J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, “BERT: "
    "Pre-training of deep bidirectional transformers for language "
    "understanding,” in Proc. NAACL-HLT, 2019, pp. 4171-4186.",

    # [28]-[30] Section IV, data
    "B. Klimt and Y. Yang, “The Enron corpus: A new dataset for email "
    "classification research,” in Proc. 15th European Conf. Machine "
    "Learning (ECML), LNCS 3201, Springer, 2004, pp. 217-226.",

    "D. Noever, “The Enron corpus: Where the email bodies are "
    "buried?,” arXiv:2001.10374, 2020.",

    "J. McKechnie, G. McDonald and C. Macdonald, “A sensitivity-aware "
    "test collection for search among personal information,” "
    "arXiv:2606.27559, 2026.",

    # [31]-[32] Section V, baselines
    "Meta AI, “Llama 3.2: Revolutionizing edge AI and vision with open, "
    "customizable models,” Meta AI Blog, 2024.",

    # CORRECTED: v1 truncated the title before "on Reddit".
    "J. Lee, L. Tian, A. Brillantes, A.-S. Mihăiţă and "
    "M.-A. Rizoiu, “Long live fine-tuning: Task-specific transformers "
    "outperform zero-shot LLMs for misinformation response classification on "
    "Reddit,” arXiv:2606.04274, 2026.",

    # [33] Section VII-E, the reproducibility finding
    "scikit-learn developers, “GroupKFold now uses stable sorting when "
    "doing the group distribution; this ensures that the splits are "
    "consistent across runs,” in Release Notes for scikit-learn 1.9, "
    "pull request #28464, 2025. [Online]. Available: "
    "https://scikit-learn.org/stable/whats_new/v1.9.html",

    # [34]-[35] Section VIII, the cross-lingual extension
    "W. Antoun, F. Baly and H. Hajj, “AraBERT: Transformer-based model for "
    "Arabic language understanding,” in Proc. 12th Int. Conf. Language "
    "Resources and Evaluation (LREC), Marseille, France, 2020.",

    "NLLB Team, M. R. Costa-jussà, J. Cross, O. Çelebi, M. Elbayad, "
    "K. Heafield, K. Heffernan, E. Kalbassi, J. Lam, D. Licht, J. Maillard, "
    "A. Sun, S. Wang, G. Wenzek, A. Youngblood et al., “No language left "
    "behind: Scaling human-centered machine translation,” arXiv:2207.04672, "
    "2022.",
]

# How each entry was checked, for the artifact release.
VERIFIED = {
    1: "vendor documentation - URL added, no authors expected",
    2: "crossref 10.3390/asi8020055", 3: "crossref 10.1007/978-3-031-68312-1_14",
    4: "crossref 10.1007/s00521-024-09707-w", 5: "arxiv 2108.08483",
    6: "acl 2024.caldpseudo-1.7", 7: "arxiv 2510.07551",
    8: "crossref 10.18280/ijsse.150602", 9: "crossref 10.1016/j.iot.2021.100444",
    10: "crossref 10.3390/info17070663", 11: "crossref 10.1016/j.eswa.2023.119924",
    12: "arxiv 2008.10863", 13: "ieee 10041388", 14: "arxiv 2208.06216",
    15: "acm 10.1145/3417334",
    16: "crossref 10.1007/978-3-030-99739-7_25 - ATTRIBUTION CORRECTED",
    17: "arxiv 1907.02956",
    18: "crossref 10.1007/s10506-023-09383-6 - YEAR CORRECTED 2023->2025",
    19: "ICLR 2024 proceedings", 20: "arxiv 2502.17041", 21: "arxiv 2408.05212",
    22: "crossref 10.1007/s44443-025-00177-1", 23: "arxiv 2601.06366",
    24: "crossref 10.1016/j.patter.2023.100804",
    25: "crossref 10.1186/s12864-019-6413-7", 26: "arxiv 2411.09539",
    27: "NAACL-HLT 2019 proceedings", 28: "ECML 2004 LNCS 3201",
    29: "arxiv 2001.10374", 30: "arxiv 2606.27559",
    31: "vendor blog - no authors expected",
    32: "arxiv 2606.04274 - TITLE COMPLETED",
    33: "scikit-learn 1.9 release notes, quoted verbatim; PR #28464",
    34: "arxiv 2003.00104 / LREC 2020 proceedings",
    35: "arxiv 2207.04672 - collective authorship, first 15 named then et al.",
}

assert len(REFS) == 35, len(REFS)
assert len(VERIFIED) == 35, len(VERIFIED)
