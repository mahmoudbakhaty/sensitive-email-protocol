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

2026-09-23 resolution pass. The earlier rounds checked that each entry HAS an
identifier. This one fetched all 62 targets and compared the title and author
list that came back. Two defects:

  * [20] named T. Berg-Kirkpatrick, who is not an author of that paper, and
    omitted R. Shokri, who is - the same defect class as the [17]
    mis-attribution. Corrected against arXiv:2310.17884 and the ICLR 2024
    proceedings page. Its OpenReview link was also replaced: OpenReview now
    answers automated requests with a browser challenge, so that link cannot
    be verified by anyone's script, a reviewer's included.
  * [49] pointed at agreestat.com/book4/, which returns 404 after a site
    restructure. Re-pointed at store.html, which resolves and lists the
    edition cited.

Three further flags were artefacts of the checking script, not of the list:
[19] Crossref splits "James Van Guilder" as given="James Van"/family="Guilder";
[47] IOS Press stores whole names in the family field; [54] is an "et al."
list by design. All three entries are correct as printed.
"""

REFS = [
    'Verizon Business, “2026 Data Breach Investigations Report,” Verizon, 2026. [Online]. Available: https://www.verizon.com/business/resources/reports/dbir/',
    'Microsoft, “Microsoft Purview Data Loss Prevention documentation,” Microsoft Learn, 2026. [Online]. Available: https://learn.microsoft.com/purview/dlp-learn-about-dlp',
    'L. Mainetti and A. Elia, “Detecting personally identifiable information through natural language processing: A step forward,” Applied System Innovation, vol. 8, no. 2, art. 55, 2025, doi:10.3390/asi8020055.',
    'M. H. Shahriar, A. V. D. M. Kayem, D. Reich and C. Meinel, “Identifying personal identifiable information (PII) in unstructured text: A comparative study on transformers,” in Database and Expert Systems Applications (DEXA), LNCS, Springer, 2024, pp. 174-181, doi:10.1007/978-3-031-68312-1_14.',
    'J. Muralitharan and C. Arumugam, “Privacy BERT-LSTM: A novel NLP algorithm for sensitive information detection in textual documents,” Neural Computing and Applications, vol. 36, no. 25, pp. 15439-15454, 2024, doi:10.1007/s00521-024-09707-w.',
    'A. K. M. N. Mehdy and H. Mehrpouyan, “A multi-input multi-output transformer-based hybrid neural network for multi-class privacy disclosure detection,” arXiv:2108.08483, 2021.',
    'M. I. Szawerna, S. Dobnik, R. Muñoz Sánchez, T. Lindström Tiedemann and E. Volodina, “Detecting personal identifiable information in Swedish learner essays,” in Proc. Workshop on Computational Approaches to Language Data Pseudonymization (CALD-pseudo), ACL, 2024, pp. 54-63, doi:10.18653/v1/2024.caldpseudo-1.7.',
    'H. Rajgarhia, S. Gupta, A. Shaik, G. P. Kumar, Y. Santhoshraj, S. N. T. Nishitha and A. Mukherji, “An evaluation study of hybrid methods for multilingual PII detection,” arXiv:2510.07551, 2025.',
    'O. Elbarbary, M. Rasslan, A. El Bolock and C. Sabty, “Hybrid AI for Arabic sensitive data detection: Enhancing privacy compliance in Egypt,” International Journal of Safety and Security Engineering, vol. 15, no. 6, pp. 1103-1109, 2025, doi:10.18280/ijsse.150602.',
    'H. Ahmed, I. Traore, S. Saad and M. Mamun, “Automated detection of unstructured context-dependent sensitive information using deep learning,” Internet of Things, vol. 16, art. 100444, 2021, doi:10.1016/j.iot.2021.100444.',
    'H. M. Qawara and H. Alhindi, “Detecting context-dependent sensitive data in unstructured text,” Information, vol. 17, no. 7, art. 663, 2026, doi:10.3390/info17070663.',
    'V. Kužina, A.-M. Petrić, M. Barišić and A. Jović, “CASSED: Context-based approach for structured sensitive data detection,” Expert Systems with Applications, vol. 223, art. 119924, 2023, doi:10.1016/j.eswa.2023.119924.',
    'J. Neerbek, “Sensitive information detection: Recursive neural networks for encoding context,” arXiv:2008.10863, 2020.',
    'S. Anand, M. Shukla and S. Lodha, “Detecting sensitive information from unstructured text in a data-constrained environment,” in Proc. 15th Int. Conf. COMmunication Systems & NETworkS (COMSNETS), IEEE, 2023, doi:10.1109/COMSNETS56262.2023.10041388.',
    'G. Gambarelli, A. Gangemi and R. Tripodi, “Is your model sensitive? SPeDaC: A new benchmark for detecting and classifying sensitive personal data,” arXiv:2208.06216, 2022.',
    'G. McDonald, C. Macdonald and I. Ounis, “How the accuracy and confidence of sensitivity classification affects digital sensitivity review,” ACM Trans. Information Systems, vol. 39, no. 1, art. 4, 2020, doi:10.1145/3417334.',
    'M. F. Sayed, N. Mallekav and D. W. Oard, “Comparing intrinsic and extrinsic evaluation of sensitivity classification,” in Advances in Information Retrieval (ECIR), LNCS, Springer, 2022, pp. 215-222, doi:10.1007/978-3-030-99739-7_25.',
    'G. McDonald, C. Macdonald and I. Ounis, “The FACTS of technology-assisted sensitivity review,” arXiv:1907.02956, 2019.',
    'K. Branting, B. Brown, C. Giannella, J. Van Guilder, J. Harrold, S. Howell and J. R. Baron, “Decision support for detecting sensitive text in government records,” Artificial Intelligence and Law, vol. 33, no. 1, pp. 171-197, 2025, doi:10.1007/s10506-023-09383-6.',
    'N. Mireshghallah, H. Kim, X. Zhou, Y. Tsvetkov, M. Sap, R. Shokri and Y. Choi, “Can LLMs keep a secret? Testing privacy implications of language models via contextual integrity theory,” in Proc. Int. Conf. Learning Representations (ICLR), 2024, arXiv:2310.17884. [Online]. Available: https://proceedings.iclr.cc/paper_files/paper/2024/hash/08305d8b2ddab98932c163ea73df065f-Abstract-Conference.html',
    'H. Li, W. Hu, H. Jing, Y. Chen, Q. Hu, S. Han, T. Chu, P. Hu and Y. Song, “PrivaCI-Bench: Evaluating privacy with contextual integrity and legal compliance,” arXiv:2502.17041, 2025.',
    'M. Miranda, E. S. Ruzzetti, A. Santilli, F. M. Zanzotto, S. Bratières and E. Rodolà, “Preserving privacy in large language models: A survey on current threats and solutions,” arXiv:2408.05212, 2024.',
    'K. Chen, X. Zhou, Y. Lin, S. Feng, L. Shen and P. Wu, “A survey on privacy risks and protection in large language models,” J. King Saud Univ. - Computer and Information Sciences, vol. 37, no. 7, art. 163, 2025, doi:10.1007/s44443-025-00177-1.',
    'P. Desai, L. Tang, Y. Meng and Z. Xi, “SafeGPT: Preventing data leakage and unethical outputs in enterprise LLM use,” arXiv:2601.06366, 2026.',
    'J. R. Landis and G. G. Koch, “The measurement of observer agreement for categorical data,” Biometrics, vol. 33, no. 1, pp. 159-174, 1977, doi:10.2307/2529310.',
    'D. Chicco, M. J. Warrens and G. Jurman, “The Matthews correlation coefficient (MCC) is more informative than Cohen’s kappa and Brier score in binary classification assessment,” IEEE Access, vol. 9, pp. 78368-78381, 2021, doi:10.1109/ACCESS.2021.3084050.',
    'S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” Patterns, vol. 4, no. 9, art. 100804, 2023, doi:10.1016/j.patter.2023.100804.',
    'D. Chicco and G. Jurman, “The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation,” BMC Genomics, vol. 21, art. 6, 2020, doi:10.1186/s12864-019-6413-7.',
    'M. Szep, D. Rueckert, R. von Eisenhart-Rothe and F. Hinterwimmer, “Fine-tuning large language models with limited data: A survey and practical guide,” arXiv:2411.09539, 2024.',
    'J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proc. NAACL-HLT, 2019, pp. 4171-4186, doi:10.18653/v1/N19-1423.',
    'B. Klimt and Y. Yang, “The Enron corpus: A new dataset for email classification research,” in Proc. 15th European Conf. Machine Learning (ECML), LNCS 3201, Springer, 2004, pp. 217-226, doi:10.1007/978-3-540-30115-8_22.',
    'D. Noever, “The Enron corpus: Where the email bodies are buried?,” arXiv:2001.10374, 2020.',
    'J. McKechnie, G. McDonald and C. Macdonald, “A sensitivity-aware test collection for search among personal information,” arXiv:2606.27559, 2026.',
    'Meta AI, “Llama 3.2: Revolutionizing edge AI and vision with open, customizable models,” Meta AI Blog, 2024. [Online]. Available: https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/',
    'J. Lee, L. Tian, A. Brillantes, A.-S. Mihăiţă and M.-A. Rizoiu, “Long live fine-tuning: Task-specific transformers outperform zero-shot LLMs for misinformation response classification on Reddit,” arXiv:2606.04274, 2026.',
    'scikit-learn developers, “GroupKFold now uses stable sorting when doing the group distribution; this ensures that the splits are consistent across runs,” in Release Notes for scikit-learn 1.9, pull request #28464, 2025. [Online]. Available: https://scikit-learn.org/stable/whats_new/v1.9.html',
    'A. Zainab, M. A. Khalid, F. U. Khan and A. Khan, “Benchmarking classical and transformer-based models for document sensitivity classification,” arXiv:2608.16928, Aug. 2026. [Online]. Available: https://arxiv.org/abs/2608.16928',
    'A. Zainab, A. Khan, M. A. Khalid and F. U. Khan, “A channel-boosted multi-agent system with iterative consultation for document sensitivity classification,” arXiv:2609.22212, Sep. 2026. [Online]. Available: https://arxiv.org/abs/2609.22212',
    'A. Ivchenko, “CTSCAN: Evaluation leakage in chest CT segmentation and a reproducible patient-disjoint benchmark,” arXiv:2604.15561, Apr. 2026. [Online]. Available: https://arxiv.org/abs/2604.15561',
    'A. A. Ntwali, L. Rück and M. Heckmann, “Detection of personal data in structured datasets using a large language model,” in Proc. Workshop on Large Language Models for Data Protection and Management (LLM-DPM), Berlin, Germany, Jun. 2025, arXiv:2506.22305. [Online]. Available: https://arxiv.org/abs/2506.22305',
    'R. Richie, S. Grover and F. Tsui, “Inter-annotator agreement is not the ceiling of machine learning performance: Evidence from a comprehensive set of simulations,” in Proc. 21st Workshop on Biomedical Language Processing (BioNLP), Dublin, Ireland, 2022, pp. 275–284, doi:10.18653/v1/2022.bionlp-1.26.',
    'L. Telkamp and M. Hulsebos, “Towards contextual sensitive data detection,” arXiv:2512.04120, Dec. 2025. [Online]. Available: https://arxiv.org/abs/2512.04120',
    'D. Antypas, I. Sen, C. Perez-Almendros, J. Camacho-Collados and F. Barbieri, “Sensitive content classification in social media: A holistic resource and evaluation,” in Proc. 9th Workshop on Online Abuse and Harms (WOAH), Aug. 2025, pp. 17–31. [Online]. Available: https://aclanthology.org/2025.woah-1.2/',
    'M. Ojala and G. C. Garriga, “Permutation tests for studying classifier performance,” J. Mach. Learn. Res., vol. 11, no. 62, pp. 1833–1863, 2010. [Online]. Available: https://www.jmlr.org/papers/v11/ojala10a.html',
    'G. C. Cawley and N. L. C. Talbot, “On over-fitting in model selection and subsequent selection bias in performance evaluation,” J. Mach. Learn. Res., vol. 11, no. 70, pp. 2079–2107, 2010. [Online]. Available: https://www.jmlr.org/papers/v11/cawley10a.html',
    'S. Roth, “Which leakage types matter? A quantitative landscape across 2,047 benchmark datasets,” arXiv:2604.04199, Apr. 2026. [Online]. Available: https://arxiv.org/abs/2604.04199',
    'M. Boguslav and K. B. Cohen, “Inter-annotator agreement and the upper limit on machine performance: Evidence from biomedical natural language processing,” Stud. Health Technol. Inform., vol. 245, pp. 298–302, 2017, doi:10.3233/978-1-61499-830-3-298.',
    'J. H. F. James, “Counting on consensus: Selecting the right inter-annotator agreement metric for NLP annotation and evaluation,” in Proc. 15th Lang. Resources and Evaluation Conf. (LREC), 2026, pp. 4434–4446. [Online]. Available: https://aclanthology.org/2026.lrec-1.347/',
    'K. L. Gwet, “Handbook of Inter-Rater Reliability,” 4th ed. Gaithersburg, MD, USA: Advanced Analytics, 2014. [Online]. Available: https://agreestat.com/store.html',
    'M. Ferrari Dacrema, P. Cremonesi and D. Jannach, “Are we really making much progress? A worrying analysis of recent neural recommendation approaches,” in Proc. 13th ACM Conf. Recommender Systems (RecSys), 2019, pp. 101–109, doi:10.1145/3298689.3347058.',
    'K. Musgrave, S. Belongie and S.-N. Lim, “A metric learning reality check,” in Proc. European Conf. Computer Vision (ECCV), 2020, arXiv:2003.08505. [Online]. Available: https://arxiv.org/abs/2003.08505',
    'G. Loiseau, D. Sileo, D. Riquet, M. Meyer and M. Tommasi, “Distilling human-aligned privacy sensitivity assessment from large language models,” arXiv:2603.29497, Mar. 2026. [Online]. Available: https://arxiv.org/abs/2603.29497',
    'W. Antoun, F. Baly and H. Hajj, “AraBERT: Transformer-based model for Arabic language understanding,” in Proc. 12th Int. Conf. Language Resources and Evaluation (LREC), Marseille, France, 2020. [Online]. Available: https://aclanthology.org/2020.osact-1.2/',
    'NLLB Team, M. R. Costa-jussà, J. Cross, O. Çelebi, M. Elbayad, K. Heafield, K. Heffernan, E. Kalbassi, J. Lam, D. Licht, J. Maillard, A. Sun, S. Wang, G. Wenzek, A. Youngblood et al., “No language left behind: Scaling human-centered machine translation,” arXiv:2207.04672, 2022.',
]

# How each entry was checked, for the artifact release. One key per line,
# because the previous shared-line layout silently lost entries to
# duplicate keys when the list was renumbered.
VERIFIED = {
    1: 'industry report - Verizon DBIR 2026, corporate author',
    2: 'vendor documentation - Microsoft Learn, corporate author',
    3: 'crossref 10.3390/asi8020055',
    4: 'crossref 10.1007/978-3-031-68312-1_14',
    5: 'crossref 10.1007/s00521-024-09707-w',
    6: 'arxiv 2108.08483',
    7: 'acl 2024.caldpseudo-1.7',
    8: 'arxiv 2510.07551',
    9: 'crossref 10.18280/ijsse.150602',
    10: 'crossref 10.1016/j.iot.2021.100444',
    11: 'crossref 10.3390/info17070663',
    12: 'crossref 10.1016/j.eswa.2023.119924',
    13: 'arxiv 2008.10863',
    14: 'ieee 10041388',
    15: 'arxiv 2208.06216',
    16: 'acm 10.1145/3417334',
    17: 'crossref 10.1007/978-3-030-99739-7_25 - ATTRIBUTION CORRECTED',
    18: 'arxiv 1907.02956',
    19: 'crossref 10.1007/s10506-023-09383-6 - YEAR CORRECTED 2023->2025',
    20: 'ICLR 2024 proceedings + arxiv 2310.17884 - re-read 2026-09-23; AUTHORS CORRECTED: T. Berg-Kirkpatrick is not an author, R. Shokri was missing. OpenReview link replaced - it now answers scripts with a browser challenge, so no reviewer could check it',
    21: 'arxiv 2502.17041',
    22: 'arxiv 2408.05212',
    23: 'crossref 10.1007/s44443-025-00177-1',
    24: 'arxiv 2601.06366',
    25: 'crossref 10.2307/2529310 - Landis & Koch 1977',
    26: 'crossref 10.1109/ACCESS.2021.3084050 - MCC vs kappa',
    27: 'crossref 10.1016/j.patter.2023.100804',
    28: 'crossref 10.1186/s12864-019-6413-7',
    29: 'arxiv 2411.09539',
    30: 'NAACL-HLT 2019 proceedings',
    31: 'ECML 2004 LNCS 3201',
    32: 'arxiv 2001.10374',
    33: 'arxiv 2606.27559',
    34: 'vendor blog - Meta AI, corporate author',
    35: 'arxiv 2606.04274 - TITLE COMPLETED',
    36: 'scikit-learn 1.9 release notes, quoted verbatim; PR #28464',
    37: 'arxiv 2608.16928 - abstract read 2026-09-21; 4 authors, 5 Aug 2026',
    38: 'arxiv 2609.22212 - abstract read 2026-09-22; same group, 2 Sep 2026',
    39: 'arxiv 2604.15561 - abstract read 2026-09-21; 1 author, 16 Apr 2026',
    40: 'arxiv 2506.22305 - abstract + HTML Table 2 read 2026-09-21',
    41: 'acl 2022.bionlp-1.26 - page read 2026-09-21; pp. 275-284 confirmed',
    42: 'arxiv 2512.04120 - abstract read 2026-09-21; 2 Dec 2025 NOT 2026',
    43: 'acl 2025.woah-1.2 - PDF read 2026-09-21; pp. 17-31; gpt-4o 75.7',
    44: 'jmlr v11 ojala10a - page read 2026-09-21; vol. 11, pp. 1833-1863',
    45: 'jmlr v11 cawley10a - page read 2026-09-22; vol. 11, pp. 2079-2107',
    46: 'arxiv 2604.04199 - abstract read 2026-09-22; single author, 5 Apr 2026',
    47: 'iospress 10.3233/978-1-61499-830-3-298 - read 2026-09-22; MEDINFO 2017',
    48: 'acl 2026.lrec-1.347 - PDF read 2026-09-22; pp. 4434-4446 confirmed',
    49: 'publisher page - 4th edition; AC1 introduced in Gwet (2001). LINK CORRECTED 2026-09-23: /book4/ returns 404 after a site restructure; the 4th ed. is listed on store.html. A 5th ed. (ISBN 978-1-7923-5463-2) now exists; AC1 is unchanged, so the edition our AC1 implementation follows is kept',
    50: 'acm 10.1145/3298689.3347058 - read 2026-09-23; RecSys 2019',
    51: 'arxiv 2003.08505 / ECCV 2020 - read 2026-09-23',
    52: 'arxiv 2603.29497 - abstract read 2026-09-23; 5 authors, 31 Mar 2026',
    53: 'arxiv 2003.00104 / LREC 2020 proceedings',
    54: 'arxiv 2207.04672 - collective authorship, first 15 named then et al.',
}

assert len(REFS) == 54, len(REFS)
assert len(VERIFIED) == 54, len(VERIFIED)
