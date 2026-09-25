<!-- SUPERSEDED. This is the run from BEFORE the subject-header bug was
found on 22 September. Its fold fingerprint is 50b3daba1a99ae32 and it counts
1,069 threads; the released build is 63e3aea5c3d37629 and 1,103 threads. Every
thread-grouped figure below moved when the corpus was rebuilt, and its "WHAT
CHANGED" section is the origin of a claim that had propagated into the README's
Known Limitations and is false against the released records. Kept because
Section VII-F compares against it. DO NOT QUOTE THESE AS RESULTS. -->

# FINAL RUN - every number in the paper comes from this one session
Kaggle, Tesla T4 | sklearn 1.9.1 | numpy 2.0.2 | scipy 1.16.3 | torch 2.10.0+cu128
transformers 5.0.0 | python 3.12.13 | fold fingerprint 50b3daba1a99ae32

## Dataset
1382 messages | 250 strict / 422 broad | 1069 threads
annotation: both 250 / one 172 / neither 960
  observed agreement 0.8755 | chance 0.6320 | Cohen's kappa 0.6618
trivial floor: strict 0.3064, broad 0.4678

## Leakage audit
duplicates 0 | multi-message threads 173 | mixed-label 55
strongest of 8984 terms: 'in' F1 0.3210 (trivial 0.3064, margin +0.0146)

## Human ceiling, derived from the same agreement counts
One annotator scored as a classifier predicting the other:
  TP 250 | FP 86 | FN 86 | TN 960   (the 172 disagreements split evenly)
  precision 0.7440 | recall 0.7440 | F1 0.7440 | MCC 0.6618
The MCC equals Cohen's kappa exactly - for a two-class problem with symmetric
disagreement the two statistics reduce to the same quantity.
=> the task's ceiling is F1 0.744, not 1.000. RoBERTa's 0.359 is 48% of it.

## STRICT labels, thread-grouped 5-fold
model         F1      P       R      MCC     ROC     PR      mean+-sd        CI95
LinearSVM     0.2685  0.4261  0.1960  0.1919  0.7183  0.3439  0.2658+-0.0760 [0.2065,0.3307]
LogReg        0.3456  0.4076  0.3000  0.2308  0.7170  0.3487  0.3432+-0.0623 [0.2850,0.4043]
roberta-base  0.3587  0.2892  0.4720  0.1821  0.6774  0.3093  0.3583+-0.0506 [0.3086,0.4077]

## BROAD labels, thread-grouped 5-fold
model         F1      P       R       MCC     ROC     PR      mean+-sd        CI95
LinearSVM     0.4591  0.5294  0.4052  0.2687  0.7318  0.5259  0.4588+-0.0410 [0.4082,0.5057]
LogReg        0.5160  0.5154  0.5166  0.3028  0.7286  0.5156  0.5154+-0.0251 [0.4710,0.5601]
roberta-base  0.5451  0.4722  0.6445  0.3063  0.7206  0.5325  0.5431+-0.0349 [0.5031,0.5843]

## Thread-grouping cost (stratified -> grouped, F1)
strict:  LogReg 0.3571 -> 0.3456 (-0.0115) | LinearSVM 0.3002 -> 0.2685 (-0.0317)
broad:   LogReg 0.5239 -> 0.5160 (-0.0079) | LinearSVM 0.4936 -> 0.4591 (-0.0345)

## Paired, identical folds (n=5, so p cannot go below 0.0625)
transformer vs LogReg     strict  4/5 wins  mean +0.0151  p 0.625
transformer vs LinearSVM  strict  5/5 wins  mean +0.0925  p 0.0625
transformer vs LogReg     broad   3/5 wins  mean +0.0276  p 0.3125
transformer vs LinearSVM  broad   5/5 wins  mean +0.0843  p 0.0625

## WHAT CHANGED vs the numbers currently in the paper - read before editing
1. STRICT: RoBERTa now leads ONLY on F1 (0.359 vs 0.346). It loses MCC
   (0.182 vs 0.231), ROC-AUC (0.677 vs 0.717) AND PR-AUC (0.309 vs 0.349).
   The earlier draft said it won PR-AUC; it does not. The trade is explicit
   in P/R: 0.289/0.472 against 0.408/0.300.
2. BROAD: the earlier claim "every fold clears logistic regression" is NOT
   supported. Paired, it is 3 of 5 folds, p = 0.3125. RoBERTa still leads on
   pooled F1 (+0.029), MCC (+0.0035, marginal) and PR-AUC (+0.017), and
   still loses ROC-AUC (0.721 vs 0.729).
3. CONFIDENCE INTERVALS OVERLAP EVERYWHERE. Strict: SVM [0.207,0.331],
   LogReg [0.285,0.404], RoBERTa [0.309,0.408]. Broad: SVM [0.408,0.506],
   LogReg [0.471,0.560], RoBERTa [0.503,0.584]. No method is separable from
   the next. This is the honest headline and it supports the paper's thesis.
4. The only comparisons reaching the attainable floor (p = 0.0625, 5/5 folds)
   are against LinearSVM, the weakest baseline, in both label sets.
