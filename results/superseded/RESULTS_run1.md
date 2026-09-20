# Run 1 - final results (Colab, Tesla T4, 36 min)
Date: 2026-09-15 | Source: Sensitivity_FINAL.ipynb | Berkeley Enron two-annotator subset

## Dataset
1382 messages | 250 sensitive (18.1%) | 1069 threads
Trivial all-positive baseline F1 = 0.307   <-- reference point for every number below

## Annotation (answers supervisor demand #2, "Non-negotiable")
Cohen's kappa = 0.662   observed agreement = 0.876   -> substantial agreement

## Leakage audit
exact duplicates      0
multi-message threads 173
mixed-label threads   55
strongest single word 'in', F1 0.321  (vs trivial 0.307 -> NO keyword shortcut exists)

## Results
model                    F1      Precision  Recall  MCC
roberta-base (grouped)   0.3877  0.3393     0.4520  0.2319
LogReg (ungrouped)       0.3571  0.3761     0.3400  0.2242
LogReg (grouped)         0.3458  0.4157     0.2960  0.2346
LinearSVM (ungrouped)    0.3002  0.3804     0.2480  0.1895
LinearSVM (grouped)      0.2637  0.4211     0.1920  0.1871

RoBERTa per fold : 0.4609, 0.4222, 0.3902, 0.3291, 0.2353
RoBERTa fold-mean: 0.367 +- 0.079
RoBERTa confusion: TP 113  FP 220  FN 137  TN 912
Per-fold thresholds: 0.28, 0.94, 0.81, 0.60, 0.25  (unstable - see caveat)

## Thread-grouping cost (the leakage evidence)
LogReg     0.3571 -> 0.3458  (-0.011)
LinearSVM  0.3002 -> 0.2637  (-0.037)

## CAVEATS - state these in the paper
1. On MCC, RoBERTa (0.2319) does NOT beat LogReg-grouped (0.2346). They are tied.
   F1 favours RoBERTa only because it trades precision for recall.
2. Fold range 0.235-0.461 (SD 0.079) overlaps LogReg. No significant separation.
3. Thresholds tuned on ~36 validation positives -> threshold choice is noisy.
   This is the likely cause of both (1) and (2).
