# Arabic extension, run 1 (Kaggle, Tesla T4)
Source: KAGGLE_ARABIC.txt | translation 70 min + 2 x 30 min training

## Translation
NLLB-200-distilled-600M, target arb_Arab (token id 256011)
1382 messages -> 5211 chunks, 0 empty outputs, 92.7% Arabic script
Local QA of enron_arabic.json:
  clean                                    1198  (86.7%)
  cosmetic only (separator run extended)    141  (10.2%)
  GENUINELY DAMAGED                          43  ( 3.1%)
  damaged that are sensitive: 5 of 43 (11.6%) vs 18.1% corpus base rate

## Single-term audit  <-- the critical check, and it PASSED
english  best term 'in'  F1 0.321   trivial 0.306   gap +0.015
arabic   best term 'في'  F1 0.307   trivial 0.306   gap +0.001
=> translation did NOT introduce a lexical shortcut.

## Results (thread-grouped 5-fold, strict labels)
model            F1      MCC     ROC_AUC  PR_AUC
LogReg (EN)      0.3458  0.2346  0.7100   0.3371
LinearSVM (EN)   0.2637  0.1871  0.7128   0.3342
RoBERTa (EN)     0.3916  0.2258  0.6815   0.3448
LogReg (AR)      0.2827  0.1926  0.6778   0.3163
LinearSVM (AR)   0.1863  0.1436  0.6729   0.3052
AraBERT (AR)     0.3100  0.0783  0.5708   0.2325

AraBERT per fold F1 : 0.389, 0.280, 0.280, 0.256, 0.354
AraBERT thresholds  : 0.12, 0.05, 0.44, 0.33, 0.24

## Reproducibility
The English reference re-trained here returned 0.3916 / 0.2258 / 0.6815,
identical to run 2 in an earlier Kaggle session. Kaggle reproduces itself.

## DO NOT PUBLISH THE AraBERT ROW AS-IS
AraBERT's F1 (0.310) is 0.004 above the trivial floor and its MCC (0.078)
and ROC-AUC (0.571) are near chance. But the model card says:
  "It is recommended to apply our preprocessing function before
   training/testing on any dataset."
We did not. LogReg on the SAME Arabic text reaches MCC 0.193 / ROC 0.678,
so the text carries signal - AraBERT's failure is confounded with our
configuration. Re-run with ArabertPreprocessor before reporting.
=> KAGGLE_ARABIC_V2.txt does exactly that.
