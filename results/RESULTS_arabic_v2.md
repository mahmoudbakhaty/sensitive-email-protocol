# Arabic extension, run 2 - AraBERT with its documented preprocessing
Kaggle T4 | translation reused from run 1 (NLLB-200, arb_Arab, 5211 chunks)

## The question this run answered
Run 1 fed raw translated text to AraBERT and got F1 0.310 against a trivial
floor of 0.306, MCC 0.078, ROC-AUC 0.571 - effectively no learning. The model
card says "It is recommended to apply our preprocessing function before
training/testing on any dataset." We had not. So: did AraBERT fail, or did we
use it wrongly?

Answer: partly the latter. Preprocessing helps, and helps the transformer
specifically.

## AraBERT, raw vs preprocessed (identical folds, identical protocol)
              F1      MCC     ROC_AUC  PR_AUC
raw           0.3100  0.0783  0.5708   0.2325
preprocessed  0.3289  0.1210  0.6384   0.2693
delta         +0.019  +0.043  +0.068   +0.037

## Classical baselines, raw vs preprocessed
LogReg    raw   0.2827  0.1926  0.6778  0.3163
LogReg    prep  0.2963  0.2130  0.6781  0.3280
LinearSVM raw   0.1863  0.1436  0.6729  0.3052
LinearSVM prep  0.2032  0.1797  0.6734  0.3146

ROC-AUC for the classical models is UNCHANGED to three decimals (0.678, 0.673)
while AraBERT's rises 0.068. That asymmetry is the interpretable result:
preprocessing normalises spacing, diacritics and URL/e-mail forms, which a
subword tokeniser is sensitive to and a bag-of-words model is largely blind to.

## Single-term audit - unchanged by preprocessing
arabic_raw           best term 'في'  F1 0.307  trivial 0.306
arabic_preprocessed  best term 'في'  F1 0.307  trivial 0.306
Translation introduces no lexical shortcut in either version.

## English reference (carried from the single final run, same platform)
LogReg (EN)     0.3458  0.2346  0.7100  0.3371
LinearSVM (EN)  0.2637  0.1871  0.7128  0.3342
RoBERTa (EN)    0.3916  0.2258  0.6815  0.3448

## What the section should say
1. The protocol transfers. Thread grouping, duplicate control, the trivial
   floor and the single-term audit all behave the same in Arabic.
2. Performance does not. Every Arabic figure sits below its English
   counterpart.
3. Even preprocessed, AraBERT loses MCC, ROC-AUC and PR-AUC to Arabic
   logistic regression - the SAME pattern as RoBERTa against English logistic
   regression. The encoder disadvantage is not a translation artefact.
4. Confounds that remain and cannot be separated here: translationese,
   AraBERT's pretraining corpus vs RoBERTa's, and the 3.1% of messages the
   translation damaged.

## A correction worth recording
Before this run, a CPU experiment on a local machine showed preprocessing
barely moving the classical scores, and that was used to argue preprocessing
probably was not the explanation. Wrong twice over: the classical models being
insensitive says nothing about the transformer, and the local numbers moved in
the OPPOSITE direction to Kaggle's (LogReg 0.281->0.267 locally, 0.283->0.296
on Kaggle) because of the scikit-learn version difference. Only the
single-environment run is evidence.
