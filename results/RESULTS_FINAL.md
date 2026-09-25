# FINAL RUN - every number in the paper comes from this session

GENERATED from results/RESULTS_FINAL.json by scripts/final_md.py. Do not edit
by hand: this file was hand-written until 25 September and had drifted onto the
withdrawn 1,069-thread build, fingerprint 50b3daba1a99ae32, while the JSON
beside it carried the released one.

Tesla T4 | sklearn 1.9.1 | numpy 2.0.2 | torch 2.10.0+cu128
python 3.12.13 | fold fingerprint 63e3aea5c3d37629

## Dataset
1382 messages | 250 strict / 422 broad | 1103 threads
annotation: both 250 / one 172 / neither 960
  observed agreement 0.8755 | Cohen's kappa 0.6618
trivial floor: strict 0.3064, broad 0.4678

## Leakage audit
duplicates 0 | multi-message threads 171 | mixed-label 54
strongest of 8984 terms: 'in' F1 0.321

## Human ceiling
One annotator scored as a classifier predicting the other: F1 0.744
=> the task's ceiling is F1 0.744, not 1.000. The encoder's 0.4099 is 55%
   of it.

## STRICT labels, thread-grouped 5-fold
model         F1      P       R       MCC     ROC     PR      mean+-sd        CI95
LinearSVM     0.2235 0.3704 0.1600 0.1433 0.7143 0.3235 0.2193+-0.0646 [0.1635,0.2841]
LogReg        0.3505 0.4213 0.3000 0.2402 0.7109 0.3219 0.3481+-0.0303 [0.2920,0.4084]
roberta-base  0.4099 0.3350 0.5280 0.2529 0.7104 0.3642 0.3981+-0.0774 [0.3633,0.4592]

## BROAD labels, thread-grouped 5-fold
model         F1      P       R       MCC     ROC     PR      mean+-sd        CI95
LinearSVM     0.4667 0.5335 0.4147 0.2764 0.7234 0.5072 0.4665+-0.0162 [0.4202,0.5117]
LogReg        0.5095 0.5120 0.5071 0.2954 0.7232 0.5012 0.5080+-0.0214 [0.4632,0.5532]
roberta-base  0.5303 0.4290 0.6943 0.2654 0.7121 0.5352 0.5264+-0.0326 [0.4918,0.5690]

## Thread-grouping cost (stratified -> grouped, F1)
strict:  LogReg 0.3571 -> 0.3505 (-0.0066) | LinearSVM 0.3002 -> 0.2235 (-0.0767)
broad:   LogReg 0.5239 -> 0.5095 (-0.0144) | LinearSVM 0.4936 -> 0.4667 (-0.0269)

## Paired, identical folds (n=5, so p cannot go below 0.0625)
transformer vs LinearSVM broad           5/5 wins  mean +0.0599  p 0.0625
transformer vs LinearSVM strict          5/5 wins  mean +0.1787  p 0.0625
transformer vs LogReg broad              3/5 wins  mean +0.0184  p 0.3125
transformer vs LogReg strict             4/5 wins  mean +0.0499  p 0.125

## What this run says
Confidence intervals overlap everywhere; no method is separable from the next.
That is the honest headline and it is what the paper argues from. Quote every
figure against both bounds - the trivial floor below and the annotator
reference of F1 0.744 above.
