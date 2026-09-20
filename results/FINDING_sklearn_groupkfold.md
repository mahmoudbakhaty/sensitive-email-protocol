# Why the same code gave two different numbers

## The observation
LogReg, thread-grouped 5-fold, strict labels, identical data and seed:

    scikit-learn 1.6.1 (Kaggle stock)   F1 0.3220  MCC 0.1985
    scikit-learn 1.8.0 (local machine)  F1 0.3220  MCC 0.1985
    scikit-learn 1.9.1 (pip at runtime) F1 0.3458  MCC 0.2346

## The cause, from the scikit-learn 1.9.0 release notes
> "GroupKFold now uses `stable` sorting when doing the group distribution.
>  This ensures that the splits are consistent across runs."
> -- PR #28464, by marikabergengren and Adrin Jalali

So 1.9 is the FIXED version. The older versions used unstable sorting, which
the maintainers changed precisely because it made splits inconsistent.

## The mechanism, measured not inferred
Fold-assignment fingerprint (md5 of the concatenated test-index lists):

    1.6.1  3556420f17e7e4fa
    1.8.0  3556420f17e7e4fa   <- two independent environments agree
    1.9.1  50b3daba1a99ae32   <- different partition

Fold sizes are identical (277/277/276/276/276); the MEMBERSHIP differs.

## What this means for the paper
1. Tables III-VI come from a 1.9.1 session, i.e. the corrected splitter.
   They stand. The paper pins scikit-learn==1.9.1.
2. Section VII-E must be rewritten. Its current story - "two hosted platforms
   agreed, a local machine did not" - is wrong on both halves. The true story
   is a library version boundary, and it is a better illustration of the
   paper's own argument: a grouped-CV result depended on an undocumented
   sorting detail that the library authors later changed.
3. The first addendum run (unpinned, 1.6.1) produced confidence intervals on
   a DIFFERENT partition than the point estimates in Table III, and its
   paired test compared folds that were not the same folds. Both are void.
   The pinned re-run replaces them.

## Method note worth keeping
Neither reasoning nor version numbers settled this. What settled it was
hashing the fold assignment and comparing. Any future claim that two runs are
comparable should carry that fingerprint.
