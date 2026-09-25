# -*- coding: utf-8 -*-
"""The statistical bounds, checked against populations whose answer is known.

Three bugs have been found in these functions, and none of them was visible by
reading:

  * the quantile bound returned the sample's most extreme value when the
    sample could not support the promise. With fifty points from a uniform, a
    1% promise then broke 60% of the time.
  * the precision bound treated "every sampled item above the cut was correct"
    as certainty. The Clopper-Pearson limit for h = n is alpha**(1/n) - 0.10
    at n = 1, not 1.0 - and one lucky message opened the gate, blocking 42
    messages at 40% precision under a 90% promise.
  * the binomial was given the message count on clustered data, overstating
    the sample and overshooting the promise by up to 6.3x.

All three were caught by running the function against a distribution whose
true answer is known and counting how often the promise broke. That is what
this file does, so the next one is caught the same way.

A bound is not a bound until it has been run against something that could
break it.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import filter_system as FS                                   # noqa: E402

DRAWS = 2000
ALPHA = 0.10
SEED = 0


def check_lower(rng):
    """Share BELOW the cut must be <= q, in at least 1 - alpha of draws."""
    ok = True
    print("  lower bound: share below the cut should be <= q")
    for n in (20, 50, 75, 200, 1132):
        for q in (0.01, 0.05, 0.10):
            bad = 0
            for _ in range(DRAWS):
                v = np.sort(rng.rand(n))
                t = FS._lower_bound_quantile(v, q, ALPHA)
                # for U(0,1) the population share below t is t itself
                bad += (max(t, 0.0) > q)
            rate = bad / DRAWS
            good = rate <= ALPHA + 0.02          # a little slack for 2000 draws
            ok &= good
            print("    n=%-5d q=%.2f  broken %5.1f%%  %s"
                  % (n, q, 100 * rate, "ok" if good else "*** TOO OFTEN ***"))
    return ok


def check_upper(rng):
    """NOTE: _upper_bound_quantile is not called by the system. _set_policy
    uses the lower bound and the precision threshold only. These checks are
    kept because the function is correct, but they bind nothing the guarantee
    runs through - crossed_thresholds_withdraw_blocking() below does."""
    """Share ABOVE the cut must be <= q, in at least 1 - alpha of draws."""
    ok = True
    print("  upper bound: share above the cut should be <= q")
    for n in (20, 50, 200, 1132):
        for q in (0.01, 0.05):
            bad = 0
            for _ in range(DRAWS):
                v = np.sort(rng.rand(n))
                t = FS._upper_bound_quantile(v, q, ALPHA)
                bad += ((1.0 - min(t, 1.0)) > q)
            rate = bad / DRAWS
            good = rate <= ALPHA + 0.02
            ok &= good
            print("    n=%-5d q=%.2f  broken %5.1f%%  %s"
                  % (n, q, 100 * rate, "ok" if good else "*** TOO OFTEN ***"))
    return ok


def check_precision(rng):
    """Precision above the cut must be >= the promise, or refuse to cut.

    The population: the top 10% of the risk range is positive with probability
    `true_p`, the rest at a 10% base rate. A bound promising 0.90 must either
    find a cut that delivers it or return infinity."""
    ok = True
    print("  precision bound: promise 0.90, or block nothing")
    print("    %-10s %10s %14s %14s" % ("true prec.", "blocked in",
                                        "promise broken", "verdict"))
    for true_p in (0.98, 0.95, 0.80, 0.50, 0.20):
        blocked, broken = 0, 0
        for _ in range(400):
            n = 300
            risk = rng.rand(n)
            p = np.where(risk > 0.9, true_p, 0.10)
            y = (rng.rand(n) < p).astype(int)
            t = FS._precision_threshold(risk, y, 0.90, ALPHA)
            if np.isinf(t):
                continue
            blocked += 1
            sel = risk >= t
            if sel.sum() and y[sel].mean() < 0.90:
                broken += 1
        # Two conditions, and the second is the one that matters.
        #
        # "Never broke the promise" alone does not test the bound. With the
        # h == n bug reintroduced, the function blocked on a population whose
        # true precision is 0.20 in 95 of 400 draws - and the observed
        # precision of what it blocked still came out above 0.90, because
        # blocking greedily deep into the ranking sweeps in enough positives
        # to look fine. The test passed while the bound was broken.
        #
        # So a population that cannot support the promise must make it REFUSE,
        # not merely avoid being caught.
        rate = blocked / 400.0
        if true_p >= 0.95:
            good = broken == 0 and blocked > 0          # should act
        elif true_p <= 0.50:
            good = broken == 0 and rate <= 0.02         # should refuse
        else:
            good = broken == 0                          # borderline: either
        ok &= good
        print("    %-10.2f %10d %14d %14s"
              % (true_p, blocked, broken, "ok" if good else "*** WRONG ***"))
    return ok


def check_clustering(rng):
    """With n_eff below n, the bound must get MORE conservative, not less."""
    ok = True
    print("  clustering: a smaller effective sample must not loosen the cut")
    for q in (0.01, 0.05):
        loosened = 0
        for _ in range(500):
            v = np.sort(rng.rand(400))
            full = FS._lower_bound_quantile(v, q, ALPHA)
            clustered = FS._lower_bound_quantile(v, q, ALPHA, n_eff=80)
            loosened += (clustered > full + 1e-12)
        good = loosened == 0
        ok &= good
        print("    q=%.2f  loosened in %d of 500  %s"
              % (q, loosened, "ok" if good else "*** LOOSER WHEN CLUSTERED ***"))
    return ok


def crossed_thresholds_withdraw_blocking():
    """When the two thresholds cross, is the block side actually withdrawn?

    The old code set both to the median risk score, which is neither a leak
    bound nor a precision bound. Everything above the middle of the score
    distribution was then blocked with nothing behind the decision. Nothing
    caught it, because the contract checks in the repo tested the leak half
    only - so this asserts the policy directly rather than the rate it happens
    to produce.

    Built so the thresholds MUST cross: a large leak allowance pushes t_allow
    up, and a class this separable puts the certifiable precision cut low."""
    rng = np.random.RandomState(7)
    n = 3000
    y = (rng.rand(n) < 0.45).astype(int)
    texts = ["spam win prize %d" % i if y[i]
             else "meeting notes %d" % i for i in range(n)]
    f = FS.SensitivityFilter(max_leak=0.40, max_false_block=0.40)
    f.fit(texts, y, np.arange(n))
    crossed = not np.isfinite(f.t_block)
    print("  %-56s %s"
          % ("thresholds crossed -> blocking withdrawn",
             "yes" if crossed else "*** NO: t_block %.4f ***" % f.t_block))
    if crossed:
        a, _r = f.actions(texts)
        blocked = int((a == FS.BLOCK).sum())
        print("  %-56s %d" % ("messages blocked once withdrawn", blocked))
        return blocked == 0
    return False



def main():
    rng = np.random.RandomState(SEED)
    print("bounds checked against populations with a known answer, "
          "%d draws each" % DRAWS)
    print("a bound at %.0f%% confidence may break its promise in at most "
          "%.0f%% of draws" % (100 * (1 - ALPHA), 100 * ALPHA))
    print()
    ok = True
    ok &= check_lower(rng)
    print()
    ok &= check_upper(rng)
    print()
    ok &= check_precision(rng)
    print()
    ok &= check_clustering(rng)
    print()
    print("the policy when the two thresholds cross")
    ok &= crossed_thresholds_withdraw_blocking()
    print()
    print("bounds hold" if ok else "A BOUND DOES NOT HOLD")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
