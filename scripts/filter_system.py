# -*- coding: utf-8 -*-
"""The filter as a system: a message goes in, a decision comes out.

Everything else in this repository measures classifiers. A deployed filter is
not a classifier. It is a policy with three outcomes - let it through, hold it
for a person, block it - and the question it has to answer is not "what is the
F1" but "how much of the traffic can I handle without a human, and what does
the traffic I do handle cost me".

Those are different questions and they have different answers. A model at F1
0.40 sounds useless. The same model, asked to auto-decide only where it is
confident and to escalate the rest, can clear most of the traffic at an error
rate a person would accept. That is the framework the title promises, and this
is it as something you can call.

THE POLICY. Two thresholds on one risk score:

    risk >= t_block     -> BLOCK      (auto)
    risk <= t_allow     -> ALLOW      (auto)
    otherwise           -> ESCALATE   (a person reads it)

THE CONSTRAINT IS THE POINT. In data-loss prevention the two errors are not
worth the same. A sensitive message auto-allowed is a leak. A harmless message
escalated is a minute of someone's time. So the thresholds are not chosen to
maximise accuracy; t_allow is chosen as the largest value at which at most
MAX_LEAK_RATE of the sensitive messages in the validation split would be
auto-allowed, and t_block likewise bounds how many harmless messages are
auto-blocked. Whatever coverage that leaves is the answer, not the target.

WHERE THE THRESHOLDS COME FROM. Validation threads inside the training fold,
the same 30% split used everywhere in this paper. The test fold sees neither
the fit nor the thresholds.

WHAT IS AND IS NOT HERE. The components that run on a CPU - word tf-idf and
character n-grams - are fitted here. The fine-tuned encoder and the
instruction-tuned model are slots: `add_component` takes any scorer, and
`hybrid_framework.py` fills them when a GPU is available. The system is
measured with what it has, and reports which components were present.
"""
import io
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import io_paths                                              # noqa: E402
import strengthen as S                                       # noqa: E402

OUT = io_paths.result_out("RESULTS_FILTER_SYSTEM.json")
FOLDS, SEED, VAL_FRAC = 5, 42, 0.30

# At most this share of genuinely sensitive messages may be auto-allowed, and
# at most this share of harmless ones auto-blocked. Both are policy, not
# findings: a deployment sets them from what a leak and an interruption cost.
MAX_LEAK_RATE = 0.05
MAX_FALSE_BLOCK_RATE = 0.01
# One-sided confidence for the threshold bounds. 0.5 is the point estimate,
# which is what broke the contract; 0.10 is a 90% bound.
CONF_ALPHA = 0.10

ALLOW, ESCALATE, BLOCK = "ALLOW", "ESCALATE", "BLOCK"


def _lower_bound_quantile(sorted_vals, q, alpha, n_eff=None):
    """The largest cut below which at most a q share of the population sits,
    with confidence 1 - alpha.

    If the cut is the k-th smallest of n samples, the share below it is
    Binomial(n, q)-distributed; k is safe while P(Binom(n, q) < k) <= alpha.
    Taking the largest such k gives the most permissive honest threshold.

    When even k = 0 fails - the sample is too small to promise q at this
    confidence - the answer is minus infinity, which auto-allows nothing. The
    first version returned the sample minimum instead, which is not a bound on
    anything: with fifty points from a uniform, the share below the minimum is
    about 1/51, so a promise of 1% was broken 60% of the time. A threshold
    taken from the most extreme point in a sample is exactly the estimate this
    function exists to replace."""
    from scipy.stats import binom
    n = len(sorted_vals)
    if n == 0:
        return float("-inf")
    m = n if n_eff is None else max(1, min(n, int(n_eff)))
    k = 0
    while k < m and binom.cdf(k, m, q) <= alpha:
        k += 1
    # k is an index into the effective sample; map it back to the message-level
    # order statistics proportionally.
    j = int(round(k * n / float(m)))
    return float(sorted_vals[j - 1]) if j > 0 else float("-inf")


def _upper_bound_quantile(sorted_vals, q, alpha, n_eff=None):
    """Mirror of the above for the block side: the smallest cut above which at
    most a q share of the population sits. Plus infinity when the sample
    cannot support the promise, which auto-blocks nothing."""
    from scipy.stats import binom
    n = len(sorted_vals)
    if n == 0:
        return float("inf")
    m = n if n_eff is None else max(1, min(n, int(n_eff)))
    k = 0
    while k < m and binom.cdf(k, m, q) <= alpha:
        k += 1
    j = int(round(k * n / float(m)))
    return float(sorted_vals[n - j]) if j > 0 else float("inf")


class SensitivityFilter(object):
    """Fit on labelled threads, then decide about one message at a time."""

    def __init__(self, max_leak=MAX_LEAK_RATE,
                 max_false_block=MAX_FALSE_BLOCK_RATE,
                 conf_alpha=CONF_ALPHA):
        self.max_leak = max_leak
        self.max_false_block = max_false_block
        self.conf_alpha = conf_alpha
        self.components = []          # (name, vectoriser, model, calibrator)
        self.fusion = None
        self.t_allow = None
        self.t_block = None
        self.extra = []               # (name, scorer) supplied from outside
        self._n_eff_pos = None        # threads, not messages
        self._n_eff_neg = None

    # ---- fitting ---------------------------------------------------------

    def add_component(self, name, scorer):
        """Plug in a scorer this machine cannot fit - an encoder, an LLM.

        `scorer` maps a list of messages to a float per message. It must have
        been produced without seeing the messages it will be asked about."""
        self.extra.append((name, scorer))

    def _calibrate(self, s_va, y_va):
        """Map a component's raw score to a probability, fitted on validation.

        Isotonic by default. It calibrates well and it is a step function:
        whole intervals of input collapse to one output, and with a few
        hundred validation points those steps are wide. calibrator_ties.py
        measures what that costs a rate contract - at the block threshold,
        12% of harmless messages land on a single value, so "at most 5% above
        this cut" has no solution. Override to trade calibration quality for
        resolution."""
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0,
                                 y_max=1.0).fit(s_va, y_va)
        return iso.predict

    def _specs(self):
        return [
            ("word tf-idf",
             TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                             max_features=50000),
             lambda: LogisticRegression(max_iter=2000,
                                        class_weight="balanced")),
            ("character n-grams",
             TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                             sublinear_tf=True, max_features=200000),
             lambda: LogisticRegression(max_iter=3000,
                                        class_weight="balanced")),
        ]

    def fit(self, texts, y, groups):
        """Fit components on the training part, calibrate and set the policy
        on held-out validation threads."""
        y = np.asarray(y)
        groups = np.asarray(groups)
        rng = np.random.RandomState(SEED)
        th = np.array(sorted(set(groups)))
        rng.shuffle(th)
        va_th = set(th[:max(1, int(VAL_FRAC * len(th)))])
        va = np.array([i for i, t in enumerate(groups) if t in va_th])
        tr = np.array([i for i, t in enumerate(groups) if t not in va_th])
        assert len(va) and len(tr), "empty split"

        tr_txt = [texts[i] for i in tr]
        va_txt = [texts[i] for i in va]
        cols_tr, cols_va = [], []
        self.components = []
        for name, vec, make in self._specs():
            Xtr = vec.fit_transform(tr_txt)
            m = make().fit(Xtr, y[tr])
            s_tr = m.predict_proba(Xtr)[:, 1]
            s_va = m.predict_proba(vec.transform(va_txt))[:, 1]
            # calibrated on VALIDATION, so the fusion sees honest probabilities
            cal = self._calibrate(s_va, y[va])
            self.components.append((name, vec, m, cal))
            cols_tr.append(cal(s_tr))
            cols_va.append(cal(s_va))

        for name, scorer in self.extra:
            cols_tr.append(np.asarray(scorer(tr_txt), dtype=float))
            cols_va.append(np.asarray(scorer(va_txt), dtype=float))

        # the fusion is fitted on validation too: it is part of the policy,
        # not part of any component
        self.fusion = LogisticRegression(
            max_iter=1000, class_weight="balanced").fit(
                np.column_stack(cols_va), y[va])
        risk_va = self.fusion.predict_proba(np.column_stack(cols_va))[:, 1]
        va_groups = groups[va]
        self._n_eff_pos = len(set(va_groups[y[va] == 1]))
        self._n_eff_neg = len(set(va_groups[y[va] == 0]))
        self._set_policy(risk_va, y[va])
        return self

    def _set_policy(self, risk, y):
        """t_allow and t_block as BOUNDS, not point estimates.

        The point-estimate version - numpy's quantile of the validation
        positives - delivered a 4.0% leak rate when 1% was requested, and a
        6.3% false-block rate when 1% was requested. A percentile estimated
        from about seventy-five points has a wide interval, and roughly half
        the time the true quantile sits below the estimate; every one of those
        times the contract breaks.

        Each threshold is therefore a one-sided binomial confidence bound: the
        most permissive order statistic still consistent with the promise at
        confidence 1 - conf_alpha. It automates less, which is the price of a
        bound meaning what it says. policy_transfer.py measures both the price
        and whether the contract now holds."""
        pos, neg = np.sort(risk[y == 1]), np.sort(risk[y == 0])
        # The binomial assumes independent draws. These are not: messages in a
        # thread share a topic, an author and often a label, which is the whole
        # premise of the protocol. Counting them as independent overstates the
        # sample and the bound breaks - the block side delivered 2 to 3 times
        # the promised rate while the bound itself was provably correct. So the
        # binomial is given the number of THREADS the sample covers, not the
        # number of messages, while the order statistics stay at message level.
        self.t_allow = (_lower_bound_quantile(pos, self.max_leak,
                                              self.conf_alpha,
                                              n_eff=self._n_eff_pos)
                        if len(pos) else 0.0)
        self.t_block = (_upper_bound_quantile(neg, self.max_false_block,
                                              self.conf_alpha,
                                              n_eff=self._n_eff_neg)
                        if len(neg) else 1.0)
        if self.t_block <= self.t_allow:      # degenerate: escalate nothing
            self.t_block = self.t_allow = float(np.median(risk))

    # ---- deciding --------------------------------------------------------

    def risk(self, texts):
        cols = []
        for _name, vec, m, cal in self.components:
            cols.append(cal(m.predict_proba(vec.transform(texts))[:, 1]))
        for _name, scorer in self.extra:
            cols.append(np.asarray(scorer(texts), dtype=float))
        return self.fusion.predict_proba(np.column_stack(cols))[:, 1]

    def decide(self, text):
        """One message in, one decision out."""
        r = float(self.risk([text])[0])
        if r >= self.t_block:
            action, why = BLOCK, "risk %.3f at or above the block threshold" % r
        elif r <= self.t_allow:
            action, why = ALLOW, "risk %.3f at or below the allow threshold" % r
        else:
            action, why = ESCALATE, "risk %.3f between the thresholds" % r
        return {"action": action, "risk": round(r, 4), "reason": why,
                "thresholds": {"allow": round(self.t_allow, 4),
                               "block": round(self.t_block, 4)},
                "components": [n for n, _v, _m, _c in self.components]
                              + [n for n, _s in self.extra]}

    def actions(self, texts):
        r = self.risk(texts)
        out = np.full(len(r), ESCALATE, dtype=object)
        out[r >= self.t_block] = BLOCK
        out[r <= self.t_allow] = ALLOW
        return out, r


def operating_characteristics(actions, y):
    """What a deployment would actually see."""
    n = len(y)
    auto = actions != ESCALATE
    allowed, blocked = actions == ALLOW, actions == BLOCK
    sens = y == 1
    leaked = int((allowed & sens).sum())
    false_block = int((blocked & ~sens).sum())
    auto_err = leaked + false_block
    return {
        "messages": int(n),
        "auto_handled": int(auto.sum()),
        "auto_share": round(float(auto.mean()), 4),
        "escalated": int((~auto).sum()),
        "escalated_share": round(float((~auto).mean()), 4),
        "auto_allowed": int(allowed.sum()),
        "auto_blocked": int(blocked.sum()),
        "sensitive_auto_allowed": leaked,
        "leak_rate_of_sensitive": round(float(leaked / max(1, sens.sum())), 4),
        "harmless_auto_blocked": false_block,
        "false_block_rate_of_harmless":
            round(float(false_block / max(1, (~sens).sum())), 4),
        "errors_among_auto": auto_err,
        "error_rate_among_auto":
            round(float(auto_err / max(1, auto.sum())), 4),
        "sensitive_in_escalated": int(((~auto) & sens).sum()),
        "escalated_precision":
            round(float(((~auto) & sens).sum() / max(1, (~auto).sum())), 4),
    }


def main():
    df = S.build(dedup=True)
    texts = df.text.tolist()
    y = df.label.values
    g = df.thread_key.values
    print("messages %d | threads %d | sensitive %d (%.1f%%)"
          % (len(df), df.thread_key.nunique(), y.sum(), 100 * y.mean()))
    print("policy: at most %.0f%% of sensitive messages auto-allowed, "
          "%.0f%% of harmless ones auto-blocked"
          % (100 * MAX_LEAK_RATE, 100 * MAX_FALSE_BLOCK_RATE))
    print()

    actions = np.full(len(y), ESCALATE, dtype=object)
    risks = np.zeros(len(y))
    thresholds = []
    for tr, te in GroupKFold(n_splits=FOLDS).split(texts, y, g):
        assert not (set(g[tr]) & set(g[te])), "thread leaked into test"
        f = SensitivityFilter().fit([texts[i] for i in tr], y[tr], g[tr])
        a, r = f.actions([texts[i] for i in te])
        actions[te], risks[te] = a, r
        thresholds.append({"allow": round(f.t_allow, 4),
                           "block": round(f.t_block, 4)})

    oc = operating_characteristics(actions, y)
    print("OUT OF FOLD, %d messages" % oc["messages"])
    print("  handled automatically   %5d  (%.1f%%)"
          % (oc["auto_handled"], 100 * oc["auto_share"]))
    print("    of which allowed      %5d" % oc["auto_allowed"])
    print("    of which blocked      %5d" % oc["auto_blocked"])
    print("  escalated to a person   %5d  (%.1f%%)"
          % (oc["escalated"], 100 * oc["escalated_share"]))
    print()
    print("  what the automatic decisions cost")
    print("    sensitive auto-allowed  %3d  = %.1f%% of all sensitive "
          "(policy allows %.0f%%)"
          % (oc["sensitive_auto_allowed"], 100 * oc["leak_rate_of_sensitive"],
             100 * MAX_LEAK_RATE))
    print("    harmless auto-blocked   %3d  = %.1f%% of all harmless "
          "(policy allows %.0f%%)"
          % (oc["harmless_auto_blocked"],
             100 * oc["false_block_rate_of_harmless"],
             100 * MAX_FALSE_BLOCK_RATE))
    print("    error rate among the automatic decisions  %.1f%%"
          % (100 * oc["error_rate_among_auto"]))
    print()
    print("  what the person gets")
    print("    %d messages, %d of them sensitive (%.1f%%) against a %.1f%% "
          "base rate"
          % (oc["escalated"], oc["sensitive_in_escalated"],
             100 * oc["escalated_precision"], 100 * y.mean()))

    out = {"environment": S.ENV,
           "policy": {"max_leak_rate": MAX_LEAK_RATE,
                      "max_false_block_rate": MAX_FALSE_BLOCK_RATE,
                      "val_frac": VAL_FRAC, "folds": FOLDS},
           "components": ["word tf-idf", "character n-grams"],
           "absent": ["fine-tuned encoder", "instruction-tuned model"],
           "note": "thresholds chosen on validation threads inside each "
                   "training fold; the test fold sees neither the fit nor "
                   "the policy",
           "per_fold_thresholds": thresholds,
           "operating_characteristics": oc}
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print()
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
