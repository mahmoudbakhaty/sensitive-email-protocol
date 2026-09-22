# -*- coding: utf-8 -*-
"""Exercise the model-call path itself, with a tiny model on CPU.

The Kaggle run failed inside Judge.score with

    TypeError: embedding(): argument 'indices' (position 2) must be Tensor,
               not BatchEncoding

because apply_chat_template returns a BatchEncoding on some transformers
versions and a bare tensor on others. The harness test could not have caught
it: it stubs the judge out. This test does not stub it. It loads a tiny real
causal LM, builds a real chat prompt, and reads a real probability off real
logits - everything the 7B run does, at a size that fits on a laptop.
"""
import sys

import llm_protocol as L


class TinyJudge(L.Judge):
    """The real Judge, pointed at a model small enough to run here."""

    def __init__(self, model_id):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.chat_template is None:
            # Enough of a template to exercise the same code path.
            self.tok.chat_template = (
                "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n"
                "{% endfor %}{% if add_generation_prompt %}assistant: "
                "{% endif %}")
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
        self.calls = 0
        self.yes = self._variants(["Yes", " Yes", "yes", " yes"])
        self.no = self._variants(["No", " No", "no", " no"])
        assert self.yes and self.no, "answer tokens not found"


def main():
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    print("loading %s ..." % model_id, flush=True)
    try:
        judge = TinyJudge(model_id)
    except Exception as exc:
        print("could not load a tiny model (%s). Network blocked?" % exc)
        return 2
    print("loaded.")

    shots = [("Budget review", "The quarterly budget is attached.", 0),
             ("Re: counsel", "Legal advises we do not disclose this.", 1)]

    for name, sh in (("zero-shot", []), ("few-shot", shots)):
        v = judge.score(sh, "Quick question",
                        "Are you free for a call tomorrow about the filing?")
        assert isinstance(v, float), "score is not a float: %r" % (v,)
        assert 0.0 <= v <= 1.0, "score out of range: %r" % v
        print("  %-10s P(Yes) = %.4f" % (name, v))

    # Long input must not blow up the template or the truncation.
    long_v = judge.score(shots, "Long", "word " * 4000)
    assert 0.0 <= long_v <= 1.0
    print("  long input P(Yes) = %.4f" % long_v)

    print()
    print("model-call path verified end to end on a real model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
