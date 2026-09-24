# -*- coding: utf-8 -*-
"""Draw a blind sample and build the instrument for annotating it.

The open item this release cannot close by computation: our labels come from
the Enron subset's genre categories, mapped by us onto the paper's Section III
definition, and nobody has ever checked a message against that definition
directly. mapping_sensitivity.py bounds how much the mapping could matter. It
cannot tell you whether the mapping is right.

This builds what is needed to find out, so that doing it costs an hour rather
than a project.

FOUR THINGS MAKE THE RESULT WORTH HAVING, AND ALL FOUR ARE STRUCTURAL:

  * The annotator never sees our label. It is written to a separate file that
    the instrument does not read and does not contain.
  * The sample is stratified on our label, so both cells are estimated with
    similar precision instead of the negative cell swamping the positive one.
    Prevalence is weighted back in when scoring.
  * At most one message per thread. Two messages from one thread are close to
    one judgement made twice, and would make the sample look larger than it is.
  * A REPEAT BLOCK, and this one was added after testing the instrument rather
    than before. Some messages appear twice, far apart, under different ids.
    Without them the score is uninterpretable: an annotator who is 90% self
    consistent and a mapping that is genuinely blind to a tenth of the
    definition produce the SAME recall, and nothing in the data separates them.
    The repeats measure the annotator's own noise floor, so the disagreements
    can be read against it instead of against an assumption.
  * THE LABEL IS DERIVED, NOT ASKED. Section III's own test is operational -
    "an annotator must be unable to justify the label by quoting a term" - so
    the instrument asks the three questions the definition is made of and
    computes the label from them. Asking "is this sensitive?" directly would
    collect an intuition and call it a definition.

The three questions, straight from Section III:

  Q1  Would disclosure outside the intended audience cause organisational
      harm?                                                        (the core)
  Q2  Can you justify that by quoting a term from the message?
                                      (if yes, it is NOT context-dependent)
  Q3  Does it carry an account number, national identifier or credential?
                                              (if yes, out of scope [3]-[9])

  sensitive  <=>  Q1 yes  AND  Q2 no  AND  Q3 no

Q2's quoted term is recorded. A disagreement with a term attached is one a
reader can adjudicate; a disagreement without one is an assertion.

    python scripts/make_annotation_sample.py [--n 100] [--repeat 20]

Writes annotation/sample.json, key.json and annotate.html into annotation/.
Open the HTML in any browser - it is self-contained and makes no network
requests. Do not open key.json until the annotation is exported.
"""
import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import strengthen as S                                       # noqa: E402

OUTDIR = os.path.join(REPO, "annotation")

CRITERIA = [
    {"id": "harm",
     "q": "Would disclosure outside its intended audience cause "
          "organisational harm?",
     "help": "Section III's core test. Harm to the organisation, not "
             "embarrassment to an individual. If you cannot see a route to "
             "harm, answer no.",
     "options": ["yes", "no", "unsure"]},
    {"id": "quotable",
     "q": "Can you justify that judgement by quoting a term from the message?",
     "help": "Section III: “an annotator must be unable to justify the "
             "label by quoting a term.” If one word or phrase carries the "
             "judgement on its own, the message is not context-dependent - it "
             "is what a keyword list already catches. Answer only if you said "
             "yes above.",
     "options": ["yes", "no"],
     "needs_quote_when": "yes",
     "only_if": {"harm": "yes"}},
    {"id": "identifier",
     "q": "Does it carry an account number, national identifier or "
          "credential?",
     "help": "Out of scope regardless of content - pattern matching and NER "
             "handle these, and including them would inflate any figure with "
             "cases the existing control already catches.",
     "options": ["yes", "no"]},
]


def draw(df, n, seed):
    """Stratified on our label, one message per thread, shuffled."""
    rng = np.random.RandomState(seed)
    idx_by_thread = {}
    for i, t in enumerate(df.thread_key.values):
        idx_by_thread.setdefault(t, []).append(i)
    # one per thread, chosen by the seed rather than by position, so a thread's
    # first message is not systematically the one that gets judged
    one = np.array(sorted(rng.choice(v) for v in idx_by_thread.values()))
    y = df.label.values[one]
    pos, neg = one[y == 1], one[y == 0]
    half = n // 2
    if len(pos) < half:
        raise SystemExit("only %d positive threads; ask for n <= %d"
                         % (len(pos), 2 * len(pos)))
    pick = np.concatenate([rng.choice(pos, half, replace=False),
                           rng.choice(neg, n - half, replace=False)])
    rng.shuffle(pick)
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=20,
                    help="messages shown twice, to measure the noise floor")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = S.build(dedup=True)
    pick = draw(df, a.n, a.seed)

    sample, key = [], {}
    for rank, i in enumerate(pick):
        text = df.text.values[i]
        aid = hashlib.sha256(("%d|%s" % (a.seed, text[:400]))
                             .encode("utf-8")).hexdigest()[:12]
        sample.append({"aid": aid, "n": rank + 1, "text": text})
        lab = int(df.label.values[i])
        key[aid] = {"our_label": lab,
                    "stratum": "positive" if lab else "negative",
                    "thread_key": str(df.thread_key.values[i]),
                    "repeat_of": None}

    sample, key = add_repeats(sample, key, df, pick, a.repeat, a.seed)

    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
    sp = os.path.join(OUTDIR, "sample.json")
    kp = os.path.join(OUTDIR, "key.json")
    hp = os.path.join(OUTDIR, "annotate.html")

    json.dump({"seed": a.seed, "criteria": CRITERIA, "items": sample},
              io.open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"seed": a.seed, "n": len(sample),
               "corpus_positive_rate": round(float(df.label.mean()), 4),
               "key": key},
              io.open(kp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    io.open(hp, "w", encoding="utf-8").write(page(sample, a.seed))

    first = {k: v for k, v in key.items() if v["repeat_of"] is None}
    npos = sum(1 for v in first.values() if v["our_label"])
    nrep = len(key) - len(first)
    print("sample drawn: %d distinct messages, %d our-positive / %d "
          "our-negative" % (len(first), npos, len(first) - npos))
    print("  one message per thread, stratified, shuffled, seed %d" % a.seed)
    print("  plus %d repeats shown a second time, to measure your own "
          "consistency" % nrep)
    print("  %d judgements in total" % len(sample))
    print()
    print("  %s" % hp)
    print("      open this in a browser and work through it. Self-contained,")
    print("      no network, progress is kept if you close the tab.")
    print("  %s" % sp)
    print("      the same items as data. Carries no labels.")
    print("  %s" % kp)
    print("      OUR labels. Do not open this until you have exported.")
    print()
    print("  then: python scripts/score_annotation.py "
          "annotation/annotations.json")
    return 0


def add_repeats(sample, key, df, pick, k, seed, min_gap_frac=0.33):
    """Show some messages twice, at a GUARANTEED distance, under a new id.

    The first version drew sources from the first third and shuffled only the
    tail, which guarantees nothing: the sole constraint was "original in the
    first third, repeat somewhere after it", so a repeat could land one
    position later. At the shipped seed the nearest pair happened to be 28
    apart and the package's own test passed; at seed 20 it is 3 and at seed 71
    it is 2, and about nine seeds in ten fail that test. A repeat the annotator
    remembers measures their memory rather than their consistency, and the
    noise floor it produces is too low - which makes every rate read against it
    look more significant than it is.

    Placement is constructive now. Walking the sample in order, a repeat
    becomes eligible only once `gap` further items have been emitted, and
    eligible repeats are released in a shuffled order so the pattern is not
    predictable. The separation is asserted before the list is returned."""
    if k <= 0:
        return sample, key
    rng = np.random.RandomState(seed + 1)
    n = len(sample)
    gap = max(1, int(min_gap_frac * n))
    src = set(int(x) for x in rng.choice(np.arange(n // 2),
                                         size=min(k, n // 2), replace=False))

    out, pending = [], []
    for i, item in enumerate(sample):
        out.append(item)
        if i in src:
            aid = hashlib.sha256(("%d|repeat|%s" % (seed, item["aid"]))
                                 .encode("utf-8")).hexdigest()[:12]
            key[aid] = dict(key[item["aid"]])
            key[aid]["repeat_of"] = item["aid"]
            pending.append((len(out) - 1 + gap,
                            {"aid": aid, "n": 0, "text": item["text"]}))
        ready = [p for p in pending if p[0] <= len(out)]
        rng.shuffle(ready)
        for p in ready:
            out.append(p[1])
            pending.remove(p)
    for _earliest, rep in pending:          # distance already exceeded
        out.append(rep)

    where = {it["aid"]: i for i, it in enumerate(out)}
    gaps = [where[a] - where[key[a]["repeat_of"]]
            for a in key if key[a].get("repeat_of")]
    assert gaps and min(gaps) >= gap,         "repeat separation not met: %d < %d" % (min(gaps), gap)
    for rank, it in enumerate(out):
        it["n"] = rank + 1
    return out, key


# ---- the instrument -------------------------------------------------------

def page(items, seed):
    data = json.dumps(items, ensure_ascii=False)
    crit = json.dumps(CRITERIA, ensure_ascii=False)
    return TEMPLATE.replace("/*DATA*/", data).replace(
        "/*CRIT*/", crit).replace("/*SEED*/", str(seed))


TEMPLATE = u"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Section III Annotation</title>
<style>
:root{
  --bg:#f6f7f9; --card:#ffffff; --ink:#15181d; --ink2:#4a5261; --ink3:#828b9c;
  --line:#e2e6ec; --accent:#2f5fd0; --yes:#1f7a4d; --no:#9a3434; --warn:#8a6d1f;
  --chip:#eef1f6;
}
:root:not([data-theme="light"]){ }
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#14161a; --card:#1c1f25; --ink:#e9edf3; --ink2:#b3bcca; --ink3:#7f8899;
    --line:#2b3039; --accent:#7ba2f0; --yes:#5fc38c; --no:#e3807f;
    --warn:#d8bd6a; --chip:#262b33;
  }
}
:root[data-theme="dark"]{
  --bg:#14161a; --card:#1c1f25; --ink:#e9edf3; --ink2:#b3bcca; --ink3:#7f8899;
  --line:#2b3039; --accent:#7ba2f0; --yes:#5fc38c; --no:#e3807f;
  --warn:#d8bd6a; --chip:#262b33;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:24px 16px 96px}
header{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
  margin-bottom:6px}
h1{font-size:19px;margin:0;letter-spacing:-.01em}
.sub{color:var(--ink3);font-size:13px}
.bar{height:5px;background:var(--chip);border-radius:3px;overflow:hidden;
  margin:14px 0 20px}
.bar i{display:block;height:100%;background:var(--accent);width:0;
  transition:width .2s}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:18px;margin-bottom:16px}
.msg{white-space:pre-wrap;font:13px/1.65 ui-monospace,SFMono-Regular,
  Menlo,Consolas,monospace;max-height:44vh;overflow:auto;color:var(--ink2);
  background:var(--bg);border:1px solid var(--line);border-radius:8px;
  padding:14px}
.q{margin-top:18px;padding-top:16px;border-top:1px solid var(--line)}
.q:first-child{border-top:0;padding-top:0;margin-top:0}
.qt{font-weight:600;margin-bottom:4px}
.qh{color:var(--ink3);font-size:13px;margin-bottom:10px}
.opts{display:flex;gap:8px;flex-wrap:wrap}
button.opt{appearance:none;border:1px solid var(--line);background:var(--card);
  color:var(--ink);border-radius:8px;padding:7px 16px;font-size:14px;
  cursor:pointer;font-family:inherit}
button.opt:hover{border-color:var(--accent)}
button.opt[aria-pressed="true"]{border-color:var(--accent);
  background:var(--accent);color:#fff}
input.quote{width:100%;margin-top:10px;padding:9px 11px;border-radius:8px;
  border:1px solid var(--line);background:var(--bg);color:var(--ink);
  font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;font-family:inherit}
.verdict{margin-top:18px;padding:12px 14px;border-radius:8px;
  background:var(--chip);font-size:14px}
.verdict b{font-size:15px}
.nav{position:fixed;left:0;right:0;bottom:0;background:var(--card);
  border-top:1px solid var(--line);padding:10px 16px}
.navin{max-width:880px;margin:0 auto;display:flex;gap:10px;
  align-items:center;flex-wrap:wrap}
.spacer{flex:1}
button.nb{appearance:none;border:1px solid var(--line);background:var(--card);
  color:var(--ink);border-radius:8px;padding:8px 16px;font-size:14px;
  cursor:pointer;font-family:inherit}
button.nb.primary{background:var(--accent);border-color:var(--accent);
  color:#fff}
button.nb:disabled{opacity:.45;cursor:default}
.hint{color:var(--ink3);font-size:12px}
.dim{color:var(--ink3)}
.skip{opacity:.4;pointer-events:none}
@media(max-width:560px){.wrap{padding:16px}.navin{gap:8px}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Section III annotation</h1>
    <span class="sub" id="pos"></span>
    <span class="spacer"></span>
    <span class="sub" id="saved"></span>
  </header>
  <div class="bar"><i id="prog"></i></div>
  <div class="card"><div class="msg" id="msg"></div></div>
  <div class="card" id="qs"></div>
  <div class="card"><div class="verdict" id="verdict"></div></div>
</div>
<div class="nav"><div class="navin">
  <button class="nb" id="prev">&larr; Previous</button>
  <button class="nb primary" id="next">Next &rarr;</button>
  <span class="hint">keys: 1 / 2 / 3 to answer &middot; &larr; &rarr; to move</span>
  <span class="spacer"></span>
  <button class="nb" id="export">Export annotations</button>
</div></div>
<script>
const ITEMS = /*DATA*/;
const CRIT  = /*CRIT*/;
const SEED  = /*SEED*/;
const KEY = "s3annot-" + SEED + "-" + ITEMS.length + "-"
            + (ITEMS[0] ? ITEMS[0].aid : "");
let A = {}, i = 0;
try { A = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { A = {}; }

function save(){
  try { localStorage.setItem(KEY, JSON.stringify(A));
        document.getElementById("saved").textContent = "saved"; }
  catch(e){ document.getElementById("saved").textContent =
        "not saved in this browser \\u2014 export before closing"; }
}
function cur(){ return ITEMS[i]; }
function ans(){ return A[cur().aid] || (A[cur().aid] = {}); }

function derive(a){
  if (a.identifier === "yes") return ["out of scope",
      "carries an identifier \\u2014 Section III excludes it"];
  if (a.harm === "no") return ["not sensitive", "no route to organisational harm"];
  if (a.harm === "unsure") return ["unsure", "harm judgement not settled"];
  if (a.identifier !== "no") return ["incomplete",
      "answer the identifier question - it decides whether Section III covers "
      + "this message at all"];
  if (a.harm === "yes" && a.quotable === "yes") return ["not sensitive",
      "the judgement is carried by a quoted term, so it is not context-dependent"];
  if (a.harm === "yes" && a.quotable === "no") return ["SENSITIVE",
      "harm on disclosure, and no single term carries the judgement"];
  return ["incomplete", "answer the questions above"];
}

function render(){
  const it = cur(), a = ans();
  document.getElementById("msg").textContent = it.text;
  document.getElementById("pos").textContent =
    "message " + it.n + " of " + ITEMS.length;
  document.getElementById("prog").style.width =
    (100 * Object.keys(A).filter(k => derive(A[k])[0] !== "incomplete").length
     / ITEMS.length) + "%";

  const box = document.getElementById("qs");
  box.innerHTML = "";
  CRIT.forEach(c => {
    const skip = c.only_if && Object.keys(c.only_if)
      .some(k => a[k] !== c.only_if[k]);
    const d = document.createElement("div");
    d.className = "q" + (skip ? " skip" : "");
    d.innerHTML = '<div class="qt">' + c.q + '</div>' +
                  '<div class="qh">' + c.help + '</div>';
    const o = document.createElement("div");
    o.className = "opts";
    c.options.forEach(v => {
      const b = document.createElement("button");
      b.className = "opt"; b.textContent = v; b.type = "button";
      b.setAttribute("aria-pressed", a[c.id] === v ? "true" : "false");
      b.onclick = () => { a[c.id] = v; save(); render(); };
      o.appendChild(b);
    });
    d.appendChild(o);
    if (c.needs_quote_when && a[c.id] === c.needs_quote_when){
      const q = document.createElement("input");
      q.className = "quote"; q.placeholder = "quote the term \\u2014 required";
      q.value = a.quote || "";
      q.oninput = () => { a.quote = q.value; save(); };
      d.appendChild(q);
    }
    box.appendChild(d);
  });

  const v = derive(a);
  document.getElementById("verdict").innerHTML =
    '<b>' + v[0] + '</b> <span class="dim">\\u2014 ' + v[1] + '</span>';
  document.getElementById("prev").disabled = (i === 0);
  document.getElementById("next").disabled = (i === ITEMS.length - 1);
}

document.getElementById("prev").onclick = () => { if(i>0){ i--; render(); } };
document.getElementById("next").onclick = () => {
  if(i < ITEMS.length-1){ i++; render(); } };
document.getElementById("export").onclick = () => {
  const out = { seed: SEED, exported: new Date().toISOString(),
                annotations: {} };
  Object.keys(A).forEach(k => {
    const v = derive(A[k]);
    out.annotations[k] = Object.assign({}, A[k],
      { derived: v[0], reason: v[1] });
  });
  const b = new Blob([JSON.stringify(out, null, 1)],
                     {type:"application/json"});
  const u = URL.createObjectURL(b), el = document.createElement("a");
  el.href = u; el.download = "annotations.json"; el.click();
  URL.revokeObjectURL(u);
};
document.onkeydown = e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "ArrowRight") document.getElementById("next").click();
  if (e.key === "ArrowLeft")  document.getElementById("prev").click();
  const a = ans();
  if (["1","2","3"].includes(e.key)){
    const first = CRIT.find(c => !(c.only_if && Object.keys(c.only_if)
      .some(k => a[k] !== c.only_if[k])) && a[c.id] === undefined);
    if (first && first.options[+e.key - 1]){
      a[first.id] = first.options[+e.key - 1]; save(); render();
    }
  }
};
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
