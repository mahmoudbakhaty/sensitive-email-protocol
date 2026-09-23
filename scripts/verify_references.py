# -*- coding: utf-8 -*-
"""Resolve every reference and compare what comes back with what is printed.

The paper's artifact section promises a log of how each reference was
resolved. This is the script behind that log, so a reviewer can re-run it
rather than take the log's word.

Counting identifiers is not enough. An earlier round confirmed that all 54
entries carried a DOI, an arXiv id or a URL, and that round still left two
defects standing: one entry named an author who is not on the paper, and one
pointed at a page that had started returning 404. Both were found only by
fetching the target and reading the metadata that came back.

What is checked, per entry:
  * DOIs against the Crossref registry - registration, title, author families.
  * arXiv ids against the abstract page's citation_* meta tags. The
    export.arxiv.org API is unreachable from some networks and answers others
    with 406, so the abs page is the portable source for the same metadata.
  * Plain URLs by fetching them and looking for the printed title in the body,
    which catches a soft 404 - a "page not found" served with status 200.

Three kinds of false positive are expected and are NOT defects:
  * Crossref splits "James Van Guilder" as given="James Van", family="Guilder".
  * IOS Press stores whole names in the family field.
  * An "et al." list is shorter than the registry's on purpose.
Each is reported with its own tag so it is not mistaken for a real mismatch.

Two targets cannot be checked by any script and are verified by hand:
OpenReview now answers automated requests with a browser challenge, and
ai.meta.com rejects non-browser clients with 400. Where a link like that is
the only identifier, prefer one that a reviewer's script can also resolve.
"""
import io
import json
import os
import re
import sys
import time
import unicodedata
import html
import urllib.request as UR

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paper_refs_v2 as R                                    # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results",
                   "RESULTS_reference_resolution.json")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537"
                    ".36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*"}
SOFT404 = ("page not found", "404 not found", "not be found",
           "no longer available", "does not exist")
# these hosts refuse scripted clients; the entries are verified by hand
BOT_WALLED = ("openreview.net", "ai.meta.com")
LQUO, RQUO = u"“", u"”"


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return re.sub(r"[^a-z]", "",
                  "".join(c for c in s if not unicodedata.combining(c)).lower())


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def title_of(ref):
    m = re.search(LQUO + "(.+?),?" + RQUO, ref, re.S)
    return m.group(1) if m else ""


def surnames_of(ref):
    head = re.sub(r"\bet al\.?", "", ref.split(LQUO)[0]).replace(" and ", ", ")
    out = []
    for part in head.split(","):
        toks = [t for t in part.strip().strip(".").split()
                if not re.fullmatch(r"[A-Z][\w.-]*\.", t)]
        if toks:
            out.append(fold(" ".join(toks)))
    return [x for x in out if x]


def targets_of(ref):
    t = []
    for d in re.findall(r"doi:(10\.\S+?)(?:[.,]?\s|$)", ref):
        t.append(("doi", "https://doi.org/" + d.rstrip(".,")))
    for a in re.findall(r"arXiv:(\d{4}\.\d{4,5})", ref, re.I):
        t.append(("arxiv", "https://arxiv.org/abs/" + a))
    for u in re.findall(r"https?://[^\s'\"]+", ref):
        t.append(("url", u.rstrip(".,")))
    return t


def get(url, tries=3):
    last = None
    for k in range(tries):
        try:
            with UR.urlopen(UR.Request(url, headers=UA), timeout=45) as f:
                return f.getcode(), f.read().decode("utf-8", "replace")
        except Exception as e:                                # noqa: BLE001
            last = e
            time.sleep(3 * (k + 1))
    raise last


def meta(doc, name):
    return [html.unescape(v) for v in re.findall(
        r'<meta[^>]+name="%s"[^>]+content="([^"]*)"' % name, doc)]


def titles_agree(mine, got):
    return bool(mine) and bool(got) and (
        norm(mine)[:45] in norm(got) or norm(got)[:45] in norm(mine))


def compare_authors(mine, got):
    """Returns (verdict, extra, missing).

    Registry quirks are named rather than reported as defects."""
    if not got:
        return "no author data", [], []
    extra = [a for a in mine if a not in got]
    missing = [a for a in got if a not in mine]
    if not extra and not missing:
        return "match", [], []
    if extra and all(any(a in g or g in a for g in got) for a in extra):
        return "registry name-field quirk", extra, missing
    if not extra and len(mine) < len(got):
        return "et al. by design", [], missing
    return "MISMATCH", extra, missing


def check_one(n, ref):
    mine_t, mine_a = title_of(ref), surnames_of(ref)
    rec = {"n": n, "printed_title": mine_t, "verified_note": R.VERIFIED[n],
           "checks": []}
    for kind, url in targets_of(ref):
        c = {"kind": kind, "url": url}
        if any(h in url for h in BOT_WALLED):
            c.update(status="bot-walled",
                     note="verified by hand; this host refuses scripts")
            rec["checks"].append(c)
            continue
        try:
            if kind == "doi":
                code, body = get("https://api.crossref.org/works/"
                                 + url.split("doi.org/", 1)[1])
                m = json.loads(body)["message"]
                got_t = (m.get("title") or [""])[0]
                got_a = [fold(a.get("family", "")) for a in m.get("author", [])]
            elif kind == "arxiv":
                code, body = get(url)
                got_t = (meta(body, "citation_title") or [""])[0]
                got_a = [fold(a.split(",")[0])
                         for a in meta(body, "citation_author")]
                time.sleep(1.0)
            else:
                code, body = get(url)
                flat = html.unescape(re.sub(r"<[^>]+>", " ", body)).lower()
                c.update(status=code,
                         soft_404=any(p in flat[:6000] for p in SOFT404),
                         title_in_page=norm(mine_t)[:40] in norm(flat))
                rec["checks"].append(c)
                continue
            verdict, extra, missing = compare_authors(
                mine_a, [g for g in got_a if g])
            c.update(status=code, registry_title=got_t,
                     title_match=titles_agree(mine_t, got_t), authors=verdict)
            if verdict == "MISMATCH":
                c.update(not_on_paper=extra, missing_from_ours=missing)
        except Exception as e:                                # noqa: BLE001
            c.update(status="UNREACHABLE",
                     error=("%s: %s" % (type(e).__name__, e))[:140])
        rec["checks"].append(c)
    return rec


def bad_checks(rec):
    out = []
    for c in rec["checks"]:
        s = c.get("status")
        if s == "UNREACHABLE" or (isinstance(s, int) and s >= 400) \
                or c.get("soft_404") or c.get("title_match") is False \
                or c.get("authors") == "MISMATCH":
            out.append(c)
    return out


def main():
    log, problems = [], []
    for n, ref in enumerate(R.REFS, 1):
        rec = check_one(n, ref)
        bad = bad_checks(rec)
        rec["clean"] = not bad
        if bad:
            problems.append(n)
        print("[%2d] %-8s %s" % (n, "CLEAN" if not bad else "PROBLEM",
                                 rec["printed_title"][:62]), flush=True)
        for c in bad:
            print("       %s %s -> %s"
                  % (c["kind"], c["url"][-52:],
                     c.get("error") or c.get("authors") or c.get("status")))
        log.append(rec)

    json.dump({"checked": len(log), "entries_with_a_problem": problems,
               "log": log},
              io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n%d references, %d targets, problems: %s"
          % (len(log), sum(len(r["checks"]) for r in log), problems or "none"))
    print("written %s" % OUT)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
