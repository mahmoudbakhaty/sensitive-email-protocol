# -*- coding: utf-8 -*-
"""Did you get the same corpus we did?

The release says the benchmark is not redistributed but rebuilt from a public
archive. That promise is only worth something if a reader can tell whether the
archive they downloaded is the archive we used, and until now nothing recorded
it: no checksum of the tarball, and no corpus-level digest of the rebuilt data.

The one fingerprint the release did publish, `fold_fingerprint`, cannot serve
that purpose. It hashes the fold assignment, and Section VII-F of the paper is
about how the scikit-learn version changes exactly that. A reader on any other
version gets a different value and has no way to know whether the data
differed or only the splitter did. So this script separates the two:

  * ARCHIVE     sha256 of the tarball as served.
  * CORPUS      a digest of the rebuilt messages themselves - text and labels,
                in a fixed order. Independent of library versions. This is the
                one that answers "same data?".
  * FOLDS       the published fold fingerprint, reported beside the
                scikit-learn version that produced it, and expected to differ
                on a different version.

Exits non-zero only if the corpus itself differs. A fold mismatch on a
different scikit-learn is reported and is not an error.
"""
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.request as UR

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

URL = "https://bailando.berkeley.edu/enron/enron_with_categories.tar.gz"

# Recorded 2026-09-23 from the archive as served, and the corpus rebuilt from
# it. The counts are the ones every table in the paper is computed over.
EXPECTED = {
    "archive_sha256":
        "08625500ab4c032f99acfc1f8eed125b2cd5e9c4280c7ce2d7ce0bd28a190625",
    "archive_bytes": 4523350,
    "corpus_digest": "493f90a5df5b8f40",
    "messages": 1382,
    "threads": 1103,
    "strict_positives": 250,
    "broad_positives": 422,
    "fold_fingerprint": "63e3aea5c3d37629",
    "fold_fingerprint_sklearn": "1.9.1",
}


def corpus_digest(df):
    """Over the message texts and both labels, order-independent.

    Not over the folds: this must answer "same data?" without depending on
    which scikit-learn drew the splits.

    Not in the dataframe's own order either. build() orders messages with
    sorted(glob(...)) over paths, and the path separator differs by platform -
    a backslash sorts after the digits in the directory names, a forward slash
    before them - so the same corpus comes out in a different order on Windows
    and on POSIX. Each message is hashed alone and the digest is taken over
    the sorted hashes, so the order cannot matter."""
    rows = []
    for t, a, b in zip(df.text.tolist(), df.label.tolist(),
                       df.label_either.tolist()):
        h = hashlib.sha256()
        h.update(t.encode("utf-8", "replace"))
        h.update(str(len(t)).encode())
        h.update(("%d%d" % (int(a), int(b))).encode())
        rows.append(h.hexdigest())
    return hashlib.sha256("".join(sorted(rows)).encode()).hexdigest()[:16]


def fold_fingerprint(df):
    folds = list(GroupKFold(n_splits=5).split(
        np.zeros(len(df)), df.label.values, df.thread_key.values))
    return hashlib.md5("|".join(",".join(map(str, te)) for _, te in folds)
                       .encode()).hexdigest()[:16]


def fetch(dest):
    print("downloading %s" % URL, flush=True)
    req = UR.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with UR.urlopen(req, timeout=300) as r, io.open(dest, "wb") as f:
        f.write(r.read())
    return dest


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", help="a tarball already downloaded")
    ap.add_argument("--extracted", help="a directory already extracted")
    ap.add_argument("--work", default=os.path.join(HERE, "_corpus_check"))
    a = ap.parse_args()

    ok = True
    data_dir = a.extracted

    if not data_dir:
        os.path.isdir(a.work) or os.makedirs(a.work)
        arc = a.archive or fetch(os.path.join(a.work, "corpus.tar.gz"))
        raw = io.open(arc, "rb").read()
        got = hashlib.sha256(raw).hexdigest()
        same = got == EXPECTED["archive_sha256"]
        print()
        print("ARCHIVE")
        print("  bytes   %d  (expected %d)"
              % (len(raw), EXPECTED["archive_bytes"]))
        print("  sha256  %s" % got)
        print("          %s" % ("matches the recorded archive" if same else
                                "DIFFERS from the recorded archive - the "
                                "publisher may have reissued it"))
        tarfile.open(arc).extractall(a.work)
        data_dir = os.path.join(a.work, "enron_with_categories")
    else:
        print("ARCHIVE  skipped (--extracted given)")

    import strengthen as S
    S.DATA_DIR = data_dir
    df = S.build(dedup=True)

    got = {"corpus_digest": corpus_digest(df),
           "messages": int(len(df)),
           "threads": int(df.thread_key.nunique()),
           "strict_positives": int(df.label.sum()),
           "broad_positives": int(df.label_either.sum())}
    print()
    print("CORPUS  (independent of library versions - this is the one that "
          "answers 'same data?')")
    for k in ("messages", "threads", "strict_positives", "broad_positives",
              "corpus_digest"):
        good = got[k] == EXPECTED[k]
        ok &= good
        print("  %-17s %-18s %s" % (k, got[k], "ok" if good else
                                    "DIFFERS, expected %s" % EXPECTED[k]))

    fp = fold_fingerprint(df)
    print()
    print("FOLDS   (version-bound - see Section VII-F and "
          "results/FINDING_sklearn_groupkfold.md)")
    print("  scikit-learn here    %s   (recorded run used %s)"
          % (sklearn.__version__, EXPECTED["fold_fingerprint_sklearn"]))
    print("  fold fingerprint     %s   (recorded %s)"
          % (fp, EXPECTED["fold_fingerprint"]))
    if fp == EXPECTED["fold_fingerprint"]:
        print("  identical splits")
    else:
        print("  different splits. On a different scikit-learn this is "
              "expected and is NOT a reproduction failure:")
        print("  GroupKFold's assignment changed between versions, which is "
              "what Section VII-F measures.")
        print("  The corpus block above is the check that matters.")

    print()
    print("corpus reproduces exactly" if ok else
          "CORPUS DOES NOT REPRODUCE - do not compare any number in the paper "
          "against a run on this data")
    json.dump({"expected": EXPECTED, "observed": got,
               "fold_fingerprint_here": fp,
               "sklearn_here": sklearn.__version__, "corpus_ok": bool(ok)},
              io.open(os.path.join(os.path.dirname(HERE), "results",
                                   "RESULTS_corpus_verification.json"),
                      "w", encoding="utf-8"), indent=1)
    return 0 if bool(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
