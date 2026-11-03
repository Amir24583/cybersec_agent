"""Can retrieval even find the facts? A cheap check before spending GPU time.

For each held-out eval question, retrieves the top-k passages and counts a
reference key point as "reachable" when most of its content words appear in
the retrieved text. This is a rough lexical proxy, not a grade: it can miss
paraphrases and can credit passages that share words without stating the fact.
Use it to compare corpus and k settings, then confirm with the real eval.

Run: python scripts/retrieval_check.py            # k=4, all sources
     python scripts/retrieval_check.py --k 6 --show
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag import Retriever, tokenize  # noqa: E402

EVAL = "data/eval.jsonl"


def reachable(point, text_tokens, need=0.6):
    words = {t for t in tokenize(point) if len(t) > 2}
    if not words:
        return False
    return len(words & text_tokens) / len(words) >= need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--show", action="store_true", help="print retrieved titles")
    args = ap.parse_args()

    r = Retriever()
    rows = [json.loads(l) for l in open(EVAL) if l.strip()]
    by_dom = defaultdict(lambda: [0, 0])
    for row in rows:
        hits = r.search(row["question"], k=args.k, sources=args.sources)
        toks = set()
        for h in hits:
            toks |= set(tokenize(h["title"] + " " + h["text"]))
        got = [reachable(p, toks) for p in row["key_points"]]
        for key in (row["domain"], "all"):
            by_dom[key][0] += sum(got)
            by_dom[key][1] += len(got)
        print(f"{row['id']:16s} {sum(got)}/{len(got)}")
        if args.show:
            for h in hits:
                print(f"      {h['score']:6.1f}  {h['source']:6s} {h['title'][:90]}")

    print(f"\nKey points reachable in top-{args.k} passages:")
    for key, (a, n) in by_dom.items():
        print(f"  {key:14s} {a:3d}/{n:3d}  {100 * a / n:.0f}%")


if __name__ == "__main__":
    main()
