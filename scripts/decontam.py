"""Check the held-out eval set for overlap with the training data.

Compares every question in data/eval.jsonl against every user question in the
training file, using the same normalization and similarity measure as
curate.py. An eval question that is a near-duplicate of a training question
would inflate the fine-tuned model's score, so this exits non-zero if any pair
reaches curate.py's duplicate threshold.

Run: python scripts/decontam.py
"""
import json
import os
import sys
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(__file__))
from curate import DUP_RATIO, norm  # noqa: E402

EVAL = "data/eval.jsonl"
TRAIN = "data/seed_scaled.jsonl"
WARN_RATIO = 0.6                   # report, but don't fail, above this


def main():
    evals = [json.loads(l) for l in open(EVAL) if l.strip()]
    train = [norm(json.loads(l)["messages"][1]["content"]) for l in open(TRAIN) if l.strip()]
    print(f"eval questions: {len(evals)} | training questions: {len(train)}\n")

    fails = 0
    for e in evals:
        q = norm(e["question"])
        best, best_q = 0.0, ""
        for t in train:
            r = SequenceMatcher(None, q, t).ratio()
            if r > best:
                best, best_q = r, t
        if best >= WARN_RATIO:
            tag = "FAIL" if best >= DUP_RATIO else "warn"
            fails += best >= DUP_RATIO
            print(f"  [{tag}] {e['id']} ratio {best:.2f}\n    eval:  {q[:100]}\n    train: {best_q[:100]}")

    if fails:
        print(f"\n{fails} eval question(s) duplicate training data. Rewrite them before evaluating.")
        sys.exit(1)
    print("\nNo eval question duplicates a training question.")


if __name__ == "__main__":
    main()
