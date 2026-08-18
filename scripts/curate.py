"""Curate raw generated rows into a clean, deduped, balanced training file.

Reads data/seed.jsonl (hand-written, trusted) + data/generated_raw.jsonl
(synthetic), validates format, drops near-duplicates (within the batch and
against the seed), enforces length bounds, and writes data/seed_scaled.jsonl.

The original data/seed.jsonl is never modified. Point train.py's DATA at
data/seed_scaled.jsonl once you're happy with the result.

Run: python scripts/curate.py
"""
import json
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(__file__))
from taxonomy import SYSTEM, TARGET  # noqa: E402

SEED = "data/seed.jsonl"
RAW = "data/generated_raw.jsonl"
OUT = "data/seed_scaled.jsonl"
MIN_LEN, MAX_LEN = 400, 6000       # assistant answer char bounds
DUP_RATIO = 0.85                   # question similarity threshold


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def valid(o):
    m = o.get("messages")
    if not m or len(m) != 3:
        return False
    if [x["role"] for x in m] != ["system", "user", "assistant"]:
        return False
    if m[0]["content"] != SYSTEM:
        return False
    if not m[1]["content"].strip():
        return False
    if not (MIN_LEN <= len(m[2]["content"]) <= MAX_LEN):
        return False
    return True


def near_dup(q, seen_norms):
    """True if q is a near-duplicate of anything already accepted."""
    qn = norm(q)
    if qn in seen_norms:
        return True
    # cheap prefilter by first 6 words, then ratio check
    for sn in seen_norms:
        if abs(len(sn) - len(qn)) > 60:
            continue
        if SequenceMatcher(None, qn, sn).ratio() >= DUP_RATIO:
            return True
    return False


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    seed = load(SEED)
    raw = load(RAW)
    print(f"seed rows: {len(seed)} | raw generated rows: {len(raw)}\n")

    kept, seen = [], set()
    stats = Counter()

    # seed rows are trusted — keep all, register their questions
    for o in seed:
        kept.append(o)
        seen.add(norm(o["messages"][1]["content"]))
        stats["seed_kept"] += 1

    for o in raw:
        if not valid(o):
            stats["invalid"] += 1
            continue
        q = o["messages"][1]["content"]
        if near_dup(q, seen):
            stats["dup"] += 1
            continue
        seen.add(norm(q))
        kept.append(o)
        stats["gen_kept"] += 1

    with open(OUT, "w") as f:
        for o in kept:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    n = len(kept)
    print("Curation results:")
    for k in ["seed_kept", "gen_kept", "dup", "invalid"]:
        print(f"  {k:10s} {stats[k]}")
    print(f"\nFinal dataset: {n} rows -> {OUT}\n")

    dom = Counter(o.get("domain", "?") for o in kept)
    print("Domain balance:")
    for d in ["education", "offensive", "blue_team", "threat_intel"]:
        tgt = int(TARGET.get(d, 0) * 100)
        print(f"  {d:13s} {dom[d]:4d}  ({100 * dom[d] // n:2d}%)   target {tgt}%")

    print("\nReminder: spot-check threat_intel rows for factual accuracy before "
          "training (generated facts can still be wrong).")
    print("Then set DATA = \"data/seed_scaled.jsonl\" in train.py and retrain.")


if __name__ == "__main__":
    main()
