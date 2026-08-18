"""Synthetic training-data generator (Phase 2 scaling).

For each topic in the taxonomy, few-shot prompt the API with real seed rows
from the SAME domain, and request several *distinct* examples. Output is
appended to data/generated_raw.jsonl (raw, uncurated). Resumable: topics
already generated are skipped.

Prereqs (one-time):
    uv pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...      # or: ant auth login

Run:
    python scripts/generate.py                    # all domains, default counts
    python scripts/generate.py --domain threat_intel --per-topic 3
"""
import argparse
import json
import os
import random
import sys

import anthropic

sys.path.insert(0, os.path.dirname(__file__))
from taxonomy import DOMAINS, SYSTEM  # noqa: E402

SEED = "data/seed.jsonl"
RAW = "data/generated_raw.jsonl"
DONE = "data/.generated_done.txt"

# Bulk generation: a cheaper model with temperature (for diversity) is the
# right tool here. Opus 4.7/4.8 reject `temperature`, so we use Haiku/Sonnet.
# Bump to claude-sonnet-4-6 for higher-quality prose at ~5x the cost.
MODEL = "claude-haiku-4-5"


def load_exemplars():
    """Group real seed rows by domain to use as few-shot examples."""
    by_domain = {}
    for line in open(SEED):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        d = o.get("domain", "education")
        by_domain.setdefault(d, []).append(o["messages"])
    return by_domain


def build_prompt(domain, topic, exemplars, k):
    shots = random.sample(exemplars, min(3, len(exemplars)))
    demo = "\n\n".join(
        f"USER: {m[1]['content']}\nASSISTANT: {m[2]['content']}" for m in shots
    )
    grounding = ""
    if domain == "threat_intel":
        grounding = (
            "\nCRITICAL: Only state facts you are confident are correct about "
            "the specific real entity named in the topic. Do NOT invent CVE "
            "numbers, ATT&CK IDs, dates, or version numbers. If unsure of a "
            "specific detail, describe it conceptually instead of guessing.\n"
        )
    return (
        f"You are creating training data for a cybersecurity assistant. Here are "
        f"{len(shots)} real examples of the exact style and depth required:\n\n"
        f"{demo}\n\n"
        f"Now write {k} NEW, DISTINCT question-and-answer pairs about this topic: "
        f"\"{topic}\".\n"
        f"Requirements:\n"
        f"- Vary the phrasing and scenario across the {k} pairs.\n"
        f"- Match the style: structured, accurate, include detection/mitigation "
        f"where relevant; frame offensive content around authorized/CTF/education.\n"
        f"- Each answer roughly 800-2200 characters.\n"
        f"{grounding}"
        f"Return ONLY a JSON array of objects: "
        f'[{{"user": "...", "assistant": "..."}}, ...]. No other text.'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS) + ["all"], default="all")
    ap.add_argument("--per-topic", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="cap topics (for a proof run)")
    args = ap.parse_args()

    client = anthropic.Anthropic()
    exemplars = load_exemplars()
    done = set(open(DONE).read().split("\n")) if os.path.exists(DONE) else set()

    domains = list(DOMAINS) if args.domain == "all" else [args.domain]
    jobs = [(d, t) for d in domains for t in DOMAINS[d] if f"{d}::{t}" not in done]
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"Model: {MODEL} | topics to generate: {len(jobs)} | {args.per_topic}/topic")
    added = 0
    with open(RAW, "a") as fout, open(DONE, "a") as fdone:
        for i, (domain, topic) in enumerate(jobs, 1):
            prompt = build_prompt(domain, topic, exemplars[domain], args.per_topic)
            try:
                resp = client.messages.create(
                    model=MODEL, max_tokens=8192, temperature=1.0,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = next(b.text for b in resp.content if b.type == "text").strip()
                text = text[text.find("["): text.rfind("]") + 1]  # strip any preamble
                pairs = json.loads(text, strict=False)
            except Exception as e:
                print(f"  [{i}/{len(jobs)}] SKIP {domain}/{topic[:40]} -> {e}")
                continue
            for p in pairs:
                if not p.get("user") or not p.get("assistant"):
                    continue
                obj = {"domain": domain, "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": p["user"].strip()},
                    {"role": "assistant", "content": p["assistant"].strip()},
                ]}
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                added += 1
            fdone.write(f"{domain}::{topic}\n")
            fdone.flush()
            print(f"  [{i}/{len(jobs)}] {domain:12s} +{len(pairs)}  ({topic[:45]})")
    print(f"\nDone. Added {added} raw rows -> {RAW}")
    print("Next: python scripts/curate.py")


if __name__ == "__main__":
    main()
