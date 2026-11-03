"""Score base vs. fine-tuned answers on the held-out eval set with Claude as judge.

For each question the judge sees both answers, labeled A and B in random order
(so it can't favor a position), plus the question's reference key points. It
returns, per answer, which key points are covered and a 1-5 accuracy score,
then which answer is better overall. Results are written to
outputs/eval/results.json and a summary table to outputs/eval/results.md.

Prereqs: run eval_generate.py for both models first, and
    export ANTHROPIC_API_KEY=sk-ant-...      # or: ant auth login

Run: python scripts/judge.py                                  # finetuned vs base
     python scripts/judge.py --ref base --cand base_rag        # any two answer files
Answer files are outputs/eval/<name>.jsonl. Comparing other pairs writes
results_<cand>_vs_<ref>.json/.md so earlier results are kept.
"""
import argparse
import json
import random
from collections import defaultdict

import anthropic

EVAL = "data/eval.jsonl"
OUT_DIR = "outputs/eval"
DOMAINS = ["education", "blue_team", "threat_intel", "offensive"]

# A stronger model than the one that generated the training data, so the
# judge isn't grading its own style.
MODEL = "claude-opus-5-5"

JUDGE_SYSTEM = (
    "You grade answers from cybersecurity assistants for an evaluation. "
    "Be strict about factual accuracy: wrong CVE details, event IDs, commands, "
    "or rule syntax count against an answer even if it is well written. "
    "Judge only content, not length or formatting."
)

GRADE = {
    "type": "object",
    "properties": {
        "key_points_covered": {"type": "array", "items": {"type": "boolean"}},
        "accuracy": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "errors": {"type": "string"},
    },
    "required": ["key_points_covered", "accuracy", "errors"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "A": GRADE,
        "B": GRADE,
        "preferred": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["A", "B", "preferred", "reason"],
    "additionalProperties": False,
}


def build_prompt(row, answer_a, answer_b):
    points = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(row["key_points"]))
    return (
        f"Question:\n{row['question']}\n\n"
        f"Reference key points a strong answer covers:\n{points}\n\n"
        f"<answer_A>\n{answer_a}\n</answer_A>\n\n"
        f"<answer_B>\n{answer_b}\n</answer_B>\n\n"
        f"For each answer:\n"
        f"- key_points_covered: exactly {len(row['key_points'])} booleans, one per "
        f"reference key point in order, true if the answer states it correctly.\n"
        f"- accuracy: 5 = no factual errors, 3 = some errors or misleading claims, "
        f"1 = mostly wrong.\n"
        f"- errors: the specific factual errors, or an empty string.\n"
        f"Then set preferred to the answer more useful to a security learner, "
        f"or tie, and give a one-sentence reason."
    )


def judge(client, row, answer_a, answer_b):
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": build_prompt(row, answer_a, answer_b)}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        # If a safety classifier declines (offensive-security questions can
        # trigger it), re-run on Anthropic's recommended fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        return None
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def fit(flags, n):
    """Force the judge's key-point list to length n (pads with False)."""
    return (list(flags) + [False] * n)[:n]


def summarize(results, ref="base", cand="finetuned"):
    ms = (ref, cand)
    agg = defaultdict(lambda: {"n": 0, "kp": {m: [0, 0] for m in ms},
                               "acc": {m: 0 for m in ms},
                               "win": 0, "tie": 0, "loss": 0})
    for r in results:
        for key in (r["domain"], "all"):
            a = agg[key]
            a["n"] += 1
            for m in ms:
                kp = r[m]["key_points_covered"]
                a["kp"][m][0] += sum(kp)
                a["kp"][m][1] += len(kp)
                a["acc"][m] += r[m]["accuracy"]
            a[{cand: "win", "tie": "tie", ref: "loss"}[r["preferred"]]] += 1

    lines = [
        f"| Domain | Questions | Key points: {ref} | Key points: {cand} "
        f"| Accuracy (1-5): {ref} | Accuracy (1-5): {cand} | {cand} W / T / L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in DOMAINS + ["all"]:
        if key not in agg:
            continue
        a = agg[key]
        kp = {m: 100 * a["kp"][m][0] / max(a["kp"][m][1], 1) for m in ms}
        acc = {m: a["acc"][m] / a["n"] for m in ms}
        label = "**All**" if key == "all" else key
        lines.append(f"| {label} | {a['n']} | {kp[ref]:.0f}% | {kp[cand]:.0f}% "
                     f"| {acc[ref]:.2f} | {acc[cand]:.2f} "
                     f"| {a['win']} / {a['tie']} / {a['loss']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N questions")
    ap.add_argument("--seed", type=int, default=42, help="seed for the A/B order")
    ap.add_argument("--ref", default="base", help="reference answers (outputs/eval/<ref>.jsonl)")
    ap.add_argument("--cand", default="finetuned", help="candidate answers to compare")
    args = ap.parse_args()
    ref, cand = args.ref, args.cand
    if ref == cand:
        ap.error("--ref and --cand must differ")
    stem = "results" if (ref, cand) == ("base", "finetuned") else f"results_{cand}_vs_{ref}"

    rows = [json.loads(l) for l in open(EVAL) if l.strip()]
    answers = {}
    for m in (ref, cand):
        answers[m] = {o["id"]: o["answer"] for o in map(json.loads, open(f"{OUT_DIR}/{m}.jsonl"))}
    if args.limit:
        rows = rows[: args.limit]

    client = anthropic.Anthropic()
    rng = random.Random(args.seed)
    results, skipped = [], []
    for i, row in enumerate(rows, 1):
        base, ft = answers[ref][row["id"]], answers[cand][row["id"]]
        ft_is_a = rng.random() < 0.5
        a, b = (ft, base) if ft_is_a else (base, ft)
        try:
            g = judge(client, row, a, b)
        except (anthropic.APIError, json.JSONDecodeError, StopIteration) as e:
            print(f"  [{i}/{len(rows)}] SKIP {row['id']} -> {e}")
            skipped.append(row["id"])
            continue
        if g is None:
            print(f"  [{i}/{len(rows)}] SKIP {row['id']} -> judge refused")
            skipped.append(row["id"])
            continue

        n = len(row["key_points"])
        for side in ("A", "B"):
            g[side]["key_points_covered"] = fit(g[side]["key_points_covered"], n)
        ft_side, base_side = ("A", "B") if ft_is_a else ("B", "A")
        preferred = {"tie": "tie", ft_side: cand, base_side: ref}[g["preferred"]]
        results.append({"id": row["id"], "domain": row["domain"],
                        ref: g[base_side], cand: g[ft_side],
                        "preferred": preferred, "reason": g["reason"]})
        print(f"  [{i}/{len(rows)}] {row['id']:16s} preferred: {preferred}")

    with open(f"{OUT_DIR}/{stem}.json", "w") as f:
        json.dump({"judge": MODEL, "ref": ref, "cand": cand,
                   "results": results, "skipped": skipped}, f, indent=2)
    table = summarize(results, ref, cand)
    with open(f"{OUT_DIR}/{stem}.md", "w") as f:
        f.write(f"Judge: {MODEL}. {cand} vs {ref}. Skipped: {len(skipped)}.\n\n{table}\n")
    print(f"\n{table}\n\nSkipped: {skipped or 'none'}")
    print(f"Wrote {OUT_DIR}/{stem}.json and {OUT_DIR}/{stem}.md")


if __name__ == "__main__":
    main()
