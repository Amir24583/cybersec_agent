"""Answer the held-out eval questions with the base or fine-tuned model.

Uses the same system prompt and Qwen <|im_start|> rendering as chat.py, with
greedy decoding so runs are repeatable. Writes one answer per question to
outputs/eval/<model>.jsonl; scripts/judge.py then scores the two files.

Run (on the GPU machine):
    python eval_generate.py --model base
    python eval_generate.py --model finetuned
"""
import argparse
import json
import os

import torch
from unsloth import FastLanguageModel

BASE = "unsloth/Qwen2.5-3B-Instruct"   # keep in sync with MODEL in train.py
ADAPTER = "outputs/cybersec-lora"
EVAL = "data/eval.jsonl"
OUT_DIR = "outputs/eval"
MAX_SEQ = 2048
MAX_NEW = 768
SYSTEM = ("You are a cybersecurity assistant for authorized security testing, "
          "CTFs, and education. Explain clearly and include detection and "
          "mitigation where relevant.")


def render(question):
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "finetuned"], required=True)
    args = ap.parse_args()

    name = BASE if args.model == "base" else ADAPTER
    model, tok = FastLanguageModel.from_pretrained(
        model_name=name, max_seq_length=MAX_SEQ, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    rows = [json.loads(l) for l in open(EVAL) if l.strip()]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{args.model}.jsonl"
    with open(out_path, "w") as fout:
        for i, r in enumerate(rows, 1):
            inputs = tok(render(r["question"]), return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            answer = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            fout.write(json.dumps({"id": r["id"], "answer": answer}, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(rows)}] {r['id']}  ({len(answer)} chars)")
    print(f"\nWrote {len(rows)} answers -> {out_path}")


if __name__ == "__main__":
    main()
