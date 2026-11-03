"""Answer the held-out eval questions with the base or fine-tuned model.

Uses the same system prompt and Qwen <|im_start|> rendering as chat.py, with
greedy decoding so runs are repeatable. Writes one answer per question to
outputs/eval/<model>.jsonl; scripts/judge.py then scores the two files.

With --rag, each question first retrieves passages from data/rag/corpus.jsonl
(see rag.py and scripts/build_corpus.py) and the model answers with them in
the prompt. The passage ids go into the output so each answer can be audited.

Run (on the GPU machine):
    python eval_generate.py --model base                    # -> base.jsonl
    python eval_generate.py --model finetuned               # -> finetuned.jsonl
    python eval_generate.py --model base --rag              # -> base_rag.jsonl
    python eval_generate.py --model finetuned --rag         # -> finetuned_rag.jsonl
    python eval_generate.py --model base --base-model unsloth/Qwen2.5-7B-Instruct \
        --name base7b                                       # -> base7b.jsonl
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
MAX_SEQ_RAG = 4096   # prompt grows by ~k * 350 tokens of passages
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
    ap.add_argument("--base-model", default=BASE, help="base model for --model base")
    ap.add_argument("--adapter", default=ADAPTER, help="LoRA dir for --model finetuned")
    ap.add_argument("--rag", action="store_true", help="add retrieved passages to the prompt")
    ap.add_argument("--k", type=int, default=4, help="passages per question with --rag")
    ap.add_argument("--name", help="output name (default: model, plus _rag with --rag)")
    args = ap.parse_args()
    out_name = args.name or args.model + ("_rag" if args.rag else "")

    retriever = None
    if args.rag:
        from rag import Retriever, rag_question
        retriever = Retriever()

    name = args.base_model if args.model == "base" else args.adapter
    model, tok = FastLanguageModel.from_pretrained(
        model_name=name, max_seq_length=MAX_SEQ_RAG if args.rag else MAX_SEQ,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    rows = [json.loads(l) for l in open(EVAL) if l.strip()]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{out_name}.jsonl"
    with open(out_path, "w") as fout:
        for i, r in enumerate(rows, 1):
            question, ctx = r["question"], []
            if retriever:
                ctx = retriever.search(question, k=args.k)
                question = rag_question(question, ctx)
            inputs = tok(render(question), return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            answer = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            rec = {"id": r["id"], "answer": answer}
            if retriever:
                rec["passages"] = [p["id"] for p in ctx]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(rows)}] {r['id']}  ({len(answer)} chars)")
    print(f"\nWrote {len(rows)} answers -> {out_path}")


if __name__ == "__main__":
    main()
