# Retrieval (RAG) experiment

## Why

The first held-out eval (40 questions, graded blind) showed the 928-row QLoRA
fine-tune of Qwen2.5-3B learned the answer style but not the facts: key-point
coverage 30% vs 29% for the base model, accuracy 1.75 vs 2.00 out of 5, and
12 wins, 9 ties, 19 losses. Threat intel was the weakest domain for both
models (18% and 20% of key points), and it is mostly lookup facts: CVE details,
ATT&CK technique IDs, data sources and detections.

A 3B model can't hold that many facts reliably, and fine-tuning on a few hundred
examples doesn't add many. Retrieval sidesteps this: look the facts up at answer
time and put them in the prompt.

## How it works

| Piece | What it does |
|---|---|
| `scripts/build_corpus.py` | Downloads the sources and writes ~220-word passages to `data/rag/corpus.jsonl` (git-ignored, rebuilt in seconds). |
| `rag.py` | BM25 search over the passages, pure Python with no extra dependencies. Keyword search fits because questions name exact IDs (T1059.001, CVE-2023-4966). |
| `eval_generate.py --rag` | Retrieves the top 4 passages per question and adds them to the user turn. The passage ids are saved with each answer. |
| `scripts/retrieval_check.py` | A cheap check, before any GPU time, of whether the retrieved text even contains the reference key points. |
| `scripts/judge.py --ref A --cand B` | Compares any two answer files with the same blind rubric. |

Sources are general public references, not chosen per question:

| Source | Passages | Notes |
|---|---:|---|
| MITRE ATT&CK Enterprise v17.1 | 2,062 | Techniques, groups, software, mitigations, with detection notes |
| CISA KEV catalog | 1,766 | One short entry per exploited CVE |
| OWASP Cheat Sheet Series | 2,916 | Web app security how-tos |
| Wikipedia, 12 security categories | ? | Fills networking, crypto and CTF basics; couldn't be fetched from the build sandbox, so its size is unknown until it runs on Colab |
| Training answers (`seed_scaled.jsonl`) | 2,133 | Claude-written, so they may carry the same errors as the fine-tune |

`data/eval.jsonl` is never read when building the corpus.

## Retrieval check (no GPU)

Share of the 198 reference key points whose content words mostly appear in the
retrieved passages. This is a lexical proxy, not a grade. For scale, the same
proxy finds 16% of key points in the base model's own answers (the judge gave
it 29%).

| Corpus (no Wikipedia yet) | Top 4 | Top 6 |
|---|---:|---:|
| ATT&CK + KEV + OWASP | 15% | |
| + training answers | 26% | 32% |

Retrieval finds the right document for nearly every ATT&CK and CVE question
(T1036, T1059.001, T1070.001, T1190, T1562.001, AS-REP roasting, NotPetya,
Stuxnet, all three CVEs). It finds little for networking and CTF questions
(BGP, VLAN hopping, Spectre, nmap scan types, RSA e=3, hash length extension),
which is what the Wikipedia source is meant to cover.

## What to run

In Colab, sections 1, 2 and 7 of `notebooks/train_colab.ipynb`. No training
and no adapter needed. It produces:

- `base.jsonl`: plain 3B base (greedy, so it reproduces the first eval)
- `base_rag.jsonl`: 3B base with retrieval
- `base7b_rag.jsonl`: 7B base with retrieval
- `finetuned_rag.jsonl`: only if the adapter is present

Then `scripts/judge.py --ref base --cand base_rag` and `--cand base7b_rag`.

## Results

Pending the Colab run.
