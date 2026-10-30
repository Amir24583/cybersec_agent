# Cybersecurity LLM — Fine-Tuning Walkthrough

A do-it-yourself guide to fine-tuning an open model into a specialized
cybersecurity assistant (offensive/CTF, blue team, threat intel, education),
running locally on a single 8 GB GPU.

## Target hardware (this machine)

| Resource | Value | Implication |
|---|---|---|
| GPU | RTX 5060 Laptop, **8 GB VRAM**, Blackwell (sm_120) | QLoRA only; needs CUDA 12.8+ builds |
| RAM | 15 GB | fine for data prep / offload |
| Disk | ~940 GB free | plenty |
| CPU | 16 cores | fine |

**Approach:** QLoRA (4-bit base + small trainable LoRA adapters). Full
fine-tuning is not possible on 8 GB.

**Honest expectations:** fine-tuning gives strong *domain adaptation* —
security fluency, tool/format knowledge, consistent style, offline use. It
shines at education/Q&A, threat-intel summarization, and blue-team format
generation (Sigma/YARA/SIEM). Deep offensive *reasoning* stays the weakest
area for small models; RAG over a CVE/knowledge base helps more there than
fine-tuning does.

---

## Pipeline overview

```
environment  →  dataset  →  QLoRA train  →  evaluate  →  export to Ollama
```

Get the environment **provably working** (load a model + run one training
step) before touching datasets.

---

## Phase 1 — Environment (do first)

Two landmines on this machine:
- **Python 3.13 is too new** for parts of the ML stack (bitsandbytes, triton,
  unsloth) → use **Python 3.11**.
- **Blackwell GPU (sm_120)** needs the **cu128** PyTorch build, or you get
  `no kernel image is available for execution on the device` at runtime.

### 1.1 Python 3.11 via uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# restart shell, or: source $HOME/.local/bin/env

cd /home/amirs/programming/Dev/cybersec_agent
uv venv --python 3.11 .venv
source .venv/bin/activate
python --version    # must say 3.11.x
```

> Alternative: `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.11 python3.11-venv`, then `python3.11 -m venv .venv`.

### 1.2 PyTorch with Blackwell (cu128) support

Install torch BEFORE unsloth so you control the build:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 1.3 VERIFY the GPU (do not skip)

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "NONE")
x = torch.randn(1000, 1000, device="cuda")
y = (x @ x).sum().item()
print("GPU matmul OK, result is finite:", y == y)
PY
```

Expected: `cuda available: True`, device `RTX 5060 Laptop GPU`, capability
`(12, 0)`, `GPU matmul OK: True`.

Troubleshooting:
- `no kernel image` → wrong torch build; reinstall with the cu128 index URL.
- `cuda available: False` → WSL2 not seeing GPU; run `nvidia-smi` in WSL, and
  update the **Windows** NVIDIA driver (WSL CUDA comes from the Windows driver).

### 1.4 Unsloth + load-a-model smoke test

```bash
uv pip install unsloth
```

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B-Instruct", max_seq_length=2048, load_in_4bit=True)
# then generate one sentence to confirm it works
```

Loads under ~5–6 GB and generates → environment is done.

---

## Phase 2 — Dataset

The dataset determines model quality far more than any hyperparameter. For
QLoRA, **quality and consistency beat raw volume** — a few thousand clean,
well-formatted examples outperform 100k noisy ones.

### 2.1 Format: chat JSONL

One JSON object per line. Each has a `messages` list (system / user /
assistant). This is the standard SFT format.

```json
{"messages": [
  {"role": "system", "content": "You are a cybersecurity assistant for authorized security testing and education."},
  {"role": "user", "content": "What is a SQL injection and how do I prevent it?"},
  {"role": "assistant", "content": "SQL injection is ... To prevent it: 1) parameterized queries ..."}
]}
```

We render each example to plain text in Qwen's `<|im_start|>` format
**ourselves** (the `transformers 5.x` chat-template engine is buggy on this
machine — see Phase 1.4), so the pipeline doesn't depend on
`apply_chat_template`.

### 2.2 The ethics / authorization frame

Every example — especially offensive ones — uses a system prompt that scopes
the assistant to **authorized testing, CTFs, and education**. Offensive answers
should explain concepts, methodology, and detection/mitigation, not hand over
turnkey attacks against real targets. This keeps the model useful for learning
and defense without training it to be a no-questions weapon.

### 2.3 Strategy: seed set first, then scale

1. **Hand-write a small seed set (~30–50 examples)** spanning the 4 domains.
   Purpose: validate the *training* pipeline end-to-end on real data before
   investing in curation.
2. **Scale with curated public data** (HuggingFace security instruction
   datasets, CVE/MITRE ATT&CK text, write-ups) + optional synthetic generation.
3. **Decontaminate**: keep an eval set separate; remove any training rows that
   overlap it.
4. **Deduplicate** near-identical rows.

### 2.4 Domain mix (rough target)

| Domain | Share | Example content |
|---|---|---|
| Education / Q&A | 35% | concept explanations, "how does X work" |
| Threat intel | 25% | CVE/ATT&CK summarization, report analysis |
| Blue team | 25% | Sigma/YARA rules, log analysis, IR steps |
| Offensive/CTF | 15% | methodology, tool usage, CTF reasoning (scoped) |

### 2.5 Quality rules

- Consistent assistant voice and structure.
- Correct, specific answers (no hallucinated CVE numbers/commands).
- Vary phrasing of questions so the model generalizes.
- Keep examples within `max_seq_length` (1024–2048 tokens).

## Phase 3 — QLoRA training

Unsloth training script. Start with the 3B (model is a config value, swappable
to 7B later). batch size 1–2, gradient accumulation, `max_seq_length`
1024–2048, gradient checkpointing.

## Phase 4 — Evaluation

Held-out questions per domain + spot checks.

## Phase 5 — Export to Ollama

Merge adapter → GGUF → `ollama create` → local `cybersec` model, offline.

---

## Progress

- [x] 1.1 Python 3.11 venv
- [x] 1.2 PyTorch cu128
- [x] 1.3 GPU verification passes
- [x] 1.4 Unsloth + smoke test (Qwen2.5-3B, 2.4 GB, LoRA attached)
- [x] Phase 2 — dataset: 110 hand-written rows, audited + domain-tagged
- [ ] Phase 2b — scale to 1-2k via scripts/ (generate.py + curate.py)
- [x] Phase 3 — first QLoRA run (works; style learned, facts weak on 96 rows)
- [ ] Phase 3b — retrain on scaled dataset
- [ ] Phase 4 — eval
- [ ] Phase 5 — export
