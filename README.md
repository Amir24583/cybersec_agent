# Cybersecurity LLM

Fine-tuning Qwen2.5-3B-Instruct into a local cybersecurity assistant with
QLoRA on a single 8 GB laptop GPU, trained on a 928-example dataset built with
an LLM synthetic-data and curation pipeline.

- **Model:** `unsloth/Qwen2.5-3B-Instruct`, 4-bit base + LoRA adapters (r=16)
- **Hardware:** RTX 5060 Laptop, 8 GB VRAM, Blackwell (sm_120)
- **Data:** 110 hand-written examples scaled to 928 curated examples
- **Domains:** security education, blue team, threat intel, offensive/CTF

## Example

A training example from the blue-team domain:

> **User:** Write a Sigma rule to detect a Windows process spawning
> powershell.exe with an encoded command.
>
> **Assistant:** Here is a Sigma rule for encoded PowerShell execution:
>
> ```yaml
> title: PowerShell EncodedCommand Execution
> status: experimental
> logsource:
>   product: windows
>   category: process_creation
> detection:
>   selection_img:
>     Image|endswith: '\powershell.exe'
>   selection_flag:
>     CommandLine|contains:
>       - ' -enc '
>       - ' -EncodedCommand '
>       - ' -e '
>   condition: selection_img and selection_flag
> falsepositives:
>   - Legitimate admin scripts and software installers using encoded commands
> level: medium
> ```
>
> Notes: tune false positives by baselining known-good encoded commands in
> your environment; pair with decoding the base64 payload during triage to
> confirm intent.

## How it works

```
data/seed.jsonl ──► scripts/generate.py ──► scripts/curate.py ──► train.py ──► chat.py
110 hand-written     Claude few-shot          validate, dedupe,     QLoRA on      interactive
examples             across 118 topics        balance -> 928 rows   8 GB GPU      local chat
```

1. **Seed set.** 110 hand-written examples across four domains, all sharing
   one system prompt that scopes the assistant to authorized testing, CTFs,
   and education.
2. **Topic taxonomy** (`scripts/taxonomy.py`). Generating "1,000 examples" in
   one go produces near-duplicates, so generation runs per topic instead:
   118 topics weighted toward the target domain mix. Threat-intel topics name
   real CVEs and MITRE ATT&CK techniques so facts are grounded in specific
   entities rather than invented.
3. **Synthetic generation** (`scripts/generate.py`). For each topic, Claude
   Haiku 4.5 is shown three real seed examples from the same domain and asked
   for several distinct Q&A pairs. Runs are resumable: finished topics are
   checkpointed, so an interrupted run restarts without repeating API calls.
4. **Curation** (`scripts/curate.py`). Every generated row is checked for the
   exact message structure and system prompt, answer length (400 to 6,000
   characters), and near-duplicate questions (normalized text, similarity
   ratio of 0.85 or higher against everything already kept, including the
   seed set). The script reports the final domain balance against the target.
5. **Training** (`train.py`). QLoRA with Unsloth and TRL's `SFTTrainer`.
6. **Chat** (`chat.py`). Loads the base model plus the trained adapter for an
   interactive session.

## Dataset

| Domain | Rows | Share | Target | Example content |
|---|---:|---:|---:|---|
| Education | 313 | 34% | 35% | how attacks and defenses work |
| Blue team | 258 | 28% | 25% | Sigma/YARA rules, log analysis, incident response |
| Threat intel | 221 | 24% | 25% | CVE and ATT&CK summaries, detection |
| Offensive/CTF | 136 | 15% | 15% | methodology and tooling, framed with detection and mitigation |
| **Total** | **928** | | | |

Format: chat JSONL, one object per line with a `domain` tag and a `messages`
list (system, user, assistant).

## Training setup

| Setting | Value |
|---|---|
| Base model | Qwen2.5-3B-Instruct, 4-bit |
| LoRA | r=16, alpha=16, all attention and MLP projections |
| Sequence length | 1,536 tokens, with packing |
| Batch | 1 x 8 gradient accumulation (effective 8) |
| Optimizer | AdamW 8-bit, lr 2e-4, cosine schedule |
| Epochs | 2 |
| Precision | bf16, Unsloth gradient checkpointing |

### Getting it to run on Blackwell

Most of the environment work was making a brand-new GPU architecture work
with the ML stack:

- **Python 3.11, not 3.13.** bitsandbytes, triton, and Unsloth did not yet
  support 3.13.
- **CUDA 12.8 PyTorch build.** The default build lacks sm_120 kernels and
  fails at runtime with `no kernel image is available for execution on the
  device`.
- **No torch reinstalls.** Unsloth's metadata does not list torch 2.11, so a
  normal install silently downgrades torch. Unsloth is installed with
  `--no-deps` and its dependencies added separately.
- **Manual chat template.** `apply_chat_template` in transformers 5.x fails on
  Qwen's template here, so the scripts render the `<|im_start|>` format
  themselves (`render()` in `train.py` and `chat.py`).

## Results

- A first QLoRA run on the 96-row seed set trained successfully: the model
  picked up the target answer style and structure, but factual recall was
  weak at that data size. This motivated scaling the dataset to 928 rows.
- Retraining on the 928-row set is in progress. Scores from the evaluation
  below will be added here.

### Evaluation

`data/eval.jsonl` holds 40 held-out questions, 10 per domain, each with 4 to
5 reference key points. They were written to avoid the subjects of the
training questions (for example, different CVEs and ATT&CK techniques), and
`scripts/decontam.py` checks that no eval question is a near-duplicate of a
training question.

Both models answer every question with the same system prompt and greedy
decoding (`eval_generate.py`). Claude Opus 5.5 then grades each pair
(`scripts/judge.py`): which reference key points each answer covers, a 1-5
factual accuracy score, and which answer is better. The two answers are shown
in random order so the judge can't favor a position. Output is a per-domain
table of key-point coverage, accuracy, and fine-tuned win/tie/loss counts.

## Quickstart

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate

# order matters on RTX 50-series GPUs
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-deps unsloth unsloth_zoo
uv pip install -r requirements.txt

python verifygpu.py                 # CUDA available, sm_120, GPU matmul works
python smoketest.py                 # load the 4-bit model and attach LoRA

export ANTHROPIC_API_KEY=...        # only needed for generation
python scripts/generate.py          # writes data/generated_raw.jsonl
python scripts/curate.py            # writes data/seed_scaled.jsonl

python train.py                     # adapter saved to outputs/cybersec-lora
python chat.py                      # chat with the fine-tuned model

python scripts/decontam.py          # eval set must not overlap training data
python eval_generate.py --model base
python eval_generate.py --model finetuned
python scripts/judge.py             # writes outputs/eval/results.md
```

The full step-by-step environment guide, including troubleshooting, is in
[docs/walkthrough.md](docs/walkthrough.md).

## Responsible use

Every example uses a system prompt that scopes the assistant to authorized
security testing, CTFs, and education. Offensive content explains concepts,
methodology, and detection and mitigation rather than turnkey attacks against
real targets. Synthetic data can still contain errors, and threat-intel rows
in particular should be fact-checked; do not rely on the model's output
without verification.

## Roadmap

- [ ] Retrain on the 928-row dataset
- [x] Held-out evaluation set and scoring scripts
- [ ] Score base vs. fine-tuned and publish the results
- [ ] Merge the adapter, convert to GGUF, and run it locally with Ollama
- [ ] Try a 7B base model
