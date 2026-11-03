"""QLoRA fine-tune for the cybersecurity assistant.

Reads data/seed.jsonl, renders each row to Qwen's <|im_start|> chat format
ourselves (the transformers-5 chat-template engine is buggy here), and trains
LoRA adapters on a 4-bit base. Tuned for an 8 GB GPU; also runs on cloud GPUs
such as Colab's T4 (see notebooks/train_colab.ipynb).
"""
import json
import os
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTConfig, SFTTrainer

# ---- config -----------------------------------------------------------------
MODEL = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-3B-Instruct")   # e.g. BASE_MODEL=unsloth/Qwen2.5-7B-Instruct
DATA = "data/seed_scaled.jsonl"
OUT = os.environ.get("LORA_OUT", "outputs/cybersec-lora")
MAX_SEQ = 1536

# ---- load 4-bit base + attach LoRA ------------------------------------------
model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42,
)

# ---- build text dataset (render Qwen format ourselves) ----------------------
def render(messages):
    out = ""
    for m in messages:
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return out

rows = [json.loads(l) for l in open(DATA) if l.strip()]
texts = [render(r["messages"]) for r in rows]
ds = Dataset.from_dict({"text": texts})
print(f"training rows: {len(ds)}")

# ---- train ------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    args=SFTConfig(
        output_dir=OUT,
        dataset_text_field="text",
        max_length=MAX_SEQ,
        packing=True,                    # concatenate short rows to fill MAX_SEQ
        packing_strategy="bfd",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch = 8
        num_train_epochs=2,
        learning_rate=2e-4,
        warmup_steps=5,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=is_bfloat16_supported(),    # T4 and older GPUs lack bf16 -> fp16
        fp16=not is_bfloat16_supported(),
        logging_steps=1,
        save_strategy="epoch",
        seed=42,
        report_to="none",
    ),
)

stats = trainer.train()
print("final loss:", stats.training_loss)

model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"saved LoRA adapter to {OUT}")
