"""Interactive chat with the fine-tuned cybersecurity model.

Loads the 4-bit base + trained LoRA adapter and renders Qwen's chat format
ourselves (transformers-5 template is buggy here). Type 'quit' to exit.
"""
import torch
from unsloth import FastLanguageModel

ADAPTER = "outputs/cybersec-lora"
MAX_SEQ = 2048
SYSTEM = ("You are a cybersecurity assistant for authorized security testing, "
          "CTFs, and education. Explain clearly and include detection and "
          "mitigation where relevant.")

model, tok = FastLanguageModel.from_pretrained(
    model_name=ADAPTER, max_seq_length=MAX_SEQ, load_in_4bit=True,
)
FastLanguageModel.for_inference(model)   # 2x faster generation

def render(history):
    out = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    for role, content in history:
        out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    return out

print("Cybersec model ready. Type 'quit' to exit.\n")
history = []
while True:
    user = input("you> ").strip()
    if user.lower() in ("quit", "exit"):
        break
    history.append(("user", user))
    inputs = tok(render(history), return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=512, temperature=0.7,
                         do_sample=True, pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"\nbot> {reply}\n")
    history.append(("assistant", reply))
