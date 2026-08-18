import torch
from unsloth import FastLanguageModel

MODEL = "unsloth/Qwen2.5-3B-Instruct"  # this config is swappable later


# load 4-bit base

model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=2048,
        load_in_4bit=True,)
print("VRAM after load (GB):",
      round(torch.cuda.memory_allocated()/1e9,2))


# # generate a reply 
# msgs = ({"role": "user", "content": "In one snetence, what is a SQL injection?"})

# #input = input("prompt here:")
# #msgs = ({"role": "user", "content": input})

# inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
# out = model.generate(inputs, max_new_tokens=60)
# print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))

# Generate one reply — manual Qwen chat format (avoids transformers-5 template bug)
prompt = (
    "<|im_start|>user\n"
    "In one sentence, what is a SQL injection?<|im_end|>\n"
    "<|im_start|>assistant\n"
)
inputs = tok(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=60)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

# attach LoRA adapters (this proves that training is set up)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",
)
print("LoRA attached OK.")
