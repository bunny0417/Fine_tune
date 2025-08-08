import os
import subprocess
import torch

# STEP 0: Install requirements and import Unsloth FIRST
try:
    from unsloth import FastLanguageModel
except ImportError:
    subprocess.call([
        "pip", "install",
        "transformers>=4.51.3",
        "trl==0.7.3",  # Specific version avoids GradScaler bug
        "bitsandbytes",
        "accelerate",
        "datasets<4.0.0",
        "peft",
        "git+https://github.com/unslothai/unsloth.git",
    ])
    from unsloth import FastLanguageModel

# ✅ Import AFTER Unsloth
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# STEP 1: Load Qwen2.5-1.5B
model_name = "Qwen/Qwen2.5-1.5B"
max_seq_length = 768

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,
    use_gradient_checkpointing=False,
    token=None,
    trust_remote_code=True,
)

tokenizer.pad_token = tokenizer.eos_token

# STEP 2: Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    bias="none",
)

# STEP 3: Load dataset
dataset = load_dataset("Alok2304/Indian_Law_Final_Dataset", split="train")
if "text" not in dataset.column_names:
    raise ValueError("Dataset missing 'text' column.")

def format_chat(example):
    txt = example["text"]
    if "[INST]" in txt and "[/INST]" in txt:
        user = txt.split("[INST]")[1].split("[/INST]")[0].strip()
        assistant = txt.split("[/INST]")[1].strip()
        return {"text": f"<|user|>{user}<|assistant|>{assistant}"}
    return {"text": f"<|user|>Hello<|assistant|>{txt}"}

dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
dataset = dataset.shuffle(seed=42).select(range(5000))

dataset = dataset.map(
    lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=max_seq_length),
    batched=True
)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# STEP 4: Data collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# STEP 5: Training config
sft_config = SFTConfig(
    output_dir="output_law_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    max_steps=1500,
    learning_rate=2e-4,
    logging_steps=20,
    save_strategy="steps",
    save_steps=500,
    report_to="none",
    dataset_text_field="text",
    completion_only_loss=True,
    fp16=True,
    bf16=False,
)

# STEP 6: Train
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    data_collator=collator,
    tokenizer=tokenizer,
    packing=True,
)

print("[INFO] Starting fine-tuning…")
trainer.train()

# STEP 7: Save model in HF format
hf_model_dir = "fine_tuned_model"
print(f"[INFO] Saving fine-tuned model to '{hf_model_dir}'")
trainer.save_model(hf_model_dir)

# STEP 8: Install llama.cpp and convert to GGUF
print("[INFO] Installing llama.cpp for GGUF conversion…")
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp"], check=True)
os.chdir("llama.cpp")
subprocess.run(["make"], check=True)

# Convert HF → GGUF
gguf_file = "../fine_tuned_model.gguf"
print(f"[INFO] Converting HF model to GGUF: {gguf_file}")
subprocess.run([
    "python3", "convert_hf_to_gguf.py",
    f"../{hf_model_dir}",
    "--outfile", gguf_file,
    "--model-type", "qwen2"  # Qwen architecture
], check=True)

# Optional: Quantize GGUF to smaller size
quantized_file = "../fine_tuned_model.Q4_K_M.gguf"
print(f"[INFO] Quantizing GGUF to {quantized_file}")
subprocess.run([
    "./quantize",
    gguf_file,
    quantized_file,
    "Q4_K_M"
], check=True)

print(f"[INFO] Conversion complete. Files saved:\n - {gguf_file}\n - {quantized_file}")
