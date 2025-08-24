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
dataset = dataset.shuffle(seed=42).select(range(1500))

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
    max_steps=500,
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

# STEP 7: Save model in HF format (local) and then copy to Google Drive
hf_model_dir = "fine_tuned_model"
print(f"[INFO] Saving fine-tuned model to local folder '{hf_model_dir}'")
trainer.save_model(hf_model_dir)

# Try to explicitly save model & tokenizer as well (helps with PEFT adapters)
try:
    model.save_pretrained(hf_model_dir)
except Exception as e:
    print("[WARN] model.save_pretrained() failed:", e)

try:
    tokenizer.save_pretrained(hf_model_dir)
except Exception as e:
    print("[WARN] tokenizer.save_pretrained() failed:", e)

# --- Mount Google Drive and copy the folder ---
# You will be prompted to authenticate when this runs in Colab.
try:
    from google.colab import drive
    print("[INFO] Mounting Google Drive at /content/drive")
    drive.mount('/content/drive')
except Exception as e:
    print("[WARN] Could not import google.colab.drive (are you in Colab?):", e)

# Destination on your Drive — change if you want a different location
DRIVE_DEST = "/content/drive/MyDrive/fine_tuned_model"

import shutil

if os.path.exists(hf_model_dir):
    # If destination exists, remove it first to ensure a clean copy
    if os.path.exists(DRIVE_DEST):
        print(f"[INFO] Removing existing Drive folder: {DRIVE_DEST}")
        shutil.rmtree(DRIVE_DEST)
    print(f"[INFO] Copying '{hf_model_dir}' -> '{DRIVE_DEST}' (this may take a while for large files)...")
    shutil.copytree(hf_model_dir, DRIVE_DEST)
    print(f"[INFO] Model copy complete. Saved to Drive at: {DRIVE_DEST}")
else:
    print(f"[ERROR] Local model folder not found: {hf_model_dir}. Nothing copied to Drive.")

# Quick verification list
if os.path.exists(DRIVE_DEST):
    print("[INFO] Files in Drive destination:")
    for root, dirs, files in os.walk(DRIVE_DEST):
        for f in files[:20]:  # print up to 20 files for brevity
            print(os.path.join(root, f))
else:
    print("[WARN] Drive destination does not exist; check mount/authentication and available Drive space.")
