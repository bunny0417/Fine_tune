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

# STEP 1: Load Microsoft Phi-2 (2.7B)
# Note: phi-2 typically uses a 2048 token context window; set max_seq_length accordingly.
model_name = "microsoft/phi-2"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,             # keep 4-bit load as in your original script
    use_gradient_checkpointing=False,
    token=None,
    trust_remote_code=True,
)

# Ensure tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# STEP 2: Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    bias="none",
)

# STEP 3: Load dataset (Diweanshu/financial-reasoning-India)
dataset = load_dataset("Diweanshu/financial-reasoning-India", split="train")

# Defensive: what columns exist
print("[INFO] Dataset columns:", dataset.column_names)
print("[INFO] Dataset size before formatting:", len(dataset))

# Build a unified "text" column compatible with your original format_chat logic.
def build_text_row(example):
    system_prompt = example.get("system_prompt") if "system_prompt" in example else None
    prompt = example.get("prompt") if "prompt" in example else None
    completion = example.get("completion") if "completion" in example else None

    if prompt is None and completion is None and system_prompt is None:
        return {"text": "<|user|>Hello<|assistant|>No content"}

    parts = []
    if system_prompt and str(system_prompt).strip():
        parts.append(f"[SYSTEM]\n{system_prompt.strip()}\n[/SYSTEM]\n")

    user_text = prompt.strip() if (prompt and str(prompt).strip()) else "Hello"
    assistant_text = completion.strip() if (completion and str(completion).strip()) else ""

    composed = "".join(parts) + f"<|user|>{user_text}<|assistant|>{assistant_text}"
    return {"text": composed}

# Map to create the "text" column, then remove original columns (so downstream code is unchanged).
dataset = dataset.map(build_text_row, remove_columns=dataset.column_names)

# Optional: quick sanity print
print("[INFO] Example formatted text (truncated 400 chars):")
print(dataset[0]["text"][:400])

# ---------------- SELECT: only the first 200 examples (preserve original order) ----------------
n = min(200, len(dataset))
if n == 0:
    raise ValueError("Loaded dataset is empty (size 0). Cannot continue training.")
print(f"[INFO] Selecting the first {n} samples from the dataset for training (original order).")
dataset = dataset.select(range(n))
# -----------------------------------------------------------------------------------------------

# Tokenize / truncate / pad
dataset = dataset.map(
    lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=max_seq_length),
    batched=True
)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# STEP 4: Data collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# STEP 5: Training config
sft_config = SFTConfig(
    output_dir="output_financial_phi2",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    max_steps=50,               # <<-- changed to 50 optimizer steps
    learning_rate=2e-4,
    logging_steps=20,
    save_strategy="steps",
    save_steps=50,              # <<-- changed to save at 50 steps
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

# Optional: print expected epochs estimate
import math
N = len(dataset)
B_per = sft_config.per_device_train_batch_size
G = sft_config.gradient_accumulation_steps
W = 1
try:
    import torch as _t
    if _t.distributed.is_initialized():
        W = _t.distributed.get_world_size()
except Exception:
    W = 1
effective_batch = B_per * W * G
steps_per_epoch = math.ceil(N / effective_batch) if effective_batch > 0 else float('inf')
expected_epochs = sft_config.max_steps / steps_per_epoch if steps_per_epoch and steps_per_epoch != float('inf') else None
print(f"[INFO] N={N}, effective_batch={effective_batch}, steps_per_epoch={steps_per_epoch}, expected_epochs≈{expected_epochs:.3f}")

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
try:
    from google.colab import drive
    print("[INFO] Mounting Google Drive at /content/drive")
    drive.mount('/content/drive')
except Exception as e:
    print("[WARN] Could not import google.colab.drive (are you in Colab?):", e)

DRIVE_DEST = "/content/drive/MyDrive/fine_tuned_model"

import shutil

if os.path.exists(hf_model_dir):
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
        for f in files[:20]:
            print(os.path.join(root, f))
else:
    print("[WARN] Drive destination does not exist; check mount/authentication and available Drive space.")
