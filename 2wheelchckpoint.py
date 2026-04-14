# ===============================
# RESUME FLAG — CHANGE THIS ONLY
# ===============================
RESUME = False   # ←←← Set to True when you want to continue from Drive checkpoint

# ===============================
# IMPORTS (dependencies already installed)
# ===============================
import unsloth
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import requests
import json
import shutil
import os
from google.colab import drive
from transformers import DataCollatorForLanguageModeling, TrainerCallback

# ===============================
# MOUNT DRIVE
# ===============================
drive.mount("/content/drive", force_remount=True)

CHECKPOINT_DRIVE_FOLDER = "/content/drive/MyDrive/llama32_2wheel_checkpoints"
os.makedirs(CHECKPOINT_DRIVE_FOLDER, exist_ok=True)

# ===============================
# HARDCODED DATASET + CLEANING
# ===============================
DATASET_URL = "https://raw.githubusercontent.com/bunny0417/Ai_Metrics/refs/heads/main/Two_Wheeler_Text_Dataset.jsonl"

print("Downloading Two_Wheeler_Text_Dataset.jsonl...")
response = requests.get(DATASET_URL)
response.raise_for_status()

raw_data = []
for line in response.text.strip().splitlines():
    if line.strip():
        try:
            raw_data.append(json.loads(line))
        except:
            continue

clean_data = []
for item in raw_data:
    q = item.get("question")
    a = item.get("answer")
    if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
        clean_data.append({"question": q, "answer": a})

print(f"✅ Loaded {len(clean_data)} valid Q&A pairs")
dataset = Dataset.from_list(clean_data)

# ===============================
# FORMAT AS LLAMA-3.2 CHAT
# ===============================
def format_example(example):
    return {
        "text": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in two-wheelers (motorcycles, scooters, bikes, electric vehicles, parts, maintenance, regulations, torque specs, troubleshooting, etc.). 
Answer accurately, concisely and professionally.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example["question"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example["answer"]}<|eot_id|>"""
    }

dataset = dataset.map(format_example, remove_columns=dataset.column_names)
print("Dataset formatted ✅")

# ===============================
# LOAD MODEL (with resume support)
# ===============================
model_name = "meta-llama/Llama-3.2-1B-Instruct"
max_seq_length = 1024

print("🚀 Starting fresh training..." if not RESUME else "🔄 Resume mode enabled...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,
    trust_remote_code=True,
)

# Apply LoRA only once, for both fresh and resume runs
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=False,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

tokenizer.pad_token = tokenizer.eos_token

# ===============================
# TOKENIZE
# ===============================
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )

dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# ===============================
# TRAIN CONFIG (frequent checkpoints for free tier)
# ===============================
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

sft_config = SFTConfig(
    output_dir="./output_llama32_2wheel",   # local folder
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=1.5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=20,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    report_to="none",
    packing=True,
    completion_only_loss=True,
    fp16=True,
    optim="adamw_8bit",
    weight_decay=0.01,
)

# ===============================
# SAVE EVERY CHECKPOINT TO GDRIVE
# ===============================
class SaveEveryCheckpointToDriveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        local_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(local_checkpoint_dir):
            drive_checkpoint_dir = os.path.join(CHECKPOINT_DRIVE_FOLDER, f"checkpoint-{state.global_step}")
            if os.path.exists(drive_checkpoint_dir):
                shutil.rmtree(drive_checkpoint_dir)
            shutil.copytree(local_checkpoint_dir, drive_checkpoint_dir)
            print(f"✅ Saved checkpoint to Drive: checkpoint-{state.global_step}")
        return control

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    data_collator=collator,
    tokenizer=tokenizer,
    callbacks=[SaveEveryCheckpointToDriveCallback()],
)

print("🚀 Starting / Resuming training...")

resume_path = None
if RESUME:
    checkpoints = [d for d in os.listdir(CHECKPOINT_DRIVE_FOLDER) if d.startswith("checkpoint-")]
    if checkpoints:
        latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        resume_path = os.path.join(CHECKPOINT_DRIVE_FOLDER, latest)
        print(f"🔄 Resuming from checkpoint: {resume_path}")
    else:
        print("No checkpoint found on Drive — continuing from scratch.")

if resume_path:
    trainer.train(resume_from_checkpoint=resume_path)
else:
    trainer.train()

# ===============================
# SAVE FINAL MERGED MODEL + GGUF
# ===============================
merged_dir = "fine_tuned_llama32_2wheel"
trainer.save_model(merged_dir)
model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)

!rm -rf llama.cpp
!git clone https://github.com/ggml-org/llama.cpp.git
!pip install -r llama.cpp/requirements.txt --quiet

output_gguf = f"{merged_dir}.gguf"
!python llama.cpp/convert-hf-to-gguf.py {merged_dir} --outfile {output_gguf} --outtype f16

DEST = f"/content/drive/MyDrive/{output_gguf}"
os.makedirs(os.path.dirname(DEST), exist_ok=True)
shutil.copy(output_gguf, DEST)
print(f"🎉 FINAL GGUF saved to Drive: {DEST}")
