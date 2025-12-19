# ===============================
# CLEAN + INSTALL ENVIRONMENT
# ===============================
!pip uninstall -y torch torchvision torchaudio torchao unsloth unsloth-zoo || true
!pip cache purge || true

# Install PyTorch with CUDA 12.1 (supports torch.int1+ dtypes)
!pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Install Unsloth (don't pin missing versions!)
!pip install --upgrade unsloth unsloth-zoo

# Install required training libs
!pip install --no-deps trl peft accelerate bitsandbytes datasets gdown

import os
os.environ["UNSLOTH_SKIP_ZOO"] = "1"          # Must be BEFORE import unsloth
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ===============================
# IMPORTS (ORDER MATTERS)
# ===============================
import unsloth
from unsloth import FastLanguageModel

import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import shutil

# ===============================
# MOUNT DRIVE
# ===============================
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# ===============================
# DOWNLOAD DATASET FROM DRIVE
# ===============================
import gdown
DRIVE_FILE_ID = "1aZWfUhH4jJgbTZ6QpN0TPlMKhSCnIq3F"
DRIVE_LINK = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
LOCAL_DATA_FILE = "dataset_drive.txt"
gdown.download(DRIVE_LINK, LOCAL_DATA_FILE, quiet=False)

with open(LOCAL_DATA_FILE, "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

dataset = Dataset.from_dict({"text": lines})

# ===============================
# LOAD MODEL + APPLY LoRA
# ===============================
model_name = "meta-llama/Llama-3.2-1B-Instruct"
max_seq_length = 768

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,
    use_gradient_checkpointing=False,
    trust_remote_code=True,
)

tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    bias="none",
)

# ===============================
# TOKENIZE
# ===============================
def tokenize_fn(ex):
    return tokenizer(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )

dataset = dataset.map(tokenize_fn, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# ===============================
# TRAIN
# ===============================
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

sft_config = SFTConfig(
    output_dir="output_llama32_law",
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
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    data_collator=collator,
    tokenizer=tokenizer,
    packing=True,
)

trainer.train()

# ===============================
# SAVE MERGED MODEL
# ===============================
merged_dir = "fine_tuned_llama32"
trainer.save_model(merged_dir)

try:
    model.save_pretrained(merged_dir)
except:
    pass

try:
    tokenizer.save_pretrained(merged_dir)
except:
    pass

# ===============================
# GGUF CONVERSION
# ===============================
!rm -rf llama.cpp
!git clone https://github.com/ggml-org/llama.cpp.git
!pip install -r llama.cpp/requirements.txt

output_gguf = f"{merged_dir}.gguf"
!python llama.cpp/convert-hf-to-gguf.py {merged_dir} --outfile {output_gguf} --outtype f16

# ===============================
# UPLOAD TO DRIVE
# ===============================
DEST = f"/content/drive/MyDrive/{output_gguf}"
os.makedirs(os.path.dirname(DEST), exist_ok=True)
shutil.copy(output_gguf, DEST)
print("GGUF uploaded to Drive:", DEST)
