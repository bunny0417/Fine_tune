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
        "trl==0.7.3",                  # Downgrade TRL to avoid GradScaler bug
        "bitsandbytes",
        "accelerate",
        "datasets<4.0.0",
        "peft",
        "git+https://github.com/unslothai/unsloth.git",
    ])
    from unsloth import FastLanguageModel

# ✅ Now import everything else AFTER Unsloth
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# STEP 1: Load Qwen2.5-1.5B with 4-bit via Unsloth
model_name = "Qwen/Qwen2.5-1.5B"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = torch.float16,                # Change to torch.bfloat16 if needed
    load_in_4bit = True,
    use_gradient_checkpointing = True,
    token = None,
    trust_remote_code = True,
)

tokenizer.pad_token = tokenizer.eos_token

# STEP 2: Apply LoRA using Unsloth
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,            # Must be 0.0 for Unsloth's fast patching
    bias="none",
)

# STEP 3: Load & preprocess dataset
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

# STEP 4: Prepare data collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# STEP 5: Define training arguments
training_args = TrainingArguments(
    output_dir="output_law_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    fp16=True,                      # Enable float16 for faster training
    bf16=False,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

# STEP 6: Define SFT config
sft_config = SFTConfig(
    output_dir=training_args.output_dir,
    per_device_train_batch_size=training_args.per_device_train_batch_size,
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    num_train_epochs=training_args.num_train_epochs,
    learning_rate=training_args.learning_rate,
    logging_steps=training_args.logging_steps,
    save_strategy=training_args.save_strategy,
    report_to=training_args.report_to,
    dataset_text_field="text",
    completion_only_loss=True,
    max_length=max_seq_length,
    fp16=True,
    bf16=False,
)

# STEP 7: Train using TRL's SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    data_collator=collator,
    tokenizer=tokenizer,
)

print("[INFO] Starting fine-tuning…")
trainer.train()

# STEP 8: Save final model
print("[INFO] Saving fine-tuned model to './fine_tuned_model'")
trainer.save_model("fine_tuned_model")
