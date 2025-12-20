# ==========================================
# 0. DISABLE WANDB (SILENCE PROMPTS)
# ==========================================
import os
os.environ["WANDB_DISABLED"] = "true"

# ==========================================
# 1. INSTALLATION (CRASH-PROOF)
# ==========================================
print("Installing Unsloth and dependencies...")
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# Force binary installs to prevent xformers build errors
!pip install -q --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
!pip install -q datasets tqdm

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from tqdm import tqdm

# ==========================================
# 2. CONFIGURATION
# ==========================================
MODEL_NAME = "huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated"
NEW_MODEL_NAME = "DeepSeek-R1-HackerOne-Uncensored-Full"
MAX_SEQ_LENGTH = 2048 
DTYPE = None 
LOAD_IN_4BIT = True 

# ==========================================
# 3. LOAD MODEL
# ==========================================
print(f"Loading {MODEL_NAME} in 4-bit mode...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==========================================
# 4. PREPARE FULL DATASET
# ==========================================
print("Loading FULL HackerOne dataset...")
dataset = load_dataset("Hacker0x01/hackerone_disclosed_reports", split="train")

def filter_high_quality(x):
    try:
        # 1. Get Content
        content = x.get('vulnerability_information', '')
        if not content or len(str(content)) < 300: # Skip junk
            return False

        # 2. Find Severity 
        severity = None
        if x.get('structured_scope'):
            severity = x['structured_scope'].get('max_severity')
        
        # 3. Check Weakness Keywords
        if not severity and x.get('weakness'):
            weakness_name = x['weakness'].get('name', '').lower()
            if any(k in weakness_name for k in ['injection', 'rce', 'takeover', 'xss', 'bypass']):
                severity = 'critical'

        # 4. Check Votes
        votes = x.get('vote_count', 0)
        
        # LOGIC: Keep if Severity is valid OR it's a popular report
        # We include Medium/Low if they have high votes to increase dataset size
        is_severe = str(severity).lower() in ['critical', 'high', 'medium', 'low']
        is_popular = (votes is not None and votes > 5)
        
        return is_severe or is_popular

    except Exception as e:
        return False

print("Filtering dataset (Keeping ALL valid reports)...")
filtered_dataset = dataset.filter(filter_high_quality)

print(f"✅ Final Training Set: {len(filtered_dataset)} reports will be used.")

# ==========================================
# 5. FORMATTING
# ==========================================
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Analyze and explain the vulnerability: {}

### Response:
Sure, here is the vulnerability analysis and exploitation guide:
<think>
1. Identify the vulnerability type from the report.
2. Analyze the impact and root cause.
3. Determine the steps to reproduce (exploit).
</think>
{}"""

def formatting_prompts_func(examples):
    instructions = examples["title"]
    outputs      = examples["vulnerability_information"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        if not output: continue
        text = alpaca_prompt.format(instruction, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

print("Formatting all data...")
train_dataset = filtered_dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 6. TRAIN (FULL EPOCH)
# ==========================================
print(f"Starting training on {len(train_dataset)} reports for 1 FULL EPOCH...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        # KEY CHANGE: Train on 100% of the data (1 epoch)
        num_train_epochs = 1, 
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # No W&B
    ),
)

trainer.train()

# ==========================================
# 7. EXPORT
# ==========================================
print("Training complete. Converting to GGUF...")
# Using Q4_K_M for balance
model.save_pretrained_gguf(NEW_MODEL_NAME, tokenizer, quantization_method = "q4_k_m")
print(f"✅ DONE! Full dataset training finished. Download the folder: {NEW_MODEL_NAME}")
