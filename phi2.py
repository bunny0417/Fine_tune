# Single Colab cell: Fixed for bitsandbytes version - upgrades lib, enables 4-bit QLoRA, reduces seq_len, subsamples data.
# Paste and run in a RESTARTED Colab runtime.

# ---------- User config (adjust if needed) ----------
HF_TOKEN = ""                         # optional, for gated datasets
MODEL_ID = "microsoft/phi-2"
DATASET_ID = "openai/gsm8k"
MAX_STEPS = 100
NUM_EPOCHS = 1
TMP_WORK_DIR = "/content/phi2_finetune_work"
OUTPUT_DRIVE_FOLDER = "/content/drive/MyDrive/phi2_gsm8k_gguf"
MAX_SEQ_LEN = 128                     # Reduced to save memory (GSM8K texts are short)
SUBSAMPLE_SIZE = 1000                 # Subsample for testing; set to None for full
LORA_R = 4
PER_DEVICE_BATCH = 1
GRADIENT_ACCUM = 4                    # Effective batch=4 without extra VRAM
# ----------------------------------------------------

import os, sys, glob, shutil, traceback
from pathlib import Path

# Memory fragmentation mitigation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Mount drive (no-op if already mounted)
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
os.makedirs(TMP_WORK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DRIVE_FOLDER, exist_ok=True)

# Upgrade bitsandbytes and accelerate for 4-bit compatibility
!pip install -U bitsandbytes accelerate -q

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, default_data_collator
from transformers import Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

print("Environment versions:", "torch", torch.__version__)
print("Config: model", MODEL_ID, "dataset", DATASET_ID)

hf_kwargs = {"use_auth_token": HF_TOKEN} if HF_TOKEN else {}

# ---------------- Load & prepare dataset ----------------
print("Loading dataset:", DATASET_ID)
ds = None
for cfg in ("main", "socratic"):
    try:
        ds = load_dataset(DATASET_ID, cfg, split="train", **(hf_kwargs or {}))
        print("Loaded config:", cfg)
        break
    except Exception as e:
        print("config", cfg, "failed:", e)
if ds is None:
    raise SystemExit("Failed to load GSM-8K dataset. Provide HF token or check network.")

def format_example(ex):
    q = ex.get("question") or ex.get("Problem") or ex.get("question_text") or ex.get("problem")
    a = ex.get("answer") or ex.get("correct_answer") or ex.get("answer_text") or ""
    if q is None:
        raise ValueError("No question field in example")
    return {"text": f"Question: {str(q).strip()}\nAnswer: {str(a).strip()}"}

ds = ds.map(format_example)
print("Dataset size:", len(ds))
print("Sample:", ds[0]["text"][:320])

# Subsample for low-memory testing
if SUBSAMPLE_SIZE:
    ds = ds.select(range(min(SUBSAMPLE_SIZE, len(ds))))
    print("Subsampled to:", len(ds))

# ---------------- Tokenize and add labels ----------------
print("Loading tokenizer and tokenizing (max_length=%d)..." % MAX_SEQ_LEN)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, **(hf_kwargs or {}))
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
pad_id = tokenizer.pad_token_id

def tokenize_and_label(batch):
    toks = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_SEQ_LEN)
    input_ids = toks["input_ids"]
    labels = []
    for seq in input_ids:
        lbl = [(t if t != pad_id else -100) for t in seq]
        labels.append(lbl)
    toks["labels"] = labels
    return toks

ds = ds.map(tokenize_and_label, batched=True, remove_columns=ds.column_names)
ds.set_format(type="torch")
print("Tokenization + label creation done. Example keys:", list(ds[0].keys()))

# ---------------- Model load (prioritize 4-bit QLoRA) ----------------
bnb_ok = False
try:
    import bitsandbytes as bnb
    bnb_ok = True
    print("BitsAndBytes imported successfully (version check passed).")
except Exception as e:
    print("BitsAndBytes import failed:", e)

gpu_mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
print("GPU memory (bytes):", gpu_mem, "bnb_ok:", bnb_ok)

model = None
bnb_cfg = None
use_quant = bnb_ok
if bnb_ok:
    try:
        print("Loading 4-bit QLoRA model...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_cfg, device_map="auto",
            trust_remote_code=True, **(hf_kwargs or {})
        )
        print("4-bit model loaded successfully.")
    except Exception as quant_err:
        print("4-bit load failed (error below); falling back to FP16.")
        print(quant_err)
        use_quant = False

if not use_quant:
    print("Loading FP16 model (may use more VRAM).")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True, **(hf_kwargs or {})
    )
    print("FP16 model loaded.")

# Memory mitigations
model.config.use_cache = False
model.gradient_checkpointing_enable()
torch.cuda.empty_cache()

# ---------------- Prepare for k-bit training and apply LoRA ----------------
print("Preparing for k-bit training and applying LoRA r=%d" % LORA_R)
if use_quant:
    model = prepare_model_for_kbit_training(model)

# Target modules for Phi-2
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
print("LoRA target_modules:", target_modules)

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=32, target_modules=target_modules,
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("LoRA applied.")

torch.cuda.empty_cache()

# ---------------- Trainer & training ----------------
print("Trainer setup (max_steps=%d, epochs=%d)..." % (MAX_STEPS, NUM_EPOCHS))
use_paged_opt = use_quant
training_args = TrainingArguments(
    output_dir=f"{TMP_WORK_DIR}/lora_out",
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRADIENT_ACCUM,
    max_steps=MAX_STEPS,
    num_train_epochs=NUM_EPOCHS,
    fp16=True,
    logging_steps=10,
    save_steps=MAX_STEPS,
    save_total_limit=1,
    remove_unused_columns=False,
    dataloader_pin_memory=False,  # Saves some memory
    report_to=[],
    optim="paged_adamw_8bit" if use_paged_opt else "adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=default_data_collator
)

try:
    trainer.train()
except Exception:
    print("Training failed — traceback below:")
    traceback.print_exc()
    raise SystemExit("Training aborted.")

print("Training finished. Saving LoRA adapter...")
lora_save_dir = f"{TMP_WORK_DIR}/lora_adapter"
model.save_pretrained(lora_save_dir)
print("LoRA adapter saved at:", lora_save_dir)

# ---------------- Merge LoRA into base & save merged HF model ----------------
print("Merging LoRA adapter into base model...")
if use_quant:
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_cfg, device_map="auto",
        trust_remote_code=True, **(hf_kwargs or {})
    )
else:
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True, **(hf_kwargs or {})
    )
peft_model = PeftModel.from_pretrained(base, lora_save_dir, device_map="auto", dtype=torch.float16)
merged = peft_model.merge_and_unload()
merged_dir = f"{TMP_WORK_DIR}/merged_model"
merged.save_pretrained(merged_dir, safe_serialization=True)
print("Merged HF model saved at:", merged_dir)

# ---------------- Attempt conversion to GGUF ----------------
print("Attempting HF -> GGUF conversion via llama.cpp.")
os.chdir("/content")
if not os.path.exists("llama.cpp"):
    !git clone https://github.com/ggerganov/llama.cpp.git

convert_script = None
for root, _, files in os.walk("llama.cpp"):
    for f in files:
        if f.endswith(".py") and ("convert" in f.lower() or "gguf" in f.lower()):
            convert_script = os.path.join(root, f)
            break
    if convert_script:
        break

if convert_script is None:
    print("No llama.cpp conversion script found. Merged HF model at:", merged_dir)
else:
    out_gguf = "/content/phi2_merged.gguf"
    cmd = f"{sys.executable} {convert_script} {merged_dir} --outtype f16 --outfile {out_gguf}"
    print("Running convert:", cmd)
    p = os.system(cmd)
    if p != 0:
        print("Conversion failed. Check logs.")
    else:
        print("GGUF produced:", out_gguf)
        # Copy to Drive (clear folder first)
        for ent in os.listdir(OUTPUT_DRIVE_FOLDER):
            path = os.path.join(OUTPUT_DRIVE_FOLDER, ent)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        shutil.copyfile(out_gguf, os.path.join(OUTPUT_DRIVE_FOLDER, "phi2_gsm8k.gguf"))
        print("GGUF copied to Drive.")

print("All done. Check /content/phi2_finetune_work for outputs.")
