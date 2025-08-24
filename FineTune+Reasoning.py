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
        "trl==0.7.3",
        "bitsandbytes",
        "accelerate",
        "datasets<4.0.0",
        "peft",
        "git+https://github.com/unslothai/unsloth.git",
    ])
    from unsloth import FastLanguageModel

# --- IMPORTANT: patch Unsloth to enable GRPO on FastLanguageModel ---
from unsloth import PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

# ✅ Import AFTER Unsloth
from transformers import AutoTokenizer, GenerationConfig
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# STEP 1: Load Qwen2.5-1.5B
model_name = "Qwen/Qwen2.5-1.5B"
max_seq_length = 1024  # give extra tokens for reasoning traces

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,
    use_gradient_checkpointing=True,
    trust_remote_code=True,
)

# Ensure pad token exists
tokenizer.pad_token = tokenizer.eos_token

# Configure generation behavior (control max tokens / sampling)
model.generation_config = GenerationConfig(
    max_new_tokens=256,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
)

# STEP 2: Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
)

# STEP 3: Load dataset and prepare prompts+answers
dataset = load_dataset("Alok2304/Indian_Law_Final_Dataset", split="train")
if "text" not in dataset.column_names:
    raise ValueError("Dataset missing 'text' column.")

SYSTEM_PROMPT = """Respond in the following exact XML format. First show your chain-of-thought inside <reasoning> tags, then give the final answer inside <answer> tags.

<reasoning>
... show step-by-step reasoning here ...
</reasoning>
<answer>
... final concise answer here ...
</answer>

Only write those tags and nothing else. The <answer> should be concise and not repeat long reasoning. Use the format exactly.
"""

def extract_answer_from_text(txt: str):
    if "<answer>" in txt and "</answer>" in txt:
        return txt.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if "####" in txt:
        parts = txt.split("####")
        if len(parts) >= 2:
            return parts[-1].strip()
    if "Answer:" in txt:
        return txt.split("Answer:")[-1].strip()
    return None

# Use plain-string prompts to avoid tokenizer.chat_template issues
def make_prompt_and_answer(example):
    user_text = example["text"].strip()
    answer = extract_answer_from_text(user_text)
    if answer:
        prompt_text = user_text.replace(answer, "").strip()
    else:
        prompt_text = user_text
    prompt_string = SYSTEM_PROMPT + "\n\n" + prompt_text + "\n\n"
    return {"prompt": prompt_string, "answer": answer or ""}

dataset = dataset.map(make_prompt_and_answer, remove_columns=dataset.column_names)
dataset = dataset.shuffle(seed=42).select(range(min(1500, len(dataset))))

# ---------------------------------------------------------------------
# STEP 4: Robust reward function compatible with Unsloth GRPO trainer
# ---------------------------------------------------------------------
import re

def extract_answer_from_generation(gen_text: str) -> str:
    """Find <answer>...</answer> or fallback to last non-empty line."""
    if not isinstance(gen_text, str):
        gen_text = str(gen_text)
    if "<answer>" in gen_text and "</answer>" in gen_text:
        return gen_text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    lines = [l.strip() for l in gen_text.splitlines() if l.strip()]
    return lines[-1] if lines else gen_text.strip()

def simple_string_overlap(a: str, b: str) -> float:
    """Normalized overlap of word-token sets (placeholder verifier)."""
    if not a or not b:
        return 0.0
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    if not sa or not sb:
        return 0.0
    overlap = sa.intersection(sb)
    # normalized by min(len(sa), len(sb)) to punish short refs less
    return float(len(overlap)) / max(1, min(len(sa), len(sb)))

def reward_function(prompts=None, completions=None, completion_ids=None, references=None, **kwargs):
    """
    Unsloth GRPO will call reward funcs with keyword args like:
      prompts, completions, completion_ids, and possibly references/labels.
    This function normalizes inputs and returns a list[float] of rewards (one per completion).
    """
    # 1) Normalize completions -> flat list of strings
    gens = []
    if completions is None:
        completions = []
    # completions may be nested lists, or list of dicts, or list of strings
    if isinstance(completions, list):
        for c in completions:
            if isinstance(c, list):
                # nested lists: extend with flattened entries
                for e in c:
                    if isinstance(e, dict):
                        # hf generation sometimes returns {"text": "..."}
                        text = e.get("text") if isinstance(e.get("text"), str) else str(e)
                        gens.append(text)
                    else:
                        gens.append(str(e))
            else:
                if isinstance(c, dict):
                    text = c.get("text") if isinstance(c.get("text"), str) else str(c)
                    gens.append(text)
                else:
                    gens.append(str(c))
    else:
        gens = [str(completions)]

    # 2) Find references (labels) if provided (try different keys)
    refs_input = references if references is not None else kwargs.get("references", None)
    if refs_input is None:
        refs_input = kwargs.get("labels", None)
    if refs_input is None:
        refs_input = kwargs.get("answers", None)

    # If no references, return zero rewards (unsupervised)
    if refs_input is None:
        return [0.0] * len(gens)

    # Normalize refs_input into a flat list aligned with gens
    refs = []
    if isinstance(refs_input, list):
        # If refs match gens length, use directly
        if len(refs_input) == len(gens):
            refs = [str(r) for r in refs_input]
        else:
            # assume refs length = batch_size (B) and gens length = B * G
            B = len(refs_input)
            if B > 0 and len(gens) % B == 0:
                G = len(gens) // B
                for r in refs_input:
                    refs.extend([str(r)] * G)
            else:
                # fallback: repeat first reference for all gens
                refs = [str(refs_input[0])] * len(gens)
    else:
        # single scalar reference -> repeat
        refs = [str(refs_input)] * len(gens)

    # Ensure lengths match
    if len(gens) != len(refs):
        # attempt to trim or pad refs
        if len(refs) < len(gens):
            refs += [refs[-1]] * (len(gens) - len(refs))
        else:
            refs = refs[:len(gens)]

    # 3) Compute reward per generation
    rewards = []
    for gen, ref in zip(gens, refs):
        gen_ans = extract_answer_from_generation(gen)
        score = simple_string_overlap(gen_ans, str(ref))
        rewards.append(float(score))
    return rewards

# ---------------------------------------------------------------------
# STEP 5: GRPO training config (fixed batch size multiple of num_generations)
# ---------------------------------------------------------------------
grpo_config = GRPOConfig(
    output_dir="output_law_model_grpo",
    per_device_train_batch_size=4,   # must be a multiple of num_generations
    gradient_accumulation_steps=1,
    max_steps=500,
    learning_rate=5e-6,
    logging_steps=20,
    save_strategy="steps",
    save_steps=500,
    num_generations=4,               # keep this at 4
    report_to="none",
    fp16=True,
)

# STEP 6: Create GRPO Trainer and run training
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,          # dataset with 'prompt' (string) and 'answer'
    tokenizer=tokenizer,
    reward_funcs=[reward_function],  # required by UnslothGRPOTrainer
    pack_prompts=True,
)

print("[INFO] Starting GRPO fine-tuning (reasoning)…")
trainer.train()

# STEP 7: Save LoRA adapter and merged model
print("[INFO] Saving LoRA adapter and merged model")
try:
    model.save_lora("grpo_saved_lora")
except Exception as e:
    print("[WARN] save_lora failed:", e)

try:
    model.save_pretrained_merged("fine_tuned_model", tokenizer=tokenizer, save_method="merged_16bit")
except Exception as e:
    print("[WARN] save_pretrained_merged failed:", e)

# Optional: copy to Google Drive (Colab) — preserves your original logic
try:
    from google.colab import drive
    print("[INFO] Mounting Google Drive at /content/drive")
    drive.mount('/content/drive')
except Exception as e:
    print("[WARN] Could not import google.colab.drive (are you in Colab?):", e)

DRIVE_DEST = "/content/drive/MyDrive/fine_tuned_model"
import shutil
if os.path.exists("fine_tuned_model"):
    if os.path.exists(DRIVE_DEST):
        print(f"[INFO] Removing existing Drive folder: {DRIVE_DEST}")
        shutil.rmtree(DRIVE_DEST)
    print(f"[INFO] Copying 'fine_tuned_model' -> '{DRIVE_DEST}' ...")
    shutil.copytree("fine_tuned_model", DRIVE_DEST)
    print(f"[INFO] Model copy complete. Saved to Drive at: {DRIVE_DEST}")
else:
    print("[WARN] Local merged model not found; check saves above.")
