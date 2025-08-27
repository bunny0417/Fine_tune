# Install dependencies
!pip install transformers huggingface_hub peft torch safetensors sentencepiece

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download
import os

# 1. Download base model
base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
base_model_path = snapshot_download(base_model_id)

# 2. Specify LoRA weights folder (must be a valid checkpoint)
lora_weights_path = "/content/lora_weights"  # <-- set to your LoRA folder

# 3. Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype="auto"
)

# 4. Load LoRA and merge (only if valid folder)
if os.path.exists(lora_weights_path) and os.listdir(lora_weights_path):
    model = PeftModel.from_pretrained(model, lora_weights_path)
    model = model.merge_and_unload()

# 5. Save merged model locally
merged_path = "/content/merged_model"
os.makedirs(merged_path, exist_ok=True)
model.save_pretrained(merged_path)

# 6. Make sure tokenizer is also saved in merged folder
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merged_path)

# 7. Convert merged model to GGUF
!git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp
!python3 /llama.cpp/convert_hf_to_gguf.py \
    "{merged_path}" \
    --outfile "/content/deepseek-f16.gguf" \
    --outtype f16
