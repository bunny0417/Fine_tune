# ==========================================
# 1. SETUP ENVIRONMENT & GPU CHECK
# ==========================================
import torch
if not torch.cuda.is_available():
    raise RuntimeError("⚠️ STOP! You are on CPU. Go to Runtime > Change runtime type > Select T4 GPU.")

from google.colab import drive
import os
import shutil

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Install Unsloth
try:
    import unsloth
except ImportError:
    print("Installing Unsloth...")
    !pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install --no-deps trl peft accelerate bitsandbytes

from unsloth import FastLanguageModel

# ==========================================
# 2. DEFINE PATHS
# ==========================================
# This is the folder ID from your link: 1K2sslO-rFFlDJvXqWREtsQz9D9iuAu1e
# We need the actual path in your mounted Drive.
# Please ensure the folder 'fine_tuned_llama32' exists in your Drive root or update the path below.

# Heuristic: Try to find the specific folder if exact path isn't known, 
# but usually it's best to specify the path where you see it in the file browser.
# Assuming you saved it as 'fine_tuned_llama32' in your Drive root:
adapter_path = "/content/drive/MyDrive/fine_tuned_llama32"

# If your folder has a different name in Drive, change 'fine_tuned_llama32' above.
if not os.path.exists(adapter_path):
    print(f"⚠️ Could not find folder at: {adapter_path}")
    print("Please check your Drive file browser on the left and copy the path to your adapter folder.")
    # Stop execution if path is wrong to prevent errors
    raise FileNotFoundError("Adapter folder not found. Update 'adapter_path' variable.")

print(f"✅ Found adapters at: {adapter_path}")

# ==========================================
# 3. LOAD & CONVERT
# ==========================================
print("Loading model and merging adapters...")

# Load model + adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = adapter_path, # Points to your Drive folder
    max_seq_length = 768,      # Must match your training settings
    dtype = None,
    load_in_4bit = True,
)

print("Converting to GGUF (this may take a few minutes)...")
# converting to 16-bit GGUF as requested
model.save_pretrained_gguf(
    adapter_path, 
    tokenizer, 
    quantization_method = "f16"
)

# ==========================================
# 4. VERIFY OUTPUT
# ==========================================
output_file = os.path.join(adapter_path, "unsloth.F16.gguf")

if os.path.exists(output_file):
    print(f"\n🎉 Success! Your GGUF file is ready.")
    print(f"Location: {output_file}")
    print("You can now download it directly from your Google Drive.")
else:
    print("❌ Error: GGUF file was not created.")
