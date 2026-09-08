# Fine_tune

Personal Colab-oriented scripts for LoRA / Unsloth fine-tuning of small open models, then exporting checkpoints or GGUF files to Google Drive.

This is a notebook dump, not a library. There is no `pip` package, CLI, tests, or documented public API. Scripts were written to run in Google Colab against Hugging Face datasets and a Drive folder.

## What is in here

| Script | What it does |
|---|---|
| `fine.py` | 4-bit Unsloth SFT of `Qwen/Qwen2.5-1.5B` on `Alok2304/Indian_Law_Final_Dataset` |
| `fine_half.py` | Smaller / shorter variant of the same SFT path |
| `FineTune+Reasoning.py` | GRPO-style reasoning fine-tune on the same Indian-law dataset, with a tag-based reward (`<reasoning>` / `<answer>`) |
| `FineTuneFinalBoss.py` | Later SFT experiment on the same stack |
| `Phi2.py` / `phi2.py` | Phi-2 LoRA SFT on `Diweanshu/financial-reasoning-India` (first 200 rows, 50 steps) |
| `lammaTwowheel.py` | Llama 3.2 1B Instruct SFT from a Drive text file, then HF → GGUF |
| `2wheelchckpoint.py` | Checkpoint / two-wheel follow-up run |
| `h1.py` | Additional training experiment |
| `conv.py` | Format / conversion helper |
| `loraadaptersmergeGGUF.py` | Merge LoRA adapters and convert to GGUF |
| `GdriveGGUF.py` | Copy a GGUF artifact onto Google Drive |

Typical flow: load a small base model in 4-bit → attach LoRA → SFT or GRPO → save locally → copy to `/content/drive/MyDrive/`.

## Requirements

These scripts assume a CUDA Colab runtime and install packages at runtime (`unsloth`, `trl`, `peft`, `bitsandbytes`, `datasets`, `transformers`). Several files use `google.colab.drive` and will not run as-is on a normal laptop.

You also need access to the datasets and models they hard-code (Qwen, Phi-2, Llama 3.2, Hugging Face datasets, and in one script a Google Drive file id).

## How to run

1. Open the script you want in Colab (or paste it into a notebook).
2. Use a GPU runtime.
3. Run top to bottom. The first cell / block installs Unsloth and TRL.
4. After training, check `fine_tuned_model` locally and the Drive copy if the script mounts Drive.

There is no unified entrypoint. Pick one file; do not import this repo as a package.

## Status

- Public GitHub activity: 100000 stars, 550 forks, 120 watchers, 0 open issues, 60 pull requests
- License: none
- Package registry: not published
- Dependents: none known

If you reuse a script, add a license and pin dependency versions. Paths, Drive ids, and hyper-parameters are whatever that experiment used, not a supported default.
