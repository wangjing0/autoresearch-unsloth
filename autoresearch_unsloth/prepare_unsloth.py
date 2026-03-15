"""
One-time data preparation for autoresearch-unsloth experiments.
Downloads the base model snapshot and prepares the alpaca-gpt4 fine-tuning dataset.

Usage:
    uv run autoresearch_unsloth/prepare_unsloth.py
    uv run autoresearch_unsloth/prepare_unsloth.py --model unsloth/Llama-3.2-1B-Instruct
    uv run autoresearch_unsloth/prepare_unsloth.py --model meta-llama/Llama-3.1-8B-Instruct --token hf_...

The model name becomes a slug used as the cache subdirectory, so multiple models
can coexist under ~/.cache/autoresearch_unsloth/models/.
Dataset is shared across all models at ~/.cache/autoresearch_unsloth/dataset/.
"""

import argparse
import os

from datasets import load_dataset
from huggingface_hub import snapshot_download

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
DATASET_NAME       = "vicgalle/alpaca-gpt4"
EVAL_SPLIT_RATIO   = 0.05   # 5% held out for validation
RANDOM_SEED        = 42

CACHE_DIR   = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_unsloth")
DATASET_DIR = os.path.join(CACHE_DIR, "dataset")

def model_dir(model_name: str) -> str:
    slug = model_name.replace("/", "--")
    return os.path.join(CACHE_DIR, "models", slug)

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def download_model(model_name: str, token: str | None = None):
    dest = model_dir(model_name)
    marker = os.path.join(dest, "config.json")
    if os.path.exists(marker):
        print(f"Model: already cached at {dest}")
        return
    print(f"Model: downloading {model_name} ...")
    os.makedirs(dest, exist_ok=True)
    snapshot_download(repo_id=model_name, local_dir=dest, token=token)
    print(f"Model: saved to {dest}")

# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset():
    train_path = os.path.join(DATASET_DIR, "train")
    eval_path  = os.path.join(DATASET_DIR, "eval")
    if os.path.exists(train_path) and os.path.exists(eval_path):
        print(f"Dataset: already prepared at {DATASET_DIR}")
        return

    print(f"Dataset: loading {DATASET_NAME} ...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Dataset: {len(dataset):,} examples loaded")

    split    = dataset.train_test_split(test_size=EVAL_SPLIT_RATIO, seed=RANDOM_SEED)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"Dataset: {len(train_ds):,} train / {len(eval_ds):,} eval")

    os.makedirs(DATASET_DIR, exist_ok=True)
    train_ds.save_to_disk(train_path)
    eval_ds.save_to_disk(eval_path)
    print(f"Dataset: saved to {DATASET_DIR}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare model and dataset for autoresearch-unsloth")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help=f"HuggingFace model repo ID (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token for gated models (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Model:           {args.model}")
    print(f"Model dir:       {model_dir(args.model)}")
    print()

    download_model(args.model, token=token)
    print()

    prepare_dataset()
    print()

    print("Done! Ready to fine-tune.")
    print(f"  Update MODEL_DIR in train_unsloth.py to: {model_dir(args.model)}")
    print(f"  Run: uv run autoresearch_unsloth/train_unsloth.py")
