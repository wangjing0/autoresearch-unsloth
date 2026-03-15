"""
One-time dataset preparation for autoresearch-unsloth experiments.
Prepares the alpaca-cleaned fine-tuning dataset. Base models are downloaded
on-demand by train_unsloth.py at the start of each run.

Usage:
    uv run autoresearch_unsloth/prepare_unsloth.py
"""

import os

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

DATASET_NAME     = "unsloth/alpaca-cleaned"
EVAL_SPLIT_RATIO = 0.05   # 5% held out for validation
RANDOM_SEED      = 42

CACHE_DIR   = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_unsloth")
DATASET_DIR = os.path.join(CACHE_DIR, "dataset")

def model_dir(model_name: str) -> str:
    slug = model_name.replace("/", "--")
    return os.path.join(CACHE_DIR, "models", slug)

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
    print(f"Cache directory: {CACHE_DIR}")
    print()

    prepare_dataset()
    print()

    print("Done! Dataset ready.")
    print("Base models are downloaded automatically when train_unsloth.py runs.")
    print("Run: uv run autoresearch_unsloth/train_unsloth.py")
