"""
Autoresearch fine-tuning script using Unsloth + LoRA. Single-GPU, single-file.
Usage: uv run autoresearch_unsloth/train_unsloth.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from prepare_unsloth import DEFAULT_MODEL_NAME, DATASET_DIR, model_dir

import torch
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# ---------------------------------------------------------------------------
# Hyperparameters (agent edits this section directly, no CLI flags needed)
# ---------------------------------------------------------------------------

LORA_R           = 16
LORA_ALPHA       = 32       # typically r or 2*r; formula: update_scale = alpha/r
LORA_DROPOUT     = 0.0      # 0 is optimized in Unsloth; try 0.05-0.1 to combat overfitting
LEARNING_RATE    = 2e-4
BATCH_SIZE       = 2        # per-device batch size
GRAD_ACCUM_STEPS = 4        # effective_batch = BATCH_SIZE * GRAD_ACCUM_STEPS
WARMUP_STEPS     = 20
MAX_STEPS        = 200      # fixed step budget — keep constant for fair comparison
WEIGHT_DECAY     = 0.01
LR_SCHEDULER     = "cosine" # "linear" | "cosine" | "cosine_with_restarts"
USE_RSLORA       = False    # rank-stabilized LoRA: scales by alpha/sqrt(r) instead of alpha/r
TARGET_MODULES   = [        # attention + MLP projection layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ---------------------------------------------------------------------------
# Fixed — do not modify
# ---------------------------------------------------------------------------

MODEL_NAME  = DEFAULT_MODEL_NAME   # change only after running prepare_unsloth.py --model <name>
MODEL_DIR   = model_dir(MODEL_NAME)
# DATASET_DIR imported from prepare_unsloth
MAX_SEQ_LEN = 1024
EVAL_STEPS  = 50

ALPACA_PROMPT = """\
Below is an instruction that describes a task, paired with an input that provides further context. \
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# ---------------------------------------------------------------------------
# Model + LoRA
# ---------------------------------------------------------------------------

t_start = time.time()
torch.cuda.reset_peak_memory_stats()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    load_in_4bit=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=USE_RSLORA,
    loftq_config=None,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

train_ds = load_from_disk(os.path.join(DATASET_DIR, "train"))
eval_ds  = load_from_disk(os.path.join(DATASET_DIR, "eval"))

original_cols = train_ds.column_names

def _format(examples):
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(ALPACA_PROMPT.format(instruction, inp, output) + tokenizer.eos_token)
    return {"text": texts}

train_ds = train_ds.map(_format, batched=True, remove_columns=original_cols)
eval_ds  = eval_ds.map(_format, batched=True, remove_columns=original_cols)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        optim="adamw_8bit",
        seed=42,
        output_dir="/tmp/autoresearch_unsloth_ckpt",
        report_to="none",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=20,
        save_strategy="no",
        dataset_num_proc=4,
    ),
)

t_start_training = time.time()
trainer.train()
t_end = time.time()

# ---------------------------------------------------------------------------
# Evaluation + summary
# ---------------------------------------------------------------------------

final_metrics = trainer.evaluate()
eval_loss = final_metrics.get("eval_loss", float("nan"))

if math.isnan(eval_loss) or eval_loss > 20:
    print("FAIL")
    exit(1)

perplexity       = math.exp(eval_loss)
peak_vram_mb     = torch.cuda.max_memory_allocated() / 1024 / 1024
training_seconds = t_end - t_start_training
effective_batch  = BATCH_SIZE * GRAD_ACCUM_STEPS

print("---")
print(f"eval_loss:        {eval_loss:.6f}")
print(f"perplexity:       {perplexity:.3f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"num_steps:        {MAX_STEPS}")
print(f"lora_r:           {LORA_R}")
print(f"lora_alpha:       {LORA_ALPHA}")
print(f"learning_rate:    {LEARNING_RATE:.2e}")
print(f"effective_batch:  {effective_batch}")
print(f"use_rslora:       {USE_RSLORA}")
