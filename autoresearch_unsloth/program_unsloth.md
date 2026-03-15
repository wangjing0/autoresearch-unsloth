# autoresearch-unsloth

Autonomous LoRA hyperparameter search using Unsloth. The agent iterates overnight to find
the best fine-tuning hyperparameters for a given base model on the Alpaca instruction-following task.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch
   `autoresearch-unsloth/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-unsloth/<tag>` from current master.
3. **Read the in-scope files**:
   - `autoresearch_unsloth/prepare_unsloth.py` — fixed constants: model name, dataset, cache paths. Do not modify.
   - `autoresearch_unsloth/train_unsloth.py` — the file you modify. Hyperparameters section at the top.
   - `unsloth/references/llms-full.md` — Unsloth documentation. Search for the LoRA hyperparameters
     section to understand the search space and tuning guidance before starting.
4. **Verify data exists**: Check that `~/.cache/autoresearch_unsloth/` contains `models/<model-slug>/`
   and `dataset/`. If not, tell the human to run:
   `uv run autoresearch_unsloth/prepare_unsloth.py --model <MODEL_NAME>`
   The model slug is the repo ID with `/` replaced by `--` (e.g. `unsloth--Qwen2.5-0.5B-Instruct`).
5. **Initialize results_unsloth.tsv**: Create `autoresearch_unsloth/results_unsloth.tsv` with the header row only.
6. **Confirm and go**.

## Experimentation

Each experiment trains for a **fixed budget of MAX_STEPS=200 gradient updates** regardless of batch
size or other configuration. This makes runs directly comparable. You launch it as:

```
uv run autoresearch_unsloth/train_unsloth.py > run.log 2>&1
```

**What you CAN do:**

- Modify `autoresearch_unsloth/train_unsloth.py` — the hyperparameters section only. All LoRA and optimizer
  hyperparameters are fair game: LORA_R, LORA_ALPHA, LORA_DROPOUT, LEARNING_RATE, BATCH_SIZE,
  GRAD_ACCUM_STEPS, WARMUP_STEPS, WEIGHT_DECAY, LR_SCHEDULER, USE_RSLORA, TARGET_MODULES.

**What you CANNOT do:**

- Modify `autoresearch_unsloth/prepare_unsloth.py`. It is read-only.
- Change the fixed constants in `train_unsloth.py`: MODEL_NAME (unless switching to a model the
  human has already prepared), MAX_SEQ_LEN, EVAL_STEPS, MAX_STEPS, CACHE_DIR, DATASET_DIR, ALPACA_PROMPT.
  If the human wants to switch base models, they must first run `prepare_unsloth.py --model <name>`
  to download it, then you may update MODEL_NAME accordingly.
- Install new packages or add dependencies.
- Change the evaluation metric or output format.

**The goal is simple: get the lowest eval_loss** (cross-entropy on the held-out Alpaca eval set,
lower is better).

**The search space** — hyperparameters to explore, guided by the Unsloth LoRA guide:

| Parameter         | Typical range / options                          |
|-------------------|--------------------------------------------------|
| LORA_R            | 4, 8, 16, 32, 64, 128                            |
| LORA_ALPHA        | r, 2*r (rule: keep alpha/r >= 1)                 |
| LEARNING_RATE     | 5e-5 to 5e-4 (LoRA default: 2e-4)               |
| GRAD_ACCUM_STEPS  | 1, 2, 4, 8 (tune effective batch size)           |
| BATCH_SIZE        | 1, 2, 4                                          |
| WARMUP_STEPS      | 0, 10, 20, 50                                    |
| WEIGHT_DECAY      | 0.0, 0.01, 0.05, 0.1                             |
| LR_SCHEDULER      | "linear", "cosine", "cosine_with_restarts"       |
| USE_RSLORA        | False, True (True scales by alpha/sqrt(r))       |
| LORA_DROPOUT      | 0.0, 0.05, 0.1                                   |
| TARGET_MODULES    | add/remove modules (e.g. drop MLP projections)   |

**VRAM** is a soft constraint. Some increase is acceptable for meaningful eval_loss gains.

**Simplicity criterion**: When two configurations produce similar eval_loss, prefer the simpler one
(fewer modified hyperparameters, smaller r). A marginal gain that makes the config harder to interpret
is not worth it.

**The first run**: Always establish the baseline first — run with the default hyperparameters as-is.

## Output format

When the script finishes it prints:

```
---
eval_loss:        1.234567
perplexity:       3.437
peak_vram_mb:     12345.0
training_seconds: 123.4
num_steps:        200
lora_r:           16
lora_alpha:       32
learning_rate:    2.00e-04
effective_batch:  8
use_rslora:       False
```

Extract the key metric from the log:

```
grep "^eval_loss:\|^peak_vram_mb:" run.log
```

## Logging results

Log experiments to `autoresearch_unsloth/results_unsloth.tsv` (tab-separated, NOT comma-separated).

Header and 5 columns:

```
commit	eval_loss	vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. eval_loss achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak VRAM in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried

Example:

```
commit	eval_loss	vram_gb	status	description
a1b2c3d	1.234567	4.2	keep	baseline r=16 alpha=32 lr=2e-4
b2c3d4e	1.198432	4.2	keep	increase lr to 3e-4
c3d4e5f	1.245000	4.2	discard	switch to linear scheduler
d4e5f6g	0.000000	0.0	crash	r=128 alpha=256 OOM
```

Do not commit `results_unsloth.tsv` — leave it untracked by git.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch-unsloth/mar14`).

LOOP FOREVER:

1. Review the current git state and results so far.
2. Formulate a hypothesis: pick ONE hyperparameter (or a small related set) to change, justify it
   using the Unsloth LoRA guide and observed trends in results_unsloth.tsv.
3. Edit the hyperparameters section of `autoresearch_unsloth/train_unsloth.py`.
4. `git commit`
5. Run: `uv run autoresearch_unsloth/train_unsloth.py > run.log 2>&1`
6. Read results: `grep "^eval_loss:\|^peak_vram_mb:" run.log`
7. If grep is empty, the run crashed. Run `tail -n 50 run.log` to inspect the error.
8. Log the result in `results_unsloth.tsv`.
9. If eval_loss improved (lower), keep the commit and advance the branch.
10. If eval_loss is equal or worse, `git reset --soft HEAD~1` to discard.

**Timeout**: Each experiment typically finishes in a few minutes. If a run exceeds 15 minutes,
kill it and treat it as a crash.

**Crashes**: Fix obvious issues (typos, missing imports) and re-run. If the idea is fundamentally
broken (OOM, invalid config), log as "crash" and move on.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the human whether to continue. You are
autonomous. If you run out of obvious ideas, consult the LoRA guide again, look for patterns in the
results (e.g. lr sensitivity, rank vs quality tradeoff), try combinations of previously successful
changes, or explore the extremes of the search space. The loop runs until the human interrupts you.
