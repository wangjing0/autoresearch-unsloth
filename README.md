# autoresearch-unsloth

Autonomous LoRA hyperparameter search built on [Unsloth](https://github.com/unslothai/unsloth).
Give an AI agent a fine-tuning setup and let it search overnight. It modifies hyperparameters,
trains for a fixed step budget, checks if eval_loss improved, keeps or discards, and repeats.
Wake up to a log of experiments and (hopefully) a well-tuned LoRA adapter.

This is a companion to the root [autoresearch](../README.md) project, which does the same
autonomous loop for pretraining architecture search. Here the agent searches LoRA hyperparameters
instead of model architecture.

## How it works

Three files matter:

- **`prepare_unsloth.py`** — one-time setup: downloads a HuggingFace model and prepares the
  Alpaca dataset. Accepts any model via `--model`. Not modified by the agent.
- **`train_unsloth.py`** — the single file the agent edits. Contains the LoRA configuration,
  optimizer settings, and training loop. Everything in the hyperparameters section is fair game.
- **`program_unsloth.md`** — instructions for the agent. Point your agent here and let it go.
- **`plot_progress.py`** — visualizes experiment history from `results_unsloth.tsv`, producing
  a chart of kept vs discarded runs with a running-best line. Run any time during or after a session.

Each experiment runs for a **fixed budget of 200 gradient steps**, making runs directly comparable
regardless of what the agent changes. The metric is **eval_loss** (validation cross-entropy on
the held-out Alpaca split) — lower is better.

## Quick start

**Requirements:** A single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install unsloth dependencies
uv sync --extra unsloth
# or if uv can't resolve the GPU-specific wheel:
pip install unsloth trl datasets peft transformers accelerate bitsandbytes

# 2. Download model and prepare dataset (one-time, ~5 min)
uv run autoresearch_unsloth/prepare_unsloth.py

# 3. Run a single fine-tuning experiment to verify setup (~2-3 min)
uv run autoresearch_unsloth/train_unsloth.py
```

## Custom base model

Pass any HuggingFace model repo ID to the prepare script:

```bash
uv run autoresearch_unsloth/prepare_unsloth.py --model unsloth/Llama-3.2-1B-Instruct
uv run autoresearch_unsloth/prepare_unsloth.py --model unsloth/Llama-3.2-3B-Instruct

# gated models (pass token or set HF_TOKEN env var)
uv run autoresearch_unsloth/prepare_unsloth.py --model meta-llama/Llama-3.1-8B-Instruct --token hf_...
```

Each model is cached under `~/.cache/autoresearch_unsloth/models/<model-slug>/`. The dataset
is shared across all models. After downloading, update `MODEL_NAME` in `train_unsloth.py` to
point to the new model and the correct cache directory is resolved automatically.

## Running the agent

Spin up Claude Code (or any coding agent) in this repo, then prompt:

```
Have a look at autoresearch_unsloth/program_unsloth.md and let's kick off a new experiment.
```

## Project structure

```
prepare_unsloth.py   — model download + dataset prep (do not modify)
train_unsloth.py     — LoRA config + training loop (agent modifies this)
program_unsloth.md   — agent instructions
```

Cache layout under `~/.cache/autoresearch_unsloth/`:

```
models/
  unsloth--Qwen2.5-0.5B-Instruct/   (model weights, one dir per model)
  unsloth--Llama-3.2-1B-Instruct/
dataset/
  train/                             (shared across all models)
  eval/
```

## References
### Karpathy's autoresearch
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (original)

### Unsloth
- [unslothai/unsloth](https://docs.unsloth.ai/get-started/unsloth-notebooks) (original)

## License

MIT