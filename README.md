# autoresearch

Autonomous LLM research loops — give an AI agent a training setup and let it experiment overnight.
This repo contains two independent modules, each a self-contained autonomous search loop.

## autoresearch (Karpathy's original autoresearch framework)

An agent edits `train.py` to modify GPT architecture and hyperparameters, trains
for a fixed 5-minute budget, checks if `val_bpb` improved, and repeats. See
[`autoresearch/README.md`](autoresearch/README.md) for full details.

```bash
uv sync
uv run autoresearch/prepare.py        # one-time: download data, train tokenizer (~2 min)
uv run autoresearch/train.py          # verify setup (~5 min)
```

**Requirements:** Single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

## autoresearch-unsloth (LoRA hyperparameter search)

An agent edits the hyperparameters section of `train_unsloth.py` — LoRA rank, learning rate,
scheduler, and more — trains for 200 gradient steps on the Alpaca instruction-following task,
checks if `eval_loss` improved, and repeats.

```bash
uv sync --extra unsloth
uv run autoresearch_unsloth/prepare_unsloth.py          # one-time: download model + dataset (~5 min)
uv run autoresearch_unsloth/train_unsloth.py            # verify setup (~2-3 min)
```

**Requirements:** Single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

Pass any HuggingFace model to the prepare script:

```bash
uv run autoresearch_unsloth/prepare_unsloth.py --model unsloth/Llama-3.2-1B-Instruct
```

After downloading, update `MODEL_NAME` in `train_unsloth.py`. Models are cached under
`~/.cache/autoresearch_unsloth/models/<model-slug>/`.

## Running the agent

For either module, spin up Claude Code (or any coding agent), then prompt:

```
Have a look at autoresearch/program.md and let's kick off a new experiment.
# or
Have a look at autoresearch_unsloth/program_unsloth.md and let's kick off a new experiment.
```

## Visualizing progress

```bash
uv run autoresearch_unsloth/plot_progress.py
# outputs autoresearch_unsloth/progress_unsloth.png
```

## References

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (original)
- [unslothai/unsloth](https://docs.unsloth.ai/get-started/unsloth-notebooks)

## License

MIT
