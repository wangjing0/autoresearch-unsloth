# autoresearch

Autonomous LLM research loops — give an AI agent a training setup and let it explore the hyperparameter space.
This repo contains three independent modules, each a self-contained autonomous search loop.

## autoresearch (Karpathy's original autoresearch framework)

An agent edits `train.py` to modify GPT architecture and hyperparameters, trains
for a fixed 5-minute budget, checks if `val_bpb` , aka model-agnostic perplexity, has improved, and repeats. See
[`autoresearch/README.md`](autoresearch/README.md) for full details.

```bash
uv sync
uv run autoresearch/prepare.py        # one-time: download data, train tokenizer (~2 min)
uv run autoresearch/train.py          # verify setup (~5 min)
```

**Requirements:** Single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

## autoresearch-unsloth (LoRA hyperparameter search)

An agent edits the hyperparameters section of `train_unsloth.py` — LoRA rank, learning rate, base model,
scheduler, and more — trains for 200 gradient steps on the designed benchmark task,
checks if `eval_loss`, or your custom defined metric on the evaluation set, has improved, and repeats.

```bash
uv sync --extra unsloth
uv run autoresearch_unsloth/prepare_unsloth.py   # one-time: download dataset (~1 min)
uv run autoresearch_unsloth/train_unsloth.py     # verify setup; downloads model on first run if not already cached
```

**Requirements:** Single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

Base models are downloaded automatically on first use by `train_unsloth.py`, provided
enough disk space and VRAM are available. The download respects the `HF_TOKEN` environment variable for gated models.
Models are cached under
`~/.cache/autoresearch_unsloth/models/<model-slug>/`.

## autoresearch-skills (diagram prompt optimization)

An agent edits `train.py` to optimize text-to-image prompts for generating technical diagrams.
The system generates diagrams via Gemini, evaluates them with Claude vision on 6 graded criteria
(text quality, color palette, layout, label discipline, visual clarity, icon quality), and uses
Pareto frontier optimization to maintain a diverse set of non-dominated prompts. The human
defines evaluation criteria in `prepare.py`; the agent searches for the best prompt strategies
in `train.py`.

```bash
uv sync
# set GOOGLE_API_KEY and ANTHROPIC_API_KEY in .env
uv run python autoresearch_skills/train.py --once    # single cycle (~2 min)
uv run python autoresearch_skills/train.py --cycles 5 # run 5 cycles
uv run python autoresearch_skills/dashboard.py        # live dashboard at localhost:8501
uv run python autoresearch_skills/train.py --reset    # reset all state
```

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), Google API key (Gemini), Anthropic API key (Claude). No GPU required.

The optimization uses Pareto frontier search across 6 criteria (each scored 0-10, overall 0-10),
LLM-generated adversarial topics that stress-test the weakest criterion, and two mutation modes
(REFINE for incremental improvement, EXPLORE for radical restructuring when scores plateau).
See [`autoresearch_skills/program.md`](autoresearch_skills/program.md) for full details.

## Running the agent

For any module, spin up Claude Code (or any coding agent), then prompt:

```
Have a look at autoresearch/program.md and let's kick off a new experiment.
# or
Have a look at autoresearch_unsloth/program_unsloth.md and let's kick off a new experiment.
# or
Have a look at autoresearch_skills/program.md and let's kick off a new experiment.
```

## Visualizing progress

```bash
uv run autoresearch_unsloth/plot_progress.py          # output progress_unsloth.png
uv run python autoresearch_skills/dashboard.py         # live web dashboard at localhost:8501
```

## References

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (original)
- [unslothai/unsloth](https://docs.unsloth.ai/get-started/unsloth-notebooks)

## License

MIT
