# autoresearch

Autonomous LLM research loops -- give an AI agent a training setup and let it explore the hyperparameter space. The repo contains three independent modules that each follow the same pattern: an agent edits a training script, runs an experiment under a fixed budget, evaluates the result against a target metric, and repeats.

All three modules share the same prerequisites: [uv](https://docs.astral.sh/uv/). GPU and API key requirements vary by module.

## Modules

### autoresearch -- GPT architecture search

Based on [Karpathy's original framework](https://github.com/karpathy/autoresearch). The agent modifies GPT architecture and hyperparameters in `train.py`, trains for a fixed 5-minute budget, and checks whether `val_bpb` (model-agnostic perplexity) has improved. See [`autoresearch/README.md`](autoresearch/README.md) for full details.

```bash
uv sync
uv run python -m autoresearch.prepare        # one-time: download data, train tokenizer (~2 min)
uv run python -m autoresearch.train          # verify setup (~5 min)
```

Requires a single NVIDIA GPU (tested on H100).

### autoresearch-unsloth -- LoRA hyperparameter search

The agent edits the hyperparameters section of `train_unsloth.py` -- LoRA rank, learning rate, base model, scheduler, and more -- then trains for 200 gradient steps and checks whether `eval_loss` (or a custom metric) has improved.

```bash
uv sync --extra unsloth
uv run python -m autoresearch_unsloth.prepare_unsloth   # one-time: download dataset (~1 min)
uv run python -m autoresearch_unsloth.train_unsloth     # verify setup; downloads model on first run
```

Requires a single NVIDIA GPU. Base models are downloaded automatically on first use and cached under `~/.cache/autoresearch_unsloth/models/<model-slug>/`. Set `HF_TOKEN` for gated models.

### autoresearch-skills -- prompt optimization

The agent optimizes text-to-image prompts in `train.py` for generating images that satisfy a set of graded criteria. Images are generated via Gemini and evaluated by Claude vision on 6 dimensions (text quality, color palette, layout, label discipline, visual clarity, icon quality). The search uses Pareto front optimization to maintain a diverse set of non-dominated prompts, with two mutation modes -- REFINE for incremental improvement and EXPLORE for radical restructuring when a plateau is detected. Additional features include a feedback history that records past mutation outcomes so the mutator avoids repeating failed strategies, round-robin topic sampling for systematic coverage across all 30 test topics, and a mutation fallback with progressive context truncation for robustness. See [`autoresearch_skills/program.md`](autoresearch_skills/program.md) for full details.

```bash
uv sync
# set GOOGLE_API_KEY and ANTHROPIC_API_KEY in .env
uv run python -m autoresearch_skills.train              # continuous optimization loop
uv run python -m autoresearch_skills.dashboard          # live dashboard at localhost:8501
```

No GPU required. Needs a Google API key (Gemini) and an Anthropic API key (Claude).

## Running the agent

For any module, open Claude Code (or any coding agent) and prompt:

```
Have a look at autoresearch/program.md and let's kick off a new experiment.
```


## Visualizing progress

```bash
uv run python -m autoresearch_unsloth.plot_progress    # outputs progress_unsloth.png
uv run python -m autoresearch_skills.dashboard          # live web dashboard at localhost:8501
```

## References

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (original)
- [unslothai/unsloth](https://docs.unsloth.ai/get-started/unsloth-notebooks)
- [Pareto front](https://en.wikipedia.org/wiki/Pareto_front)

## License

MIT
