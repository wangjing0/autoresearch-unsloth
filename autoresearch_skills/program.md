# autoresearch-skills

Autonomous diagram prompt optimization using Pareto frontier search. The agent iterates
to find the best generation prompts across 4 evaluation criteria, with early stopping
when progress stalls.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar19`). The branch
   `autoresearch-skills/<tag>` must not already exist -- this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-skills/<tag>` from current master.
3. **Read the in-scope files**:
   - `autoresearch_skills/prepare.py` -- fixed constants, eval criteria, eval function, topics, state helpers. Do not modify.
   - `autoresearch_skills/train.py` -- the file you modify. Pareto frontier logic, mutation templates,
     adversarial topic generation, optimization hyperparameters, early stopping, and the main loop.
   - `autoresearch_skills/program.md` -- this file. Read it fully before starting.
4. **Verify environment**: Check that `.env` contains `GOOGLE_API_KEY` and `ANTHROPIC_API_KEY`.
   If not, tell the human to set them up.
5. **Verify data dir**: Check that `autoresearch_skills/data/prompt.txt` exists with a seed prompt.
   `data/initial_prompt.txt` stores the original seed for dashboard comparison.
   If starting fresh, the human should provide one.
6. **Initialize**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment cycle takes ~2 minutes. The script generates 10 diagrams with the current
prompt (Gemini image gen), evaluates each against 4 criteria via Claude vision, updates the
Pareto frontier, then mutates the prompt for the next cycle. You launch it as:

```
uv run python autoresearch_skills/train.py > run.log 2>&1
```

**What you CAN do:**

- Modify `autoresearch_skills/train.py` -- this is the only file you edit. All Pareto frontier parameters,
  mutation templates, adversarial topic generation, plateau detection, early stopping, and cycle structure
  are fair game.

**What you CANNOT do:**

- Modify `autoresearch_skills/prepare.py`. It is read-only. It contains the fixed evaluation function
  (`evaluate_one`), scoring (`score_batch`), eval criteria (`EVAL_PROMPT`), base topics, and all constants.
- Install new packages or add dependencies.
- Modify the evaluation criteria. The `EVAL_PROMPT` and `evaluate_one` function in `prepare.py`
  are the ground truth metric.

**The goal is simple: get the highest score out of 40.** Each batch of 10 diagrams is scored on
4 binary criteria (legible text, pastel colors, linear layout, no numbers), for a maximum of 40.

**The search space** -- parameters to explore in `train.py`:

| Parameter                 | Location in train.py        | Description                                      |
|---------------------------|-----------------------------|--------------------------------------------------|
| PLATEAU_WINDOW            | optimization config         | Runs without improvement before switching to EXPLORE mode |
| EARLY_STOP_WINDOW         | optimization config         | Consecutive runs without improvement before stopping the loop |
| ADVERSARIAL_TOPIC_COUNT   | optimization config         | Number of LLM-generated stress-test topics per batch |
| REFINE_TEMPLATE           | mutation templates          | Prompt given to Claude for incremental mutations  |
| EXPLORE_TEMPLATE          | mutation templates          | Prompt given to Claude for radical restructuring  |
| BOTTLENECK_FOCUS          | mutation templates          | Per-criterion focused instructions when one criterion dominates |
| TOPIC_GEN_TEMPLATE        | topic generation            | Prompt for generating adversarial diagram topics  |
| STRESS_INSTRUCTIONS       | topic generation            | Per-criterion stress-test strategies              |
| select_parent()           | frontier logic              | How parents are chosen from the Pareto frontier   |
| find_weakest_criterion()  | frontier logic              | How the weakest dimension is identified           |

**Cost** is a soft constraint. Each cycle costs ~$0.50-0.80. Radical changes that dramatically
increase API calls should be justified by meaningful score improvements.

**Simplicity criterion**: When two configurations produce similar scores, prefer the simpler one.
A marginal gain that makes the code harder to understand is not worth it.

**The first run**: Always establish the baseline first -- run with the existing prompt and
default parameters as-is.

## Optimization Architecture

The system uses **Pareto frontier optimization** across 4 criteria instead of greedy
hill-climbing on a single scalar score.

**Pareto frontier** (`data/frontier.jsonl`): Non-dominated prompts. Prompt A dominates B if A is
better on at least one criterion and no worse on all others. Maintains diversity -- prompts with
different trade-offs coexist.

**Parent selection**: Each cycle selects a parent from the frontier, weighted toward prompts strong
on the overall weakest criterion.

**Adversarial topics**: Claude generates stress-test topics targeting the weakest criterion.
Mixed with standard topics from `prepare.py` for each batch.

**Two mutation modes**: REFINE (incremental, default) and EXPLORE (radical restructuring,
triggered after PLATEAU_WINDOW consecutive cycles without improvement).

**Bottleneck focus**: When one criterion is clearly the weakest (others at 9+), mutations focus
exclusively on strategies for that dimension.

**Early stopping**: The loop automatically stops after EARLY_STOP_WINDOW consecutive runs
without improvement to the best score. This prevents wasting API credits when the system
has converged.

## Output format

Each cycle prints:

```
RUN 7 | 15:33:08 | Best: 38/40 | Mode: EXPLORE | Weakest: legible
  Frontier size: 3 | Adversarial topics: 3

  SCORE: 36/40
    Legible:    6/10
    Pastel:     10/10
    Linear:     10/10
    No numbers: 10/10

  FRONTIER: Added (frontier size: 4)
  Bottleneck: legible
  Mutating prompt (EXPLORE mode)...
```

Extract key metrics:

```
grep "SCORE:\|FRONTIER:\|Mode:\|Early stopping" run.log
```

## Dashboard

Start the live dashboard to monitor progress:

```
uv run python autoresearch_skills/dashboard.py --port 8501
```

The dashboard shows: best score, baseline, improvement percentage, frontier size, current
weakest criterion, mutation mode (REFINE/EXPLORE), score-over-time chart with mode-colored
dots, Pareto frontier member cards with per-criterion breakdowns, run history table with
mode/weakest/frontier columns, and initial vs best prompt comparison. Auto-refreshes every 15s.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch-skills/mar19`).

LOOP FOREVER:

1. Review the current state: `data/state.json`, `data/frontier.jsonl`, recent entries in `data/results.jsonl`.
2. Formulate a hypothesis: pick ONE aspect of `train.py` to change. Consider: mutation template
   wording, parent selection strategy, adversarial topic design, plateau detection threshold,
   early stopping window, frontier management, bottleneck focus instructions.
3. Edit `autoresearch_skills/train.py`.
4. `git commit`
5. Run: `uv run python autoresearch_skills/train.py --once > run.log 2>&1`
6. Read results: `grep "SCORE:\|FRONTIER:" run.log` and check `data/state.json`, `data/frontier.jsonl`.
7. If grep is empty, the run crashed. Run `tail -n 50 run.log` to inspect the error.
8. If the score improved or the frontier grew meaningfully, keep the commit and advance.
9. If no improvement, `git reset --soft HEAD~1` to discard.

**Early stopping**: When running multi-cycle (`--cycles N` or continuous), the loop automatically
stops after EARLY_STOP_WINDOW (default 3) consecutive runs without improvement. The agent
should interpret this as a signal to change strategy in `train.py` before restarting.

**Timeout**: Each cycle typically finishes in ~2 minutes. If a run exceeds 5 minutes, kill it
and treat it as a crash.

**Crashes**: Fix obvious issues (typos, missing imports) and re-run. If the idea is fundamentally
broken, log it and move on.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the human whether to continue. You are
autonomous. If early stopping triggers, change your approach in `train.py` and restart. If you
run out of obvious ideas, re-read the eval criteria for new angles, examine which frontier prompts
score highest on the weakest criterion and study their structure, try radically different mutation
templates, or experiment with adversarial topic strategies. The loop runs until the human
interrupts you.

## Eval Criteria (fixed in prepare.py)

Each diagram is evaluated on 4 binary pass/fail criteria:

1. **Legible & grammatical** -- all text readable, correctly spelled, grammatically correct
2. **Pastel colors** -- soft pastel fills only, no saturated/dark colors
3. **Linear layout** -- strictly left-to-right or top-to-bottom flow
4. **No numbers** -- zero digits, ordinals, or step numbers anywhere

Score = sum of passes across 10 diagrams x 4 criteria = max 40.

## Models

- **Generation**: `gemini-2.5-flash-image` (Gemini native image gen, fixed in prepare.py)
- **Evaluation**: `claude-sonnet-4-6` (vision + structured JSON output, fixed in prepare.py)
- **Mutation**: `claude-sonnet-4-6` (prompt rewriting + adversarial topic generation, fixed in prepare.py)

## File Structure

```
autoresearch_skills/
  prepare.py            # Fixed eval harness, constants, topics (do not modify)
  train.py              # Pareto frontier optimization (agent iterates on this)
  dashboard.py          # Live web dashboard (read-only)
  program.md            # This file -- agent instructions
  __init__.py           # Package init
  data/
    prompt.txt          # Current prompt being optimized
    best_prompt.txt     # Best prompt found so far (highest total score)
    initial_prompt.txt  # Original seed prompt (for dashboard comparison)
    state.json          # Loop state (run number, best score)
    results.jsonl       # Append-only experiment log (per-run scores, mode, weakest, frontier size)
    frontier.jsonl      # Pareto frontier of non-dominated prompts
    diagrams/
      run_001/          # 10 diagrams per run
      run_002/
      ...
  tests/
    test_suite.py       # Test suite
```
