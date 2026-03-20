# autoresearch-skills

Autonomous skill prompt optimization using Pareto frontier search. The agent iterates
to find the best generation prompts across 6 evaluation criteria, switching to exploratory
mutations when progress stalls.

## Division of Responsibility

This system has a clear split between what the **human** defines and what the **agent**
searches for autonomously.

### Human defines (in prepare.py -- do not modify)

The human decides *what good looks like*. This means writing the evaluation criteria and
the eval prompt that judges generated diagrams. These are fixed before the agent starts
and do not change during a run.

- **EVAL_PROMPT** in `prepare.py` -- the exact prompt sent to Claude vision to judge each
  diagram. Defines 4 binary pass/fail criteria: legible text, pastel colors, linear layout,
  no numbers. This is the ground truth. The agent never sees or modifies it.
- **TOPICS** in `prepare.py` -- the base set of 30 diagram topics used for generation.
  These define the diversity of test cases the system evaluates against.
- **evaluate_one()** and **score_batch()** in `prepare.py` -- the evaluation harness that
  calls Claude vision and computes scores. The scoring logic is fixed.
- **Constants** in `prepare.py` -- models (GEN_MODEL, EVAL_MODEL, MUTATE_MODEL), batch size,
  cycle timing, worker counts. These set the operational constraints.
- **data/prompt.txt** -- the initial seed prompt that tells Gemini how to generate images.
  The human writes the starting point; the agent evolves it from there.

If you want to change what the system optimizes for (different criteria, different generation
style, different scoring), you modify `prepare.py` and provide a new seed prompt. Then
reset and let the agent run again.

### Agent searches for (in train.py -- the only file the agent edits)

The agent decides *how to get there*. It controls the entire optimization strategy: how
prompts are mutated, how the search explores vs exploits, how failures are diagnosed, and
how the Pareto frontier is managed.

- **Mutation templates** (REFINE_TEMPLATE, EXPLORE_TEMPLATE) -- the instructions given to
  Claude for rewriting prompts based on evaluation feedback.
- **Bottleneck focus** (BOTTLENECK_FOCUS) -- per-criterion strategies when one dimension
  is the clear weak point.
- **Adversarial topic generation** (TOPIC_GEN_TEMPLATE, STRESS_INSTRUCTIONS) -- how Claude
  generates stress-test topics that target the weakest criterion.
- **Pareto frontier logic** (select_parent, find_weakest_criterion, dominates, update_frontier)
  -- how the population of prompts is managed and parents are selected.
- **Optimization config** (PLATEAU_WINDOW, EARLY_STOP_WINDOW, ADVERSARIAL_TOPIC_COUNT) --
  thresholds that control when to switch modes and when to stop.
- **Cycle structure** (run_cycle) -- the generate/evaluate/score/frontier/mutate pipeline.

The agent's job is to find the prompt that maximizes the score defined by the human's
evaluation criteria. It does this by iterating on `train.py` -- changing mutation strategies,
tuning thresholds, trying different approaches to prompt evolution.

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
| ADVERSARIAL_TOPIC_COUNT   | optimization config         | Number of LLM-generated stress-test topics per batch |
| REFINE_TEMPLATE           | mutation templates          | Prompt given to Claude for incremental mutations  |
| EXPLORE_TEMPLATE          | mutation templates          | Prompt given to Claude for radical restructuring  |
| BOTTLENECK_FOCUS          | mutation templates          | Per-criterion focused instructions when one criterion dominates |
| TOPIC_GEN_TEMPLATE        | topic generation            | Prompt for generating adversarial topics  |
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

**Plateau detection**: After PLATEAU_WINDOW consecutive runs without improvement, the system
switches from REFINE to EXPLORE mode for radical prompt restructuring.

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


**Plateau detection**: When running multi-cycle (`--cycles N` or continuous), the system
switches to EXPLORE mode after PLATEAU_WINDOW (default 3) consecutive runs without
improvement. The agent should interpret persistent plateaus as a signal to change strategy
in `train.py` before restarting.

**Reset**: To start fresh, run `uv run python autoresearch_skills/train.py --reset`. This clears
results, state, frontier, best_prompt, and output files while preserving the seed prompt.

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

## Eval Criteria (fixed in prepare.py -- human-defined)

Each diagram is evaluated on 4 binary pass/fail criteria. These are written by the human
in `EVAL_PROMPT` inside `prepare.py` and cannot be changed by the agent:

1. **Legible & grammatical** -- all text readable, correctly spelled, grammatically correct
2. **Pastel colors** -- soft pastel fills only, no saturated/dark colors
3. **Linear layout** -- strictly left-to-right or top-to-bottom flow
4. **No numbers** -- zero digits, ordinals, or step numbers anywhere

Score = sum of passes across 10 diagrams x 4 criteria = max 40.

To change what the system optimizes for, the human modifies `EVAL_PROMPT` and the criteria
in `prepare.py`, provides a new seed prompt in `data/prompt.txt`, resets with `--reset`,
and lets the agent run again.

## Models

- **Generation**: `gemini-2.5-flash-image` (Gemini native image gen, fixed in prepare.py)
- **Evaluation**: `claude-sonnet-4-6` (vision + structured JSON output, fixed in prepare.py)
- **Mutation**: `claude-sonnet-4-6` (prompt rewriting + adversarial topic generation, fixed in prepare.py)

## File Structure

```
autoresearch_skills/
  prepare.py            # Fixed eval harness, constants, topics (human defines -- do not modify)
  train.py              # Pareto frontier optimization (agent searches -- iterates on this)
  dashboard.py          # Live web dashboard (read-only)
  program.md            # This file -- agent instructions
  __init__.py           # Package init
  tests/
    test_suite.py       # Test suite
```
