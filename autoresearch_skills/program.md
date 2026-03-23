# autoresearch-skills

Autonomous skill prompt optimization using Pareto front search. The agent iterates
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
  diagram. Defines 6 graded criteria (text quality, color palette, layout, label discipline,
  visual clarity, icon quality) scored 1-5 each. This is the ground truth. The agent never
  sees or modifies it.
- **TOPICS** in `prepare.py` -- the base set of 30 diagram topics used for generation.
  These define the diversity of test cases the system evaluates against.
- **evaluate_one()** and **score_batch()** in `prepare.py` -- the evaluation harness that
  calls Claude vision and computes scores. The scoring logic is fixed.
- **Constants** in `prepare.py` -- models (GEN_MODEL, EVAL_MODEL, MUTATE_MODEL), batch size,
  cycle timing, worker counts. These set the operational constraints.
- **INITIAL_PROMPT** in `prepare.py` -- the initial seed prompt that tells Gemini how to generate
  diagrams. Used as baseline on run 1. The human defines the starting point; the agent evolves
  it from there. The current working prompt is stored in `state/prompt.txt`.

If you want to change what the system optimizes for (different criteria, different generation
style, different scoring), you modify `prepare.py` and provide a new seed prompt. Then
reset and let the agent run again.

### Agent searches for (in train.py -- the only file the agent edits)

The agent decides *how to get there*. It controls the entire optimization strategy: how
prompts are mutated, how the search explores vs exploits, how failures are diagnosed, and
how the Pareto front is managed.

- **Mutation templates** (REFINE_TEMPLATE, EXPLORE_TEMPLATE) -- the instructions given to
  Claude for rewriting prompts based on evaluation feedback. Both templates include a
  generalization rule requiring changes to work across all 30+ topics (not just patch
  specific failures), and inject the feedback history so the mutator avoids repeating
  strategies that already failed.
- **Bottleneck focus** (BOTTLENECK_FOCUS) -- per-criterion strategies when one dimension
  is the clear weak point.
- **Adversarial topic generation** (TOPIC_GEN_TEMPLATE, STRESS_INSTRUCTIONS) -- how Claude
  generates stress-test topics that target the weakest criterion.
- **Feedback history** (append_feedback, read_feedback_history) -- after each cycle, the
  mutation outcome (run, mode, weakest criterion, score delta, front membership) is
  written to `state/feedback_history.jsonl`. The last 10 entries are injected into the
  mutation templates so the LLM can see what was tried and avoid repeating dead-end strategies.
- **Pareto front logic** (select_parent, find_weakest_criterion, dominates, update_frontier)
  -- how the population of prompts is managed and parents are selected.
- **Optimization config** (PLATEAU_WINDOW, ADVERSARIAL_TOPIC_COUNT) --
  thresholds that control when to switch modes.
- **Cycle structure** (run_cycle) -- the generate/evaluate/score/frontier/mutate pipeline.
  Topic selection uses round-robin cycling through all 30 base topics (via `topic_offset`
  in state.json), guaranteeing full coverage every ~4 cycles instead of random sampling.

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
   - `autoresearch_skills/train.py` -- the file you modify. Pareto front logic, mutation templates,
     adversarial topic generation, optimization hyperparameters, early stopping, and the main loop.
   - `autoresearch_skills/program.md` -- this file. Read it fully before starting.
4. **Verify environment**: Check that `.env` contains `GOOGLE_API_KEY` and `ANTHROPIC_API_KEY`.
   If not, tell the human to set them up.
5. **Verify state dir**: Check that `autoresearch_skills/state/prompt.txt` exists with a seed prompt.
   The initial seed prompt is defined as `INITIAL_PROMPT` in `prepare.py`.
   If starting fresh, the human should provide one in `state/prompt.txt`.
6. **Initialize**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment cycle takes ~2 minutes. The script generates 10 diagrams with the current
prompt (Gemini image gen), evaluates each against 4 criteria via Claude vision, updates the
Pareto front, then mutates the prompt for the next cycle. You launch it as:

```
uv run python -m autoresearch_skills.train > run.log 2>&1
```

**What you CAN do:**

- Modify `autoresearch_skills/train.py` -- this is the only file you edit. All Pareto front parameters,
  mutation templates, adversarial topic generation, plateau detection, early stopping, and cycle structure
  are fair game.

**What you CANNOT do:**

- Modify `autoresearch_skills/prepare.py`. It is read-only. It contains the fixed evaluation function
  (`evaluate_one`), scoring (`score_batch`), eval criteria (`EVAL_PROMPT`), base topics, and all constants.
- Install new packages or add dependencies.
- Modify the evaluation criteria. The `EVAL_PROMPT` and `evaluate_one` function in `prepare.py`
  are the ground truth metric.

**The goal is simple: get the highest overall score out of 10.00.** Each diagram is scored on
6 graded criteria (1-5 each), normalized to 0-10 per criterion. Overall = mean of all 6.

**The search space** -- parameters to explore in `train.py`:

| Parameter                 | Location in train.py        | Description                                      |
|---------------------------|-----------------------------|--------------------------------------------------|
| PLATEAU_WINDOW            | optimization config         | Runs without improvement before switching to EXPLORE mode |
| ADVERSARIAL_TOPIC_COUNT   | optimization config         | Number of LLM-generated stress-test topics per batch |
| REFINE_TEMPLATE           | mutation templates          | Prompt given to Claude for incremental mutations (includes anti-overfitting rule and feedback history) |
| EXPLORE_TEMPLATE          | mutation templates          | Prompt given to Claude for radical restructuring (includes anti-repetition text and feedback history) |
| BOTTLENECK_FOCUS          | mutation templates          | Per-criterion focused instructions when one criterion dominates |
| TOPIC_GEN_TEMPLATE        | topic generation            | Prompt for generating adversarial topics  |
| STRESS_INSTRUCTIONS       | topic generation            | Per-criterion stress-test strategies              |
| append_feedback()         | feedback history            | Records mutation outcome to state/feedback_history.jsonl after each cycle |
| read_feedback_history()   | feedback history            | Reads last N entries from feedback history for injection into mutation templates |
| select_parent()           | frontier logic              | How parents are chosen from the Pareto front   |
| find_weakest_criterion()  | frontier logic              | How the weakest dimension is identified           |
| _mutate_with_fallback()   | mutation resilience         | Retries mutate_prompt with progressively reduced context (20→10→5 failures, 5→3→0 frontier members) if output is unusable |

**Cost** is a soft constraint. Each cycle costs ~$0.50-0.80. Radical changes that dramatically
increase API calls should be justified by meaningful score improvements.

**Simplicity criterion**: When two configurations produce similar scores, prefer the simpler one.
A marginal gain that makes the code harder to understand is not worth it.

**The first run**: Always establish the baseline first -- run with the existing prompt and
default parameters as-is.

## Optimization Architecture

The system uses **Pareto front optimization** across 6 graded criteria instead of greedy
hill-climbing on a single scalar score.

**Pareto front** (`state/frontier.jsonl`): Non-dominated prompts. Prompt A dominates B if A is
better on at least one criterion and no worse on all others. Maintains diversity -- prompts with
different trade-offs coexist.

**Parent selection**: Each cycle selects a parent from the frontier, weighted toward prompts strong
on the overall weakest criterion.

**Adversarial topics**: Claude generates stress-test topics targeting the weakest criterion.
Mixed with standard topics from `prepare.py` for each batch. Standard topics are selected
via round-robin cycling through all 30 base topics (offset tracked in `state.json`), ensuring
full coverage every ~4 cycles instead of random sampling with potential blind spots.

**Two mutation modes**: REFINE (incremental, default) and EXPLORE (radical restructuring,
triggered after PLATEAU_WINDOW consecutive cycles without improvement).

**Anti-overfitting guidance**: Both mutation templates instruct the mutator that every change
must generalize across all 30+ diverse topics, not patch specific failure cases. EXPLORE_TEMPLATE
also warns against repeating previously attempted restructuring approaches.

**Feedback history** (`state/feedback_history.jsonl`): After each cycle, the outcome (run number,
mode, weakest criterion, score delta, whether added to the front) is appended. The last 10
entries are injected into the mutation prompt via `{feedback_history}`, giving the LLM awareness
of what was already tried so it can explore genuinely new directions.

**Mutation resilience** (`_mutate_with_fallback`): Wraps `mutate_prompt` with three truncation
levels. If the returned prompt is empty, under 50 characters, or identical to the input, retries
with progressively less context: 20→10→5 failures, 5→3→0 frontier members. Falls back to the
current prompt unchanged if all levels fail.

**Bottleneck focus**: When one criterion is clearly the weakest (others at 9+), mutations focus
exclusively on strategies for that dimension.

**Plateau detection**: After PLATEAU_WINDOW consecutive runs without improvement, the system
switches from REFINE to EXPLORE mode for radical prompt restructuring.

## Output format

Each cycle prints:

```
RUN 7 | 15:33:08 | Best: 7.50/10 | Mode: EXPLORE | Weakest: text_quality
  Frontier size: 3 | Adversarial topics: 3

  SCORE: 6.83/10
    Text Quality         5.00/10
    Color Palette        8.75/10
    Layout               7.50/10
    Label Discipline     6.25/10
    Visual Clarity       7.50/10
    Icon Quality         6.00/10

  FRONTIER: Added (frontier size: 4)
  Bottleneck: text_quality
  Mutating prompt (EXPLORE mode)...
```

Extract key metrics:

```
grep "SCORE:\|FRONTIER:\|Mode:" run.log
```

## Dashboard

Start the live dashboard to monitor progress:

```
uv run python -m autoresearch_skills.dashboard --port 8501
```

The dashboard shows: best score, baseline, improvement percentage, frontier size, current
weakest criterion, mutation mode (REFINE/EXPLORE), score-over-time chart with mode-colored
dots, Pareto front member cards with per-criterion breakdowns, run history table with
mode/weakest/frontier columns, and initial vs best prompt comparison. Auto-refreshes every 15s.


**Plateau detection**: When running multi-cycle (`--cycles N` or continuous), the system
switches to EXPLORE mode after PLATEAU_WINDOW (default 3) consecutive runs without
improvement. The agent should interpret persistent plateaus as a signal to change strategy
in `train.py` before restarting.

**Reset**: To start fresh, run `uv run python -m autoresearch_skills.train --reset`. This clears
results, state (including `topic_offset`), frontier, best_prompt, feedback history, and output
files while preserving the seed prompt.

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

Each diagram is evaluated on 6 graded criteria (1-5 scale). These are written by the human
in `EVAL_PROMPT` inside `prepare.py` and cannot be changed by the agent:

1. **Text Quality** (1-5) -- legibility, spelling, spacing, consistent sizing
2. **Color Palette** (1-5) -- harmonious soft pastel colors with deliberate color coding
3. **Layout** (1-5) -- linear flow with uniform spacing and clean arrows
4. **Label Discipline** (1-5) -- concise 2-4 word labels, no numbers/ordinals, consistent style
5. **Visual Clarity** (1-5) -- publication-quality composition, consistent styling, good whitespace
6. **Icon Quality** (1-5) -- clear, relevant, consistently styled line-art icons in every box

Each criterion's raw average (1-5) is mapped to 0-10 via `(avg - 1) / 4 * 10`.
Overall score = mean of all 6 normalized scores, max 10.00.

To change what the system optimizes for, the human modifies `EVAL_PROMPT` and the criteria
in `prepare.py`, provides a new seed prompt in `state/prompt.txt`, resets with `--reset`,
and lets the agent run again.

## Models

- **Generation**: `gemini-2.5-flash-image` (Gemini native image gen, fixed in prepare.py)
- **Evaluation**: `claude-sonnet-4-6` (vision + structured JSON output, fixed in prepare.py)
- **Mutation**: `claude-sonnet-4-6` (prompt rewriting + adversarial topic generation, fixed in prepare.py)

## File Structure

```
autoresearch_skills/
  prepare.py            # Fixed eval harness, constants, topics (human defines -- do not modify)
  train.py              # Pareto front optimization (agent searches -- iterates on this)
  dashboard.py          # Live web dashboard (read-only)
  program.md            # This file -- agent instructions
  __init__.py           # Package init
  tests/
    test_suite.py       # Test suite
  state/
    state.json          # run_number, best_score, plateau_streak, topic_offset
    prompt.txt          # Current working prompt (next cycle input)
    best_prompt.txt     # Prompt that achieved the all-time best score
    results.jsonl       # One entry per cycle: scores, mode, weakest, frontier_size
    frontier.jsonl      # Current Pareto front members (non-dominated prompts)
    feedback_history.jsonl  # Mutation outcomes for LLM context injection
    diagrams/           # Generated PNG diagrams, one subdirectory per run
```
