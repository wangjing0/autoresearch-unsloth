# autoresearch-diagrams

This is an experiment to have the LLM optimize its own diagram generation prompts.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar19`). The branch `autoresearch/<tag>` must not already exist -- this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` -- fixed constants, eval criteria, eval function, topics, state helpers. Do not modify.
   - `train.py` -- the file you modify. Generation, mutation template, main optimization loop.
   - `dashboard.py` -- live web dashboard. Read-only context, not part of the optimization.
4. **Verify environment**: Check that `.env` contains `NANO_BANANA_API_KEY` and `ANTHROPIC_API_KEY`. If not, tell the human to set them up.
5. **Verify data dir**: Check that `data/prompt.txt` exists with a seed prompt. If starting fresh, the human should provide one.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment cycle takes ~2 minutes. The script generates 10 diagrams with the current prompt (Gemini image gen), evaluates each against 4 criteria via Claude vision, keeps the prompt if it beats the best score, then mutates it for the next cycle. You launch it as: `python3 train.py`

**What you CAN do:**
- Modify `train.py` -- this is the only file you edit. Everything is fair game: the mutation template, generation strategy, mutation logic, cycle structure, batch handling, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation function (`evaluate_one`), scoring (`score_batch`), eval criteria, topics, and all constants.
- Install new packages or add dependencies.
- Modify the evaluation criteria. The `EVAL_PROMPT` and `evaluate_one` function in `prepare.py` are the ground truth metric.

**The goal is simple: get the highest score out of 40.** Each batch of 10 diagrams is scored on 4 binary criteria (legible text, pastel colors, linear layout, no numbers), for a maximum of 40 points. Everything in `train.py` is fair game: change the mutation strategy, the generation prompt structure, how failures are analyzed, how the prompt evolves.

**Cost** is a soft constraint. Each cycle costs ~$0.40-0.60. Radical changes that dramatically increase API calls should be justified by meaningful score improvements.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline -- run `python3 train.py --once` with the existing prompt as-is.

## Output format

Each cycle prints a summary like this:

```
SCORE: 32/40
    Legible:    8/10
    Pastel:     9/10
    Linear:     7/10
    No numbers: 8/10
```

Results are also logged to `data/results.jsonl` as append-only JSONL entries with per-criterion breakdowns.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar19`).

LOOP FOREVER:

1. Look at the current state: best score so far, current prompt, recent results in `data/results.jsonl`.
2. Form a hypothesis about what change to `train.py` might improve scores (mutation template wording, generation strategy, failure analysis approach, etc).
3. Edit `train.py` with the change.
4. git commit.
5. Run a cycle: `python3 train.py --once > run.log 2>&1`
6. Read out the results: `grep "SCORE:" run.log` and check `data/state.json` for the best score.
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
8. If the score improved, keep the commit and advance.
9. If the score is equal or worse, git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. You're advancing the branch so that you can iterate.

**Crashes**: If a run crashes, use your judgment. If it's a typo or missing import, fix and re-run. If the idea is fundamentally broken, discard and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder -- re-read the eval criteria for new angles, try different mutation strategies, experiment with prompt structure, try combining previous near-misses. The loop runs until the human interrupts you, period.

## Eval Criteria (fixed in prepare.py)

Each diagram is evaluated on 4 binary pass/fail criteria:

1. **Legible & grammatical** -- all text readable, correctly spelled, grammatically correct
2. **Pastel colors** -- soft pastel fills only, no saturated/dark colors
3. **Linear layout** -- strictly left-to-right or top-to-bottom flow
4. **No numbers** -- zero digits, ordinals, or step numbers anywhere

Score = sum of passes across 10 diagrams x 4 criteria = max 40.

## Models

- **Generation**: `gemini-2.5-flash-image` (Gemini native image gen)
- **Evaluation**: `claude-sonnet-4-6` (vision + structured JSON output)
- **Mutation**: `claude-sonnet-4-6` (prompt rewriting based on failure analysis)

## File Structure

```
prepare.py            # Fixed eval harness, constants, topics (do not modify)
train.py              # Generation, mutation, main loop (agent iterates on this)
dashboard.py          # Live web dashboard (Chart.js)
program.md            # This file -- agent instructions
data/
  prompt.txt          # Current prompt being optimized
  best_prompt.txt     # Best prompt found so far
  state.json          # Loop state (run number, best score)
  results.jsonl       # Append-only experiment log
  diagrams/
    run_001/          # 10 diagrams per run
    run_002/
    ...
```
