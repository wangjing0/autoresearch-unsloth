#!/usr/bin/env python3
"""
Diagram Autoresearch — Self-improving prompt optimization.

This is the file the agent iterates on. Contains generation, mutation,
and the main optimization loop. The fixed evaluation harness lives in
prepare.py (do not modify).

Usage:
    python3 train.py              # Continuous loop
    python3 train.py --once       # Single cycle
    python3 train.py --cycles 5   # Run N cycles
"""

import argparse
import json
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from prepare import (
    GEMINI_KEY, ANTHROPIC_KEY,
    GEN_MODEL, EVAL_MODEL, MUTATE_MODEL,
    BASE_DIR, BEST_PROMPT_FILE, RESULTS_FILE, DIAGRAMS_DIR,
    BATCH_SIZE, CYCLE_SECONDS, MAX_GEN_WORKERS, MAX_EVAL_WORKERS,
    TOPICS,
    evaluate_one, score_batch,
    load_state, save_state, load_prompt, save_prompt,
)

# ─── Mutation Prompt ──────────────────────────────────────────────────────────

MUTATION_TEMPLATE = """You are optimizing a text-to-image prompt for generating technical diagrams. The prompt is sent to Gemini's image generation model. Your goal: modify it so generated diagrams consistently pass ALL 4 evaluation criteria.

CURRENT PROMPT:
---
{current_prompt}
---

LAST BATCH RESULTS ({score}/40):
- Legible & grammatical: {leg_rate}/10
- Pastel colors: {col_rate}/10
- Linear layout: {lin_rate}/10
- No numbers/ordinals: {num_rate}/10

COMMON FAILURES:
{failures}

BEST SCORE SO FAR: {best_score}/40

RULES FOR YOUR MODIFICATION:
- Keep the core whiteboard/hand-drawn aesthetic
- For any criterion below 8/10, add VERY explicit constraints
- If numbers keep appearing: emphasize "ABSOLUTELY NO numbers, step numbers, ordinals, sequence indicators, or numerical labels of any kind"
- If layout isn't linear: specify "MUST flow in a single straight line from left to right" or "from top to bottom"
- If text is garbled: add "All text must be real, correctly spelled English words"
- If colors aren't pastel: list exact colors to use and explicitly ban dark/saturated fills
- Be specific and imperative — image models respond to direct commands
- Keep prompt under 400 words
- Return ONLY the new prompt text — no explanation, no markdown fences"""

# ─── Generation (Gemini) ─────────────────────────────────────────────────────


def generate_one(gemini_client, prompt: str, topic: str, output_path: Path) -> bool:
    """Generate a single diagram via Gemini image generation."""
    from google.genai import types

    full_prompt = f"{prompt}\n\nDiagram to create: {topic}"
    try:
        response = gemini_client.models.generate_content(
            model=GEN_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(part.inline_data.data)
                return True
        return False
    except Exception as e:
        print(f"    GEN ERROR: {e}")
        return False


# ─── Mutation (Claude) ───────────────────────────────────────────────────────


def mutate_prompt(
    anthropic_client,
    current_prompt: str,
    eval_results: list[dict],
    best_score: int,
) -> str:
    """Use Claude to improve the prompt based on failure analysis."""
    scores = score_batch(eval_results)

    all_failures = []
    for r in eval_results:
        for f in r.get("failures", []):
            all_failures.append(f)

    unique_failures = list(dict.fromkeys(all_failures))[:20]
    failures_text = "\n".join(f"- {f}" for f in unique_failures) if unique_failures else "- None"

    mutation_prompt = MUTATION_TEMPLATE.format(
        current_prompt=current_prompt,
        score=scores["total"],
        leg_rate=scores["legible"],
        col_rate=scores["pastel"],
        lin_rate=scores["linear"],
        num_rate=scores["no_numbers"],
        best_score=best_score,
        failures=failures_text,
    )

    response = anthropic_client.messages.create(
        model=MUTATE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": mutation_prompt}],
    )
    return response.content[0].text.strip()


# ─── Main Cycle ──────────────────────────────────────────────────────────────


def run_cycle(gemini_client, anthropic_client, state: dict) -> dict:
    """Run one autoresearch optimization cycle."""
    run_num = state["run_number"] + 1
    state["run_number"] = run_num
    run_dir = DIAGRAMS_DIR / f"run_{run_num:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt = load_prompt()
    topics = random.sample(TOPICS, min(BATCH_SIZE, len(TOPICS)))

    print(f"\n{'='*60}")
    print(f"RUN {run_num} | {datetime.now().strftime('%H:%M:%S')} | Best: {state['best_score']}/40")
    print(f"{'='*60}")

    # ── Generate ──────────────────────────────────────────────────
    print(f"\n  Generating {BATCH_SIZE} diagrams...")
    generated: list[tuple[int, str, Path]] = []

    with ThreadPoolExecutor(max_workers=MAX_GEN_WORKERS) as pool:
        futures = {}
        for i, topic in enumerate(topics):
            out = run_dir / f"diagram_{i:02d}.png"
            f = pool.submit(generate_one, gemini_client, prompt, topic, out)
            futures[f] = (i, topic, out)

        for f in as_completed(futures):
            i, topic, out = futures[f]
            try:
                ok = f.result()
            except Exception as e:
                ok = False
                print(f"    [{i+1}/{BATCH_SIZE}] ERROR: {e}")
            if ok:
                generated.append((i, topic, out))
                print(f"    [{i+1}/{BATCH_SIZE}] generated: {topic[:50]}")
            else:
                print(f"    [{i+1}/{BATCH_SIZE}] FAILED: {topic[:50]}")

    if not generated:
        print("  ERROR: No diagrams generated. Skipping cycle.")
        save_state(state)
        return state

    # ── Evaluate ──────────────────────────────────────────────────
    print(f"\n  Evaluating {len(generated)} diagrams via Claude...")
    eval_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_EVAL_WORKERS) as pool:
        futures = {}
        for i, topic, path in generated:
            f = pool.submit(evaluate_one, anthropic_client, path)
            futures[f] = (i, topic, path)

        for f in as_completed(futures):
            i, topic, path = futures[f]
            try:
                result = f.result()
            except Exception as e:
                result = None
                print(f"    [{i+1}] EVAL ERROR: {e}")

            if result:
                eval_results.append(result)
                criteria_pass = sum([
                    result.get("legible_and_grammatical", False),
                    result.get("pastel_colors", False),
                    result.get("linear_layout", False),
                    result.get("no_numbers", False),
                ])
                fails = result.get("failures", [])
                print(f"    [{i+1}] {criteria_pass}/4 | {'; '.join(fails) if fails else 'all pass'}")
            else:
                eval_results.append({
                    "legible_and_grammatical": False,
                    "pastel_colors": False,
                    "linear_layout": False,
                    "no_numbers": False,
                    "failures": ["eval_error"],
                })
                print(f"    [{i+1}] 0/4 | eval failed")

    # ── Score ─────────────────────────────────────────────────────
    scores = score_batch(eval_results)
    score = scores["total"]

    print(f"\n  SCORE: {score}/40")
    print(f"    Legible:    {scores['legible']}/10")
    print(f"    Pastel:     {scores['pastel']}/10")
    print(f"    Linear:     {scores['linear']}/10")
    print(f"    No numbers: {scores['no_numbers']}/10")

    # ── Log ───────────────────────────────────────────────────────
    log_entry = {
        "run": run_num,
        "timestamp": datetime.now().isoformat(),
        "score": score,
        "max": 40,
        "criteria": {"legible": scores["legible"], "pastel": scores["pastel"], "linear": scores["linear"], "no_numbers": scores["no_numbers"]},
        "prompt_len": len(prompt),
        "generated": len(generated),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # ── Keep or discard ───────────────────────────────────────────
    if score > state["best_score"]:
        state["best_score"] = score
        BEST_PROMPT_FILE.write_text(prompt)
        print(f"\n  NEW BEST! {score}/40 (was {state.get('best_score', -1)})")
        print(f"  Saved to: {BEST_PROMPT_FILE}")
    else:
        print(f"\n  No improvement ({score} vs best {state['best_score']})")
        if BEST_PROMPT_FILE.exists():
            print("  Reverting to best prompt for next mutation")

    # ── Mutate ────────────────────────────────────────────────────
    if score < 40:
        print("\n  Mutating prompt...")
        base_prompt = BEST_PROMPT_FILE.read_text().strip() if BEST_PROMPT_FILE.exists() else prompt
        new_prompt = mutate_prompt(anthropic_client, base_prompt, eval_results, state["best_score"])
        save_prompt(new_prompt)
        print(f"  New prompt ({len(new_prompt)} chars):")
        preview = new_prompt[:200].replace("\n", " ")
        print(f"    {preview}...")
    else:
        print("\n  PERFECT 40/40! Prompt fully optimized.")

    save_state(state)
    return state


# ─── Entry Point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Diagram autoresearch loop")
    parser.add_argument("--once", action="store_true", help="Run a single cycle")
    parser.add_argument("--cycles", type=int, default=0, help="Run N cycles (0=infinite)")
    args = parser.parse_args()

    if not GEMINI_KEY:
        print("ERROR: NANO_BANANA_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    if not ANTHROPIC_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from google import genai
    import anthropic

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    gemini_client = genai.Client(api_key=GEMINI_KEY)
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    state = load_state()

    print("Diagram Autoresearch")
    print(f"  Gen model:    {GEN_MODEL}")
    print(f"  Eval model:   {EVAL_MODEL}")
    print(f"  Mutate model: {MUTATE_MODEL}")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Cycle:        {CYCLE_SECONDS}s")
    print(f"  State:        run {state['run_number']}, best {state['best_score']}/40")

    if args.once:
        run_cycle(gemini_client, anthropic_client, state)
        return

    max_cycles = args.cycles or float("inf")
    i = 0
    while i < max_cycles:
        start = time.time()
        try:
            state = run_cycle(gemini_client, anthropic_client, state)
        except Exception as e:
            print(f"\n  CYCLE ERROR: {e}")
            traceback.print_exc()
        elapsed = time.time() - start
        i += 1

        if i < max_cycles:
            wait = max(0, CYCLE_SECONDS - elapsed)
            if wait > 0:
                print(f"\n  Waiting {wait:.0f}s until next cycle...")
                time.sleep(wait)
            else:
                print(f"\n  Cycle took {elapsed:.0f}s (>{CYCLE_SECONDS}s budget)")

    print(f"\nDone. Best score: {state['best_score']}/40")
    if BEST_PROMPT_FILE.exists():
        print(f"Best prompt: {BEST_PROMPT_FILE}")


if __name__ == "__main__":
    main()
