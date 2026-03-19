#!/usr/bin/env python3
"""
Diagram Autoresearch -- Pareto frontier prompt optimization.

Maintains a Pareto frontier of non-dominated prompts across 4 criteria.
Uses LLM-generated adversarial topics to stress-test weak criteria.
Selects parents from the frontier weighted toward the weakest dimension.

The fixed evaluation harness lives in prepare.py (do not modify).

Usage:
    uv run python autoresearch_skills/train.py              # Continuous loop
    uv run python autoresearch_skills/train.py --once       # Single cycle
    uv run python autoresearch_skills/train.py --cycles 5   # Run N cycles
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from autoresearch_skills.prepare import (
    GEMINI_KEY, ANTHROPIC_KEY,
    GEN_MODEL, EVAL_MODEL, MUTATE_MODEL,
    BASE_DIR, BEST_PROMPT_FILE, RESULTS_FILE, DIAGRAMS_DIR,
    BATCH_SIZE, CYCLE_SECONDS, MAX_GEN_WORKERS, MAX_EVAL_WORKERS,
    TOPICS,
    evaluate_one, score_batch,
    load_state, save_state, load_prompt, save_prompt,
)

CRITERIA = ["legible", "pastel", "linear", "no_numbers"]
FRONTIER_FILE = BASE_DIR / "frontier.jsonl"
ADVERSARIAL_TOPIC_COUNT = 3

# ─── Pareto Frontier ─────────────────────────────────────────────────────────


def dominates(a: dict, b: dict) -> bool:
    """True if a is strictly better on at least one criterion and no worse on all."""
    dominated_one = False
    for c in CRITERIA:
        if a.get(c, 0) < b.get(c, 0):
            return False
        if a.get(c, 0) > b.get(c, 0):
            dominated_one = True
    return dominated_one


def update_frontier(frontier: list[dict], candidate: dict) -> tuple[list[dict], bool]:
    """Add candidate if non-dominated. Prune any members it dominates."""
    for member in frontier:
        if dominates(member, candidate):
            return frontier, False
    pruned = [m for m in frontier if not dominates(candidate, m)]
    pruned.append(candidate)
    return pruned, True


def load_frontier() -> list[dict]:
    if not FRONTIER_FILE.exists():
        return []
    entries = []
    for line in FRONTIER_FILE.read_text().strip().split("\n"):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def save_frontier(frontier: list[dict]):
    FRONTIER_FILE.write_text("\n".join(json.dumps(e) for e in frontier) + "\n")


def find_weakest_criterion(frontier: list[dict]) -> str:
    """Identify which criterion has the lowest max score across the frontier."""
    if not frontier:
        return "legible"
    best_per = {c: max(m.get(c, 0) for m in frontier) for c in CRITERIA}
    return min(best_per, key=best_per.get)


def select_parent(frontier: list[dict], weakest: str) -> dict:
    """Select a frontier member, weighted toward those strong on the weakest criterion."""
    if not frontier:
        return None
    weights = [m.get(weakest, 0) + 1 for m in frontier]
    total = sum(weights)
    r = random.random() * total
    cumulative = 0
    for m, w in zip(frontier, weights):
        cumulative += w
        if r <= cumulative:
            return m
    return frontier[-1]


# ─── Adversarial Topic Generation ────────────────────────────────────────────

TOPIC_GEN_TEMPLATE = """Generate {count} novel technical diagram topics that would stress-test the "{criterion}" criterion in AI-generated diagrams.

The criterion "{criterion}" means:
{criterion_description}

Each topic should be a single sentence describing a technical workflow or system architecture, similar to these examples:
- "CI/CD pipeline: code commit flows to build, then test, then deploy, then monitor"
- "OAuth flow: user to app to authorization server to resource server"

For the "{criterion}" criterion, create topics that are particularly challenging. {stress_instructions}

Return ONLY {count} topics, one per line. No numbering, no bullets, no explanation."""

CRITERION_DESCRIPTIONS = {
    "legible": "All text in the diagram must be clearly readable, correctly spelled English words",
    "pastel": "The diagram uses only soft pastel colors for fills, no bright or dark colors",
    "linear": "The diagram flows in one clear linear direction, not circular or branching",
    "no_numbers": "The diagram contains zero numbers, step numbers, or ordinals",
}

STRESS_INSTRUCTIONS = {
    "legible": "Include topics with long technical terms (e.g., 'Authentication', 'Orchestration', 'Preprocessing') and domain-specific jargon that image models commonly misspell. The more complex the vocabulary, the better the stress test.",
    "pastel": "Include topics where the subject matter naturally suggests strong, vivid colors (e.g., alerting systems, error states, traffic lights) to test whether the prompt can override default color choices.",
    "linear": "Include topics with inherently branching or circular structures (e.g., feedback loops, decision trees, hub-and-spoke architectures) to test whether the prompt can force linear layout.",
    "no_numbers": "Include topics that naturally involve counting, versioning, or sequencing (e.g., version control, phased rollouts, tiered pricing) to test whether the prompt can suppress numbers.",
}


def generate_adversarial_topics(anthropic_client, weakest: str, count: int = ADVERSARIAL_TOPIC_COUNT) -> list[str]:
    """Use Claude to create topics that stress-test the weakest criterion."""
    prompt = TOPIC_GEN_TEMPLATE.format(
        count=count,
        criterion=weakest,
        criterion_description=CRITERION_DESCRIPTIONS.get(weakest, ""),
        stress_instructions=STRESS_INSTRUCTIONS.get(weakest, ""),
    )
    try:
        response = anthropic_client.messages.create(
            model=MUTATE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        topics = [line.strip() for line in text.split("\n") if line.strip()]
        return topics[:count]
    except Exception as e:
        print(f"    TOPIC GEN ERROR: {e}")
        return []


# ─── Mutation Templates ──────────────────────────────────────────────────────

REFINE_TEMPLATE = """You are optimizing a text-to-image prompt for generating technical diagrams. The prompt is sent to Gemini's image generation model. Your goal: modify it so generated diagrams consistently pass ALL 4 evaluation criteria.

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

{frontier_context}

{focus_instructions}

RULES FOR YOUR MODIFICATION:
- Keep the core whiteboard/hand-drawn aesthetic
- Be specific and imperative -- image models respond to direct commands
- Keep prompt under 400 words
- Return ONLY the new prompt text -- no explanation, no markdown fences"""

EXPLORE_TEMPLATE = """You are radically redesigning a text-to-image prompt. The current approach has plateaued -- incremental tweaks no longer help. Try a FUNDAMENTALLY DIFFERENT structure.

CURRENT PROMPT (PLATEAU -- do NOT tweak, RESTRUCTURE):
---
{current_prompt}
---

SCORES (stuck): legible={leg_rate}/10, pastel={col_rate}/10, linear={lin_rate}/10, no_numbers={num_rate}/10

FAILURES:
{failures}

{frontier_context}

Try ONE of these radical approaches:
1. MINIMALIST: Under 150 words. Strip all rule lists. Short direct commands only.
2. REDUCE TEXT: Use 1-2 word labels max, common short words, icons over text, 4-5 boxes total.
3. STRUCTURAL REWRITE: Describe an existing image ("This is a diagram that shows...") not instructions.
4. EXAMPLE-DRIVEN: Describe the exact ideal diagram: "First box is light purple, says 'Input'. Arrow right to light blue box 'Process'..."
5. NEGATIVE-SPACE: Focus on bans: "NEVER more than 2 words per label. NEVER words longer than 10 chars."

Be bold. The current approach has provably stalled.

Keep under 400 words. Return ONLY the new prompt text -- no explanation, no markdown fences."""

BOTTLENECK_FOCUS = {
    "legible": """CRITICAL FOCUS -- The main bottleneck is text legibility. Other criteria are near-perfect.
Do NOT add more spelling rules -- the prompt already has them and they don't help.
Try a different approach: reduce text complexity (shorter labels, fewer boxes, common short words),
use larger bolder text, or rely more on icons than words.""",
    "pastel": """CRITICAL FOCUS -- Colors are the bottleneck. List EXACT pastel colors to use (light purple, light blue, light green, light pink, light yellow). Explicitly ban all saturated, dark, neon, or bright fills.""",
    "linear": """CRITICAL FOCUS -- Layout is the bottleneck. The diagram must be a single horizontal chain: Box -> Box -> Box. Explicitly ban branching, fan-out, vertical elements, return arrows, circular flows.""",
    "no_numbers": """CRITICAL FOCUS -- Numbers keep appearing. Explicitly ban ALL digits, ordinals, step numbers, version numbers, and sequence indicators. Icons must use shapes only, never digits.""",
}

# ─── Plateau Detection ───────────────────────────────────────────────────────

PLATEAU_WINDOW = 3


def detect_plateau(best_score: int, window: int = PLATEAU_WINDOW) -> bool:
    """True if best score hasn't improved in the last `window` runs."""
    if not RESULTS_FILE.exists():
        return False
    lines = RESULTS_FILE.read_text().strip().split("\n")
    if len(lines) < window:
        return False
    recent = []
    for line in lines[-window:]:
        try:
            recent.append(json.loads(line)["score"])
        except (json.JSONDecodeError, KeyError):
            return False
    return all(s <= best_score for s in recent)


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


def _collect_failures(eval_results: list[dict]) -> str:
    all_failures = []
    for r in eval_results:
        for f in r.get("failures", []):
            all_failures.append(f)
    unique = list(dict.fromkeys(all_failures))[:20]
    return "\n".join(f"- {f}" for f in unique) if unique else "- None"


def _frontier_context(frontier: list[dict]) -> str:
    if len(frontier) <= 1:
        return ""
    lines = ["OTHER FRONTIER PROMPTS (different trade-offs that also performed well):"]
    for i, m in enumerate(frontier[:5]):
        lines.append(f"  Prompt {i+1}: legible={m.get('legible',0)}, pastel={m.get('pastel',0)}, linear={m.get('linear',0)}, no_numbers={m.get('no_numbers',0)} (total={m.get('total',0)})")
        preview = m.get("prompt", "")[:100].replace("\n", " ")
        lines.append(f"    Preview: {preview}...")
    lines.append("Consider borrowing techniques from prompts that score high on your weak criterion.")
    return "\n".join(lines)


def mutate_prompt(
    anthropic_client,
    current_prompt: str,
    eval_results: list[dict],
    best_score: int,
    explore: bool = False,
    bottleneck: str | None = None,
    frontier: list[dict] | None = None,
) -> str:
    """Use Claude to improve the prompt based on failure analysis."""
    scores = score_batch(eval_results)
    failures_text = _collect_failures(eval_results)

    focus_instructions = ""
    if bottleneck and bottleneck in BOTTLENECK_FOCUS:
        focus_instructions = BOTTLENECK_FOCUS[bottleneck]

    frontier_ctx = _frontier_context(frontier or [])

    template = EXPLORE_TEMPLATE if explore else REFINE_TEMPLATE
    mutation_prompt = template.format(
        current_prompt=current_prompt,
        score=scores["total"],
        leg_rate=scores["legible"],
        col_rate=scores["pastel"],
        lin_rate=scores["linear"],
        num_rate=scores["no_numbers"],
        best_score=best_score,
        failures=failures_text,
        focus_instructions=focus_instructions,
        frontier_context=frontier_ctx,
    )

    response = anthropic_client.messages.create(
        model=MUTATE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": mutation_prompt}],
    )
    return response.content[0].text.strip()


# ─── Main Cycle ──────────────────────────────────────────────────────────────


def run_cycle(gemini_client, anthropic_client, state: dict) -> dict:
    """Run one Pareto frontier optimization cycle."""
    run_num = state["run_number"] + 1
    state["run_number"] = run_num
    run_dir = DIAGRAMS_DIR / f"run_{run_num:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    frontier = load_frontier()
    weakest = find_weakest_criterion(frontier)
    is_plateau = detect_plateau(state["best_score"])
    mode = "EXPLORE" if is_plateau else "REFINE"

    # Select parent prompt from frontier or fall back to current
    parent = select_parent(frontier, weakest) if frontier else None
    if parent and random.random() < 0.5:
        prompt = parent["prompt"]
        print(f"  Using frontier parent (strong on {weakest})")
    else:
        prompt = load_prompt()

    # Mix standard topics + adversarial topics targeting the weakest criterion
    n_standard = BATCH_SIZE - ADVERSARIAL_TOPIC_COUNT
    standard_topics = random.sample(TOPICS, min(n_standard, len(TOPICS)))
    adversarial_topics = generate_adversarial_topics(anthropic_client, weakest)
    topics = standard_topics + adversarial_topics
    random.shuffle(topics)
    topics = topics[:BATCH_SIZE]

    print(f"\n{'='*60}")
    print(f"RUN {run_num} | {datetime.now().strftime('%H:%M:%S')} | Best: {state['best_score']}/40 | Mode: {mode} | Weakest: {weakest}")
    print(f"  Frontier size: {len(frontier)} | Adversarial topics: {len(adversarial_topics)}")
    print(f"{'='*60}")

    # ── Generate ──────────────────────────────────────────────────
    print(f"\n  Generating {len(topics)} diagrams...")
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
        "criteria": scores,
        "prompt_len": len(prompt),
        "generated": len(generated),
        "mode": mode,
        "weakest": weakest,
        "frontier_size": len(frontier),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # ── Update Pareto frontier ────────────────────────────────────
    candidate = {
        "prompt": prompt,
        "legible": scores["legible"],
        "pastel": scores["pastel"],
        "linear": scores["linear"],
        "no_numbers": scores["no_numbers"],
        "total": score,
        "run": run_num,
    }
    frontier, added = update_frontier(frontier, candidate)
    save_frontier(frontier)

    if added:
        print(f"\n  FRONTIER: Added (frontier size: {len(frontier)})")
    else:
        print(f"\n  FRONTIER: Dominated (frontier size: {len(frontier)})")

    # ── Also track simple best ────────────────────────────────────
    if score > state["best_score"]:
        state["best_score"] = score
        BEST_PROMPT_FILE.write_text(prompt)
        print(f"  NEW BEST! {score}/40")

    # ── Mutate ────────────────────────────────────────────────────
    if score < 40:
        bottleneck_name = weakest if scores.get(weakest, 10) < 8 else None
        if bottleneck_name:
            print(f"  Bottleneck: {bottleneck_name}")

        print(f"  Mutating prompt ({mode} mode)...")
        base_prompt = BEST_PROMPT_FILE.read_text().strip() if BEST_PROMPT_FILE.exists() else prompt
        new_prompt = mutate_prompt(
            anthropic_client, base_prompt, eval_results,
            state["best_score"], explore=is_plateau,
            bottleneck=bottleneck_name, frontier=frontier,
        )
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
        print("ERROR: GOOGLE_API_KEY not set", file=sys.stderr)
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
   
    print("Diagram Autoresearch (Pareto Frontier)")
    print(f"  Gen model:    {GEN_MODEL}")
    print(f"  Eval model:   {EVAL_MODEL}")
    print(f"  Mutate model: {MUTATE_MODEL}")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Cycle:        {CYCLE_SECONDS}s")
    print(f"  State:        run {state['run_number']}, best {state['best_score']}/40")
    print(f"  Frontier:     {len(load_frontier())} members")

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
