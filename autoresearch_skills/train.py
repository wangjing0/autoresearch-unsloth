#!/usr/bin/env python3
"""
Diagram Autoresearch -- Pareto frontier prompt optimization.

Maintains a Pareto frontier of non-dominated prompts across 6 graded criteria.
Uses LLM-generated adversarial topics to stress-test weak criteria.
Selects parents from the frontier weighted toward the weakest dimension.

The fixed evaluation harness lives in prepare.py (do not modify).

Usage:
    uv run python autoresearch_skills/train.py              # Continuous loop
    uv run python autoresearch_skills/train.py --once       # Single cycle
    uv run python autoresearch_skills/train.py --cycles 5   # Run N cycles
    uv run python autoresearch_skills/train.py --reset      # Reset state
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
    TOPICS, CRITERIA, MAX_SCORE, PLATEAU_WINDOW,
    evaluate_one, score_batch,
    load_state, save_state, load_prompt, save_prompt,
)

FRONTIER_FILE = BASE_DIR / "frontier.jsonl"
ADVERSARIAL_TOPIC_COUNT = 3

CRITERIA_LABELS = {
    "text_quality": "Text Quality",
    "color_palette": "Color Palette",
    "layout": "Layout",
    "label_discipline": "Label Discipline",
    "visual_clarity": "Visual Clarity",
    "icon_quality": "Icon Quality",
}

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
        return "text_quality"
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
    "text_quality": "All text must be crisp, correctly spelled, consistently sized, and well-spaced",
    "color_palette": "Must use harmonious soft pastel colors with deliberate color coding",
    "layout": "Must flow in a perfectly aligned linear chain with uniform spacing",
    "label_discipline": "Labels must be concise (2-4 words), no numbers or ordinals, consistent style",
    "visual_clarity": "Must look publication-quality with balanced composition and consistent styling",
    "icon_quality": "Must have clear, relevant, consistently styled line-art icons in every box",
}

STRESS_INSTRUCTIONS = {
    "text_quality": "Include topics with long technical terms (e.g., 'Authentication', 'Orchestration', 'Preprocessing') and domain jargon that image models commonly misspell.",
    "color_palette": "Include topics where the subject matter naturally suggests strong, vivid colors (e.g., alerting systems, error states, traffic lights) to test whether the prompt can override defaults.",
    "layout": "Include topics with inherently branching or circular structures (e.g., feedback loops, decision trees, hub-and-spoke) to test whether the prompt can force linear layout.",
    "label_discipline": "Include topics that naturally involve counting, versioning, or sequencing (e.g., version control, phased rollouts, tiered pricing) to test whether the prompt can suppress numbers.",
    "visual_clarity": "Include topics with many components (8+ steps) that risk visual clutter, overlapping elements, or inconsistent styling.",
    "icon_quality": "Include topics with abstract or hard-to-iconify concepts (e.g., 'governance', 'compliance', 'abstraction layer') to test whether the model can produce relevant icons.",
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

REFINE_TEMPLATE = """You are optimizing a text-to-image prompt for generating technical diagrams. The prompt is sent to Gemini's image generation model. Diagrams are scored on 6 criteria, each normalized to 0-10. Overall score is the mean of all 6, max 10.00.

CURRENT PROMPT:
---
{current_prompt}
---

LAST BATCH: overall {overall}/10
- Text Quality:      {s_text_quality}/10
- Color Palette:     {s_color_palette}/10
- Layout:            {s_layout}/10
- Label Discipline:  {s_label_discipline}/10
- Visual Clarity:    {s_visual_clarity}/10
- Icon Quality:      {s_icon_quality}/10

COMMON FAILURES:
{failures}

{frontier_context}

{focus_instructions}

RULES FOR YOUR MODIFICATION:
- Target the lowest-scoring criteria first
- Be specific and imperative -- image models respond to direct commands
- Keep prompt under 400 words
- Return ONLY the new prompt text -- no explanation, no markdown fences"""

EXPLORE_TEMPLATE = """You are radically redesigning a text-to-image prompt. The current approach has plateaued -- incremental tweaks no longer help. Try a FUNDAMENTALLY DIFFERENT structure.

Diagrams scored on 6 criteria (0-10 each). Current scores are stuck:
- Text Quality: {s_text_quality}/10 | Color Palette: {s_color_palette}/10 | Layout: {s_layout}/10
- Label Discipline: {s_label_discipline}/10 | Visual Clarity: {s_visual_clarity}/10 | Icon Quality: {s_icon_quality}/10
- Overall: {overall}/10

CURRENT PROMPT (PLATEAU -- do NOT tweak, RESTRUCTURE):
---
{current_prompt}
---

FAILURES:
{failures}

{frontier_context}

Try ONE of these radical approaches:
1. MINIMALIST: Under 150 words. Strip all rule lists. Short direct commands only.
2. REDUCE TEXT: Use 1-2 word labels max, common short words, icons over text, 4-5 boxes total.
3. STRUCTURAL REWRITE: Describe an existing image ("This is a diagram that shows...") not instructions.
4. EXAMPLE-DRIVEN: Describe the exact ideal diagram concretely: box colors, label text, arrow style.
5. NEGATIVE-SPACE: Focus entirely on bans: "NEVER more than 2 words per label. NEVER words longer than 10 chars."

Be bold. The current approach has provably stalled.

Keep under 400 words. Return ONLY the new prompt text -- no explanation, no markdown fences."""

BOTTLENECK_FOCUS = {
    "text_quality": """CRITICAL FOCUS -- Text quality is the main bottleneck.
Do NOT add more spelling rules -- try reducing text complexity instead.
Shorter labels (1-2 words), common short words, larger bolder text, fewer boxes.""",
    "color_palette": """CRITICAL FOCUS -- Color palette is the bottleneck.
List EXACT pastel colors to use. Ban saturated/dark/neon fills explicitly.
Ask for deliberate color coding where related concepts share hues.""",
    "layout": """CRITICAL FOCUS -- Layout is the bottleneck.
Must be a single horizontal chain: Box -> Box -> Box with uniform spacing.
Ban branching, fan-out, vertical elements, return arrows, circular flows.""",
    "label_discipline": """CRITICAL FOCUS -- Label discipline is the bottleneck.
Ban ALL digits, ordinals, step numbers, version numbers. Require concise 2-4 word labels.
Consistent capitalization style across all boxes.""",
    "visual_clarity": """CRITICAL FOCUS -- Visual clarity is the bottleneck.
Require consistent border widths, uniform box sizes, generous whitespace.
Ban drop shadows, gradients, thick borders, decorative elements.""",
    "icon_quality": """CRITICAL FOCUS -- Icon quality is the bottleneck.
Require simple line-art icons in EVERY box. Icons must be relevant to the concept.
Consistent icon style (all outline, all same stroke width). Ban empty boxes.""",
}

# ─── Plateau Detection ───────────────────────────────────────────────────────


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
        s10 = m.get("scores", {})
        parts = ", ".join(f"{c}={s10.get(c, 0)}/10" for c in CRITERIA)
        lines.append(f"  Prompt {i+1}: {parts} (overall={m.get('overall', 0)}/10)")
        preview = m.get("prompt", "")[:100].replace("\n", " ")
        lines.append(f"    Preview: {preview}...")
    lines.append("Consider borrowing techniques from prompts that score high on your weak criterion.")
    return "\n".join(lines)


def mutate_prompt(
    anthropic_client,
    current_prompt: str,
    eval_results: list[dict],
    best_score: float,
    explore: bool = False,
    bottleneck: str | None = None,
    frontier: list[dict] | None = None,
) -> str:
    """Use Claude to improve the prompt based on failure analysis."""
    scores = score_batch(eval_results)
    failures_text = _collect_failures(eval_results)
    s10 = scores["scores"]

    focus_instructions = ""
    if bottleneck and bottleneck in BOTTLENECK_FOCUS:
        focus_instructions = BOTTLENECK_FOCUS[bottleneck]

    frontier_ctx = _frontier_context(frontier or [])

    template = EXPLORE_TEMPLATE if explore else REFINE_TEMPLATE
    mutation_prompt = template.format(
        current_prompt=current_prompt,
        overall=scores["overall"],
        s_text_quality=s10.get("text_quality", 0),
        s_color_palette=s10.get("color_palette", 0),
        s_layout=s10.get("layout", 0),
        s_label_discipline=s10.get("label_discipline", 0),
        s_visual_clarity=s10.get("visual_clarity", 0),
        s_icon_quality=s10.get("icon_quality", 0),
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

    parent = select_parent(frontier, weakest) if frontier else None
    if parent and random.random() < 0.5:
        prompt = parent["prompt"]
        print(f"  Using frontier parent (strong on {weakest})")
    else:
        prompt = load_prompt()

    n_standard = BATCH_SIZE - ADVERSARIAL_TOPIC_COUNT
    standard_topics = random.sample(TOPICS, min(n_standard, len(TOPICS)))
    adversarial_topics = generate_adversarial_topics(anthropic_client, weakest)
    topics = standard_topics + adversarial_topics
    random.shuffle(topics)
    topics = topics[:BATCH_SIZE]

    print(f"\n{'='*70}")
    print(f"RUN {run_num} | {datetime.now().strftime('%H:%M:%S')} | Best: {state['best_score']}/{MAX_SCORE} | Mode: {mode} | Weakest: {weakest}")
    print(f"  Frontier size: {len(frontier)} | Adversarial topics: {len(adversarial_topics)}")
    print(f"{'='*70}")

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
                diag_scores = [result.get(c, 1) for c in CRITERIA]
                diag_avg = sum(diag_scores) / len(diag_scores)
                fails = result.get("failures", [])
                print(f"    [{i+1}] avg {diag_avg:.1f}/5 | {'; '.join(fails[:2]) if fails else 'no issues'}")
            else:
                eval_results.append({c: 1 for c in CRITERIA} | {"failures": ["eval_error"]})
                print(f"    [{i+1}] avg 1.0/5 | eval failed")

    # ── Score ─────────────────────────────────────────────────────
    scores = score_batch(eval_results)
    overall = scores["overall"]
    s10 = scores["scores"]

    print(f"\n  SCORE: {overall}/{MAX_SCORE}")
    for c in CRITERIA:
        label = CRITERIA_LABELS.get(c, c)
        print(f"    {label:20s} {s10[c]:.2f}/10")

    # ── Log ───────────────────────────────────────────────────────
    log_entry = {
        "run": run_num,
        "timestamp": datetime.now().isoformat(),
        "score": overall,
        "max": MAX_SCORE,
        "scores": s10,
        "raw_avgs": scores["raw_avgs"],
        "prompt_len": len(prompt),
        "generated": len(generated),
        "mode": mode,
        "weakest": weakest,
        "frontier_size": len(frontier),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # ── Update Pareto frontier ────────────────────────────────────
    candidate = {c: s10[c] for c in CRITERIA}
    candidate.update({"prompt": prompt, "scores": s10, "overall": overall, "run": run_num})
    frontier, added = update_frontier(frontier, candidate)
    save_frontier(frontier)

    if added:
        print(f"\n  FRONTIER: Added (frontier size: {len(frontier)})")
    else:
        print(f"\n  FRONTIER: Dominated (frontier size: {len(frontier)})")

    # ── Also track simple best ────────────────────────────────────
    if overall > state["best_score"]:
        state["best_score"] = overall
        BEST_PROMPT_FILE.write_text(prompt)
        print(f"  NEW BEST! {overall}/{MAX_SCORE}")

    # ── Mutate ────────────────────────────────────────────────────
    if overall < MAX_SCORE:
        bottleneck_name = weakest if s10.get(weakest, 10) < 5.0 else None
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
        print("\n  PERFECT SCORE! Prompt fully optimized.")

    save_state(state)
    return state


# ─── Entry Point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Diagram autoresearch loop")
    parser.add_argument("--once", action="store_true", help="Run a single cycle")
    parser.add_argument("--cycles", type=int, default=0, help="Run N cycles (0=infinite)")
    parser.add_argument("--reset", action="store_true", help="Reset state, results, frontier, and best_prompt")
    args = parser.parse_args()

    if args.reset:
        import shutil
        RESULTS_FILE.write_text("")
        save_state({"best_score": -1, "run_number": 0})
        FRONTIER_FILE.unlink(missing_ok=True)
        BEST_PROMPT_FILE.unlink(missing_ok=True)
        diagrams = DIAGRAMS_DIR
        if diagrams.exists():
            shutil.rmtree(diagrams)
            diagrams.mkdir(parents=True)
        print("Reset: cleared results, state, frontier, best_prompt, and diagrams. New run will start from scratch.")

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

    print("Diagram Autoresearch Pareto Frontier Optimization")
    print(f"  Gen model:    {GEN_MODEL}")
    print(f"  Eval model:   {EVAL_MODEL}")
    print(f"  Mutate model: {MUTATE_MODEL}")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Max score:    {MAX_SCORE} (mean of 6 criteria, each 0-10)")
    print(f"  State:        run {state['run_number']}, best {state['best_score']}/{MAX_SCORE}")
    print(f"  Frontier:     {len(load_frontier())} members")

    if args.once:
        run_cycle(gemini_client, anthropic_client, state)
        return

    max_cycles = args.cycles or float("inf")
    i = 0
    while i < max_cycles:
        start = time.time()
        print(f"Running cycle {i+1} of {max_cycles}")
        try:
            state = run_cycle(gemini_client, anthropic_client, state)
        except Exception as e:
            print(f"\n  CYCLE ERROR: {e}")
            traceback.print_exc()
        elapsed = time.time() - start
        i += 1

        print(f"\n  Cycle took {elapsed:.0f}s")

        #if detect_plateau(state["best_score"], window=PLATEAU_WINDOW):
        #    print(f"\n  Early stopping: no improvement in {PLATEAU_WINDOW} consecutive runs.")
        #    break

    print(f"\nDone. Best score: {state['best_score']}/{MAX_SCORE}")
    if BEST_PROMPT_FILE.exists():
        print(f"Best prompt: {BEST_PROMPT_FILE}")


if __name__ == "__main__":
    main()
