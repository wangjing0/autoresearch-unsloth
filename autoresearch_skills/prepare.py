#!/usr/bin/env python3
"""
Fixed evaluation harness, constants, and utilities for diagram autoresearch.
Do not modify -- this is the ground truth eval, analogous to prepare.py in
the original autoresearch pattern.

Provides: constants, file paths, topics, eval prompt, evaluate_one(),
state helpers, and score_batch().
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

GEN_MODEL = "gemini-2.5-flash-image"
EVAL_MODEL = "claude-sonnet-4-6"
MUTATE_MODEL = "claude-sonnet-4-6"

BASE_DIR = Path(__file__).resolve().parent / "state"
PROMPT_FILE = BASE_DIR / "prompt.txt"
BEST_PROMPT_FILE = BASE_DIR / "best_prompt.txt"
STATE_FILE = BASE_DIR / "state.json"
RESULTS_FILE = BASE_DIR / "results.jsonl"
FRONTIER_FILE = BASE_DIR / "frontier.jsonl"
DIAGRAMS_DIR = BASE_DIR / "diagrams"

BATCH_SIZE = 10
CYCLE_SECONDS = 120
MAX_GEN_WORKERS = 3
MAX_EVAL_WORKERS = 5
PLATEAU_WINDOW = 3
DEFAULT_CYCLES = 10

INITIAL_PROMPT = (
    "Create a clean, hand-drawn style diagram on a white background. "
    "Use soft pastel colored rounded rectangles (light purple, light blue, "
    "light green, light pink, light yellow, light teal) for each concept. "
    "Include simple line-art icons inside each box. Use thin black arrows "
    "to show relationships. Text labels should be clean, dark, and easy to "
    "read. Style should look like a whiteboard sketch or notebook illustration "
    "-- professional but approachable. No photorealistic elements. No gradients. "
    "Keep it minimal and clear."
)

# ─── Diagram Topics (diverse structures) ─────────────────────────────────────

TOPICS = [
    "CI/CD pipeline: code commit flows to build, then test, then deploy, then monitor",
    "Machine learning workflow: data collection to preprocessing to training to evaluation to deployment",
    "User authentication: login form to validation to JWT token to access granted",
    "Microservices: API gateway connecting to user service, payment service, and notification service",
    "ETL pipeline: extract from databases, transform with business rules, load into warehouse",
    "Email marketing funnel: capture lead to nurture sequence to segment to convert to retain",
    "Content creation pipeline: research to outline to draft to edit to publish",
    "Customer support flow: ticket submitted to triage to assign to resolve to follow up",
    "API request lifecycle: client request to load balancer to app server to database to response",
    "Git workflow: feature branch to pull request to code review to merge to deploy",
    "OAuth flow: user to app to authorization server to resource server",
    "Notification system: event trigger to queue to router splitting to email, push, and SMS channels",
    "Search engine pipeline: crawl pages to index to rank to serve results",
    "Video processing: upload to transcode to generate thumbnail to CDN to stream",
    "A/B testing: hypothesis to experiment design to traffic split to measure to analyze",
    "Payment processing: cart to checkout to payment gateway to bank to confirmation",
    "Recommendation engine: user activity to features to model to ranked results to display",
    "Monitoring stack: metrics collection to aggregation to alerting to dashboard",
    "Caching strategy: request to cache check then hit path or miss path to origin server",
    "Log aggregation: app logs to collector to parser to storage to visualization",
    "Message queue: producer to exchange to routing to queues to consumers",
    "Blue-green deploy: load balancer switching between blue and green environments",
    "SSO architecture: identity provider connecting to multiple service providers via tokens",
    "Data lake: raw ingestion to cataloging to processing to curated zone to analytics",
    "Serverless flow: API request to function trigger to compute to storage to response",
    "Container orchestration: registry to scheduler to node to pod to service mesh",
    "GraphQL architecture: client query to resolver to data sources to merged response",
    "Event sourcing: command to event store to projections to read models to query",
    "Feature flag system: config to evaluation engine to user targeting to rollout",
    "Rate limiting: request to counter check to allow or reject to update counter",
]

# ─── Eval Prompt (DO NOT CHANGE -- this is the fixed metric) ─────────────────

CRITERIA = ["text_quality", "color_palette", "layout", "label_discipline", "visual_clarity", "icon_quality"]
MAX_SCORE = 10.0  # overall score is 0-10, aggregated from 6 criteria

EVAL_PROMPT = """You are a strict evaluator of AI-generated technical diagrams. Score the image on 6 criteria using a 1-5 scale for each. Be demanding -- a 5 should be genuinely excellent, not merely acceptable.

Criteria (1 = terrible, 2 = poor, 3 = acceptable, 4 = good, 5 = excellent):

1. TEXT_QUALITY (1-5): How legible and correct is the text?
   1: Most text garbled, overlapping, or unreadable
   2: Some words readable but multiple misspellings or garbled sections
   3: Most text readable, minor spelling errors (1-2 words)
   4: All text readable and correctly spelled, but spacing or sizing slightly off
   5: Perfectly crisp, correctly spelled text with clean spacing and consistent sizing

2. COLOR_PALETTE (1-5): How well does it use soft pastel colors?
   2: Mostly pastel but 1-2 elements use slightly saturated colors
   3: All pastel fills but inconsistent palette (clashing pastels, too many hues)
   4: Clean consistent pastel palette with good contrast against white background
   5: Harmonious pastel palette with deliberate color coding (related concepts share hues)
   1: Bright, saturated, neon, or dark fills dominate

3. LAYOUT (1-5): How linear and clean is the flow?
   1: Chaotic, scattered, or circular arrangement
   2: General direction visible but with branching, backtracking, or misaligned elements
   3: Linear flow (left-to-right or top-to-bottom) but with minor alignment issues or crowding
   4: Clean linear flow with consistent spacing, but arrows or connectors slightly messy
   5: Perfectly aligned linear chain with uniform spacing, clean arrows, and visual rhythm

4. LABEL_DISCIPLINE (1-5): Are labels concise and free of numbers/ordinals?
   1: Contains step numbers, ordinals, version numbers, or excessive text per label
   2: No numbers but labels are verbose (full sentences) or inconsistent in style
   3: Short labels, no numbers, but some labels are vague or too abbreviated
   4: Concise, descriptive labels (2-4 words each), no numbers, consistent style
   5: Perfect labeling -- each box has a clear, concise, descriptive label with consistent capitalization and no numbers

5. VISUAL_CLARITY (1-5): How clean and professional does the overall diagram look?
   1: Cluttered, overlapping elements, hard to follow
   2: Followable but visually noisy (thick borders, drop shadows, unnecessary decoration)
   3: Clean overall but inconsistent styling (mixed border widths, uneven shapes)
   4: Professional appearance with consistent styling, good whitespace usage
   5: Publication-quality -- could appear in a textbook. Balanced composition, intentional whitespace, consistent visual hierarchy

6. ICON_QUALITY (1-5): How well do the icons/symbols inside boxes work?
   1: No icons, or icons are garbled/unrecognizable
   2: Some icons present but inconsistent (some boxes have them, others don't) or poorly drawn
   3: Icons in most boxes, recognizable but generic
   4: Clear, relevant icons in all boxes that help convey meaning
   5: Excellent icons -- simple, relevant, consistently styled line-art that enhances understanding

Respond in this exact JSON format:
{"text_quality": 3, "color_palette": 4, "layout": 3, "label_discipline": 4, "visual_clarity": 3, "icon_quality": 2, "failures": ["specific issue 1", "specific issue 2"]}

Always include at least one entry in failures describing the most significant shortcoming, even for high scores. Be specific about what you see."""

# ─── Helpers ──────────────────────────────────────────────────────────────────


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"best_score": -1, "run_number": 0}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_prompt() -> str:
    return PROMPT_FILE.read_text().strip()


def save_prompt(prompt: str):
    PROMPT_FILE.write_text(prompt)


# ─── Evaluation (Claude) ─────────────────────────────────────────────────────


def evaluate_one(anthropic_client, image_path: Path) -> dict | None:
    """Evaluate a single diagram on 6 criteria (1-5 scale each) via Claude vision."""
    image_bytes = image_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode()

    try:
        response = anthropic_client.messages.create(
            model=EVAL_MODEL,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": EVAL_PROMPT},
                    ],
                }
            ],
        )
        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
        for c in CRITERIA:
            if c in result:
                result[c] = max(1, min(5, int(result[c])))
        return result
    except Exception as e:
        print(f"    EVAL ERROR: {e}")
        return None


def score_batch(eval_results: list[dict]) -> dict:
    """Compute per-criterion scores (0-10) and overall score (0-10, 2 decimal).

    Each criterion raw average (1-5) is mapped to 0-10 via (avg-1)/4*10.
    Overall score = mean of all 6 dimension scores, rounded to 2 decimals.
    """
    if not eval_results:
        scores_10 = {c: 0.0 for c in CRITERIA}
        return {"scores": scores_10, "overall": 0.0, "max": MAX_SCORE, "raw_avgs": {c: 1.0 for c in CRITERIA}}
    n = len(eval_results)
    raw_avgs = {c: sum(r.get(c, 1) for r in eval_results) / n for c in CRITERIA}
    scores_10 = {c: round((raw_avgs[c] - 1) / 4 * 10, 2) for c in CRITERIA}
    overall = round(sum(scores_10.values()) / len(CRITERIA), 2)
    return {"scores": scores_10, "overall": overall, "max": MAX_SCORE, "raw_avgs": raw_avgs}
