#!/usr/bin/env python3
"""
Fixed evaluation harness, constants, and utilities for diagram autoresearch.
Do not modify — this is the ground truth eval, analogous to prepare.py in
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

BASE_DIR = Path(__file__).resolve().parent / "data"
PROMPT_FILE = BASE_DIR / "prompt.txt"
BEST_PROMPT_FILE = BASE_DIR / "best_prompt.txt"
INITIAL_PROMPT_FILE = BASE_DIR / "initial_prompt.txt"
STATE_FILE = BASE_DIR / "state.json"
RESULTS_FILE = BASE_DIR / "results.jsonl"
DIAGRAMS_DIR = BASE_DIR / "diagrams"

BATCH_SIZE = 10
CYCLE_SECONDS = 120
MAX_GEN_WORKERS = 3
MAX_EVAL_WORKERS = 5

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

# ─── Eval Prompt (DO NOT CHANGE — this is the fixed metric) ─────────────────

EVAL_PROMPT = """You are evaluating a diagram image against 4 strict criteria. Examine the image carefully.

Criteria:
1. LEGIBLE_AND_GRAMMATICAL: ALL text in the diagram is clearly readable — no garbled, overlapping, blurry, or cut-off text. All words are real English words spelled correctly. Sentences/phrases are grammatically correct.

2. PASTEL_COLORS: The diagram uses ONLY soft pastel colors for fills (light purple, light blue, light green, light pink, light yellow, light teal, etc). No bright, saturated, neon, or dark-colored fills. White background counts as passing.

3. LINEAR_LAYOUT: The diagram flows in ONE clear linear direction — either strictly left-to-right OR strictly top-to-bottom. Not circular, radial, scattered, hub-and-spoke, or multi-directional.

4. NO_NUMBERS: The diagram contains ZERO numbers, step numbers, ordinals (1st, 2nd, 3rd), sequence indicators (Step 1, Phase 2), or any numerical ordering. Only text labels allowed.

Rate each criterion as PASS (true) or FAIL (false). Be strict.

Respond in this exact JSON format:
{"legible_and_grammatical": true, "pastel_colors": true, "linear_layout": true, "no_numbers": true, "failures": []}

If any criterion fails, set it to false and add a brief description to the failures array. Example:
{"legible_and_grammatical": false, "pastel_colors": true, "linear_layout": true, "no_numbers": false, "failures": ["Text 'Procssing' is misspelled", "Contains 'Step 1', 'Step 2' labels"]}"""

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
    """Evaluate a single diagram against 4 criteria via Claude vision."""
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
        return json.loads(text)
    except Exception as e:
        print(f"    EVAL ERROR: {e}")
        return None


def score_batch(eval_results: list[dict]) -> dict:
    """Compute per-criterion and total scores from a batch of eval results."""
    leg = sum(1 for r in eval_results if r.get("legible_and_grammatical"))
    col = sum(1 for r in eval_results if r.get("pastel_colors"))
    lin = sum(1 for r in eval_results if r.get("linear_layout"))
    num = sum(1 for r in eval_results if r.get("no_numbers"))
    return {"legible": leg, "pastel": col, "linear": lin, "no_numbers": num, "total": leg + col + lin + num}
