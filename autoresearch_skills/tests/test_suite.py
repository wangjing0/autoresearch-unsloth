"""
Tests for autoresearch-skills suite.
Verifies module structure, imports, constants, helpers, and scoring logic
without requiring API keys or network access.
"""

import ast
import json
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SKILLS_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(SKILLS_DIR.parent))

MOD_PREFIX = "autoresearch_skills"

CRITERIA_KEYS = ["text_quality", "color_palette", "layout", "label_discipline", "visual_clarity", "icon_quality"]


# ---------------------------------------------------------------------------
# Syntax checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ["prepare.py", "train.py", "dashboard.py"])
def test_syntax(filename):
    source = (SKILLS_DIR / filename).read_text()
    ast.parse(source)


# ---------------------------------------------------------------------------
# Import chain: train.py imports from prepare.py
# ---------------------------------------------------------------------------

EXPECTED_IMPORTS = [
    "GEMINI_KEY", "ANTHROPIC_KEY",
    "GEN_MODEL", "EVAL_MODEL", "MUTATE_MODEL",
    "BASE_DIR", "BEST_PROMPT_FILE", "INITIAL_PROMPT", "RESULTS_FILE", "FRONTIER_FILE", "DIAGRAMS_DIR",
    "BATCH_SIZE", "CYCLE_SECONDS", "MAX_GEN_WORKERS", "MAX_EVAL_WORKERS",
    "TOPICS",
    "evaluate_one", "score_batch",
    "load_state", "save_state", "load_prompt", "save_prompt",
]


def _get_prepare_exports():
    tree = ast.parse((SKILLS_DIR / "prepare.py").read_text())
    names = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def test_train_imports_resolve():
    exports = _get_prepare_exports()
    for name in EXPECTED_IMPORTS:
        assert name in exports, f"train.py imports '{name}' but prepare.py does not export it"


def test_dashboard_does_not_import_train():
    tree = ast.parse((SKILLS_DIR / "dashboard.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "train" in node.module:
            pytest.fail("dashboard.py should not import from train")


# ---------------------------------------------------------------------------
# prepare.py: constants
# ---------------------------------------------------------------------------

def test_constants():
    from autoresearch_skills.prepare import BATCH_SIZE, CYCLE_SECONDS, MAX_GEN_WORKERS, MAX_EVAL_WORKERS, TOPICS
    assert BATCH_SIZE == 10
    assert CYCLE_SECONDS == 120
    assert MAX_GEN_WORKERS >= 1
    assert MAX_EVAL_WORKERS >= 1
    assert len(TOPICS) == 30


def test_models():
    from autoresearch_skills.prepare import GEN_MODEL, EVAL_MODEL, MUTATE_MODEL
    assert "gemini" in GEN_MODEL
    assert "claude" in EVAL_MODEL
    assert "claude" in MUTATE_MODEL


def test_criteria_count():
    from autoresearch_skills.prepare import CRITERIA, MAX_SCORE
    assert len(CRITERIA) == 6
    assert MAX_SCORE == 10.0


def test_eval_prompt_mentions_all_criteria():
    from autoresearch_skills.prepare import EVAL_PROMPT
    for keyword in ["TEXT_QUALITY", "COLOR_PALETTE", "LAYOUT", "LABEL_DISCIPLINE", "VISUAL_CLARITY", "ICON_QUALITY"]:
        assert keyword in EVAL_PROMPT


# ---------------------------------------------------------------------------
# prepare.py: score_batch (graded 1-5 scale)
# ---------------------------------------------------------------------------

def _make_result(scores_dict):
    return {c: scores_dict.get(c, 1) for c in CRITERIA_KEYS}


def test_score_batch_all_perfect():
    from autoresearch_skills.prepare import score_batch
    results = [_make_result({c: 5 for c in CRITERIA_KEYS}) for _ in range(10)]
    scores = score_batch(results)
    assert scores["overall"] == 10.0
    assert scores["max"] == 10.0
    for c in CRITERIA_KEYS:
        assert scores["scores"][c] == 10.0


def test_score_batch_all_minimum():
    from autoresearch_skills.prepare import score_batch
    results = [_make_result({c: 1 for c in CRITERIA_KEYS}) for _ in range(10)]
    scores = score_batch(results)
    assert scores["overall"] == 0.0
    for c in CRITERIA_KEYS:
        assert scores["scores"][c] == 0.0


def test_score_batch_mixed():
    from autoresearch_skills.prepare import score_batch
    results = [
        _make_result({"text_quality": 5, "color_palette": 3, "layout": 4, "label_discipline": 2, "visual_clarity": 3, "icon_quality": 1}),
        _make_result({"text_quality": 3, "color_palette": 5, "layout": 2, "label_discipline": 4, "visual_clarity": 1, "icon_quality": 3}),
    ]
    scores = score_batch(results)
    assert scores["scores"]["text_quality"] == 7.5  # avg 4, (4-1)/4*10 = 7.5
    assert scores["scores"]["color_palette"] == 7.5
    assert scores["raw_avgs"]["text_quality"] == 4.0


def test_score_batch_empty():
    from autoresearch_skills.prepare import score_batch
    scores = score_batch([])
    assert scores["overall"] == 0.0


def test_score_batch_missing_keys_default_to_1():
    from autoresearch_skills.prepare import score_batch
    results = [{"text_quality": 5}]
    scores = score_batch(results)
    assert scores["scores"]["text_quality"] == 10.0
    assert scores["scores"]["color_palette"] == 0.0


# ---------------------------------------------------------------------------
# prepare.py: state helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path):
    with patch(f"{MOD_PREFIX}.prepare.STATE_FILE", tmp_path / "state.json"), \
         patch(f"{MOD_PREFIX}.prepare.PROMPT_FILE", tmp_path / "prompt.txt"), \
         patch(f"{MOD_PREFIX}.prepare.BEST_PROMPT_FILE", tmp_path / "best_prompt.txt"):
        yield tmp_path


def test_load_state_default(tmp_data_dir):
    from autoresearch_skills.prepare import load_state
    state = load_state()
    assert state == {"best_score": -1, "run_number": 0}


def test_save_load_state_roundtrip(tmp_data_dir):
    from autoresearch_skills.prepare import save_state, load_state
    save_state({"best_score": 180, "run_number": 5})
    state = load_state()
    assert state["best_score"] == 180
    assert state["run_number"] == 5


def test_save_load_prompt_roundtrip(tmp_data_dir):
    from autoresearch_skills.prepare import save_prompt, load_prompt
    save_prompt("test prompt content")
    assert load_prompt() == "test prompt content"


def test_load_prompt_strips_whitespace(tmp_data_dir):
    from autoresearch_skills.prepare import load_prompt, PROMPT_FILE
    PROMPT_FILE.write_text("  prompt with spaces  \n\n")
    assert load_prompt() == "prompt with spaces"


# ---------------------------------------------------------------------------
# prepare.py: evaluate_one (mocked API, graded response)
# ---------------------------------------------------------------------------

def test_evaluate_one_success(tmp_path):
    from autoresearch_skills.prepare import evaluate_one

    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps({
        "text_quality": 4, "color_palette": 5, "layout": 3,
        "label_discipline": 4, "visual_clarity": 3, "icon_quality": 2,
        "failures": ["Icons inconsistent"]
    })

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    result = evaluate_one(mock_client, img_path)
    assert result is not None
    assert result["text_quality"] == 4
    assert result["icon_quality"] == 2
    assert result["failures"] == ["Icons inconsistent"]


def test_evaluate_one_clamps_scores(tmp_path):
    from autoresearch_skills.prepare import evaluate_one

    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps({
        "text_quality": 7, "color_palette": 0, "layout": 3,
        "label_discipline": 4, "visual_clarity": 3, "icon_quality": -1,
        "failures": []
    })

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    result = evaluate_one(mock_client, img_path)
    assert result["text_quality"] == 5
    assert result["color_palette"] == 1
    assert result["icon_quality"] == 1


def test_evaluate_one_markdown_fenced(tmp_path):
    from autoresearch_skills.prepare import evaluate_one

    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    fenced = '```json\n{"text_quality": 4, "color_palette": 4, "layout": 4, "label_discipline": 4, "visual_clarity": 4, "icon_quality": 4, "failures": []}\n```'
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = fenced

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    result = evaluate_one(mock_client, img_path)
    assert result is not None
    assert result["text_quality"] == 4


def test_evaluate_one_api_error(tmp_path):
    from autoresearch_skills.prepare import evaluate_one

    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API timeout")

    result = evaluate_one(mock_client, img_path)
    assert result is None


# ---------------------------------------------------------------------------
# train.py: mutate_prompt (mocked API)
# ---------------------------------------------------------------------------

def test_mutate_prompt():
    from autoresearch_skills.train import mutate_prompt

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "improved prompt text"

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    eval_results = [
        {"text_quality": 4, "color_palette": 2, "layout": 4, "label_discipline": 3, "visual_clarity": 3, "icon_quality": 2, "failures": ["dark blue fill", "Step 1 label"]},
        {"text_quality": 3, "color_palette": 4, "layout": 2, "label_discipline": 4, "visual_clarity": 2, "icon_quality": 3, "failures": ["circular layout"]},
    ]

    result = mutate_prompt(mock_client, "current prompt", eval_results, 150)
    assert result == "improved prompt text"

    call_args = mock_client.messages.create.call_args
    prompt_text = call_args.kwargs["messages"][0]["content"]
    assert "current prompt" in prompt_text
    assert "dark blue fill" in prompt_text


def test_mutate_prompt_deduplicates_failures():
    from autoresearch_skills.train import mutate_prompt

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "new prompt"

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    eval_results = [
        {"text_quality": 2, "color_palette": 4, "layout": 4, "label_discipline": 4, "visual_clarity": 3, "icon_quality": 3, "failures": ["garbled text"]},
        {"text_quality": 2, "color_palette": 4, "layout": 4, "label_discipline": 4, "visual_clarity": 3, "icon_quality": 3, "failures": ["garbled text"]},
    ]

    mutate_prompt(mock_client, "prompt", eval_results, 100)
    prompt_text = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert prompt_text.count("garbled text") == 1


# ---------------------------------------------------------------------------
# train.py: template format fields
# ---------------------------------------------------------------------------

def test_refine_template_format_fields():
    from autoresearch_skills.train import REFINE_TEMPLATE
    required_fields = ["current_prompt", "overall", "failures", "focus_instructions", "frontier_context", "feedback_history"]
    for field in required_fields:
        assert f"{{{field}}}" in REFINE_TEMPLATE, f"REFINE_TEMPLATE missing placeholder '{field}'"
    for c in CRITERIA_KEYS:
        assert f"{{s_{c}}}" in REFINE_TEMPLATE, f"REFINE_TEMPLATE missing criterion 's_{c}'"


def test_explore_template_format_fields():
    from autoresearch_skills.train import EXPLORE_TEMPLATE
    required_fields = ["current_prompt", "overall", "failures", "frontier_context", "feedback_history"]
    for field in required_fields:
        assert f"{{{field}}}" in EXPLORE_TEMPLATE, f"EXPLORE_TEMPLATE missing placeholder '{field}'"


# ---------------------------------------------------------------------------
# train.py: Pareto front (uses new 6 criteria)
# ---------------------------------------------------------------------------

def _make_frontier_entry(**overrides):
    base = {c: 30 for c in CRITERIA_KEYS}
    base.update({"prompt": "test", "total": sum(base.values()), "run": 1})
    base.update(overrides)
    return base


def test_pareto_dominates():
    from autoresearch_skills.train import dominates
    a = _make_frontier_entry(text_quality=40)
    b = _make_frontier_entry(text_quality=30)
    assert dominates(a, b)
    assert not dominates(b, a)
    assert not dominates(a, a)


def test_pareto_non_dominated():
    from autoresearch_skills.train import dominates
    a = _make_frontier_entry(text_quality=40, color_palette=20)
    b = _make_frontier_entry(text_quality=20, color_palette=40)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_update_frontier_adds_non_dominated():
    from autoresearch_skills.train import update_frontier
    frontier = [_make_frontier_entry(text_quality=20, color_palette=40)]
    candidate = _make_frontier_entry(text_quality=40, color_palette=20)
    new_frontier, added = update_frontier(frontier, candidate)
    assert added
    assert len(new_frontier) == 2


def test_update_frontier_rejects_dominated():
    from autoresearch_skills.train import update_frontier
    frontier = [_make_frontier_entry(text_quality=50)]
    candidate = _make_frontier_entry(text_quality=30)
    new_frontier, added = update_frontier(frontier, candidate)
    assert not added
    assert len(new_frontier) == 1


def test_update_frontier_prunes_dominated():
    from autoresearch_skills.train import update_frontier
    frontier = [_make_frontier_entry(text_quality=30)]
    candidate = _make_frontier_entry(text_quality=50)
    new_frontier, added = update_frontier(frontier, candidate)
    assert added
    assert len(new_frontier) == 1
    assert new_frontier[0]["text_quality"] == 50


def test_update_frontier_allows_same_scores_different_prompts():
    from autoresearch_skills.train import update_frontier
    frontier = [_make_frontier_entry(prompt="prompt A")]
    candidate = _make_frontier_entry(prompt="prompt B")
    new_frontier, added = update_frontier(frontier, candidate)
    assert added
    assert len(new_frontier) == 2


def test_update_frontier_rejects_true_duplicate():
    from autoresearch_skills.train import update_frontier
    frontier = [_make_frontier_entry(prompt="same prompt")]
    candidate = _make_frontier_entry(prompt="same prompt")
    new_frontier, added = update_frontier(frontier, candidate)
    assert not added
    assert len(new_frontier) == 1


# ---------------------------------------------------------------------------
# train.py: plateau detection
# ---------------------------------------------------------------------------

def test_detect_plateau_no_streak():
    from autoresearch_skills.train import detect_plateau
    state = {"best_score": 5.0, "run_number": 3, "plateau_streak": 0}
    assert not detect_plateau(state, window=3)


def test_detect_plateau_streak_below_window():
    from autoresearch_skills.train import detect_plateau
    state = {"best_score": 5.0, "run_number": 5, "plateau_streak": 2}
    assert not detect_plateau(state, window=3)


def test_detect_plateau_streak_at_window():
    from autoresearch_skills.train import detect_plateau
    state = {"best_score": 5.0, "run_number": 6, "plateau_streak": 3}
    assert detect_plateau(state, window=3)


def test_detect_plateau_streak_above_window():
    from autoresearch_skills.train import detect_plateau
    state = {"best_score": 5.0, "run_number": 10, "plateau_streak": 7}
    assert detect_plateau(state, window=3)


def test_detect_plateau_missing_key():
    from autoresearch_skills.train import detect_plateau
    state = {"best_score": 5.0, "run_number": 3}
    assert not detect_plateau(state, window=3)


# ---------------------------------------------------------------------------
# program.md: structure check
# ---------------------------------------------------------------------------

def test_program_md_exists():
    assert (SKILLS_DIR / "program.md").exists()


def test_program_md_references_correct_files():
    content = (SKILLS_DIR / "program.md").read_text()
    assert "prepare.py" in content
    assert "train.py" in content
    assert "dashboard.py" in content
    assert "autoresearch.py" not in content


def test_program_md_has_key_sections():
    content = (SKILLS_DIR / "program.md").read_text()
    for section in ["## Setup", "## Experimentation", "## Optimization Architecture", "## Eval Criteria"]:
        assert section in content, f"program.md missing section: {section}"


# ---------------------------------------------------------------------------
# File structure sanity
# ---------------------------------------------------------------------------

def test_no_old_autoresearch_py():
    assert not (SKILLS_DIR / "autoresearch.py").exists(), "autoresearch.py should have been removed"


def test_no_old_skill_md():
    assert not (SKILLS_DIR / "SKILL.md").exists(), "SKILL.md should have been replaced by program.md"


def test_state_dir_exists():
    assert (SKILLS_DIR / "state").is_dir()


def test_state_files_present():
    state_dir = SKILLS_DIR / "state"
    assert (state_dir / "prompt.txt").exists()
    assert (state_dir / "state.json").exists()
    assert (state_dir / "results.jsonl").exists()


# ---------------------------------------------------------------------------
# train.py: feedback history
# ---------------------------------------------------------------------------

def test_read_feedback_history_no_file(tmp_path):
    from autoresearch_skills.train import read_feedback_history
    with patch("autoresearch_skills.train.FEEDBACK_FILE", tmp_path / "feedback_history.jsonl"):
        result = read_feedback_history()
    assert result == "No previous attempts."


def test_append_and_read_feedback_history(tmp_path):
    from autoresearch_skills.train import append_feedback, read_feedback_history
    fb_file = tmp_path / "feedback_history.jsonl"
    with patch("autoresearch_skills.train.FEEDBACK_FILE", fb_file):
        append_feedback(5, "REFINE", "color_palette", "test prompt text here", True, 8.5, 8.2)
        result = read_feedback_history()
    assert "Run 5" in result
    assert "REFINE" in result
    assert "color_palette" in result
    assert "ADDED TO FRONTIER" in result


def test_append_feedback_not_added(tmp_path):
    from autoresearch_skills.train import append_feedback, read_feedback_history
    fb_file = tmp_path / "feedback_history.jsonl"
    with patch("autoresearch_skills.train.FEEDBACK_FILE", fb_file):
        append_feedback(3, "EXPLORE", "layout", "some prompt", False, 7.9, 8.2)
        result = read_feedback_history()
    assert "not added" in result
    assert "-0.30" in result


def test_read_feedback_history_returns_last_n(tmp_path):
    from autoresearch_skills.train import append_feedback, read_feedback_history
    fb_file = tmp_path / "feedback_history.jsonl"
    with patch("autoresearch_skills.train.FEEDBACK_FILE", fb_file):
        for i in range(15):
            append_feedback(i + 1, "REFINE", "layout", f"prompt {i}", False, 7.0, 7.0)
        result = read_feedback_history(n=5)
    lines = [l for l in result.split("\n") if l.startswith("- Run")]
    assert len(lines) == 5
    assert "Run 15" in result
    assert "Run 1 |" not in result


# ---------------------------------------------------------------------------
# train.py: round-robin topic sampling
# ---------------------------------------------------------------------------

def test_round_robin_covers_all_topics():
    from autoresearch_skills.prepare import TOPICS, BATCH_SIZE
    from autoresearch_skills.train import ADVERSARIAL_TOPIC_COUNT
    n_standard = BATCH_SIZE - ADVERSARIAL_TOPIC_COUNT
    n_topics = len(TOPICS)
    cycles_to_cover = -(-n_topics // n_standard)  # ceil division
    seen = set()
    offset = 0
    for _ in range(cycles_to_cover):
        batch = [TOPICS[(offset + i) % n_topics] for i in range(n_standard)]
        seen.update(batch)
        offset = (offset + n_standard) % n_topics
    assert seen == set(TOPICS), "Round-robin should cover all topics within expected cycles"


def test_round_robin_deterministic():
    from autoresearch_skills.prepare import TOPICS, BATCH_SIZE
    from autoresearch_skills.train import ADVERSARIAL_TOPIC_COUNT
    n_standard = BATCH_SIZE - ADVERSARIAL_TOPIC_COUNT
    n_topics = len(TOPICS)
    offset = 0
    batch_a = [TOPICS[(offset + i) % n_topics] for i in range(n_standard)]
    batch_b = [TOPICS[(offset + i) % n_topics] for i in range(n_standard)]
    assert batch_a == batch_b, "Same offset must always produce same batch"


# ---------------------------------------------------------------------------
# train.py: _mutate_with_fallback
# ---------------------------------------------------------------------------

def test_mutate_with_fallback_returns_good_result():
    from autoresearch_skills.train import _mutate_with_fallback

    good_prompt = "a" * 100
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = good_prompt

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    eval_results = [{"text_quality": 3, "color_palette": 2, "layout": 3, "label_discipline": 3, "visual_clarity": 3, "icon_quality": 3, "failures": []}]
    result = _mutate_with_fallback(mock_client, "original", eval_results, 7.0, False, None, [], "")
    assert result == good_prompt
    assert mock_client.messages.create.call_count == 1


def test_mutate_with_fallback_retries_on_short_output():
    from autoresearch_skills.train import _mutate_with_fallback

    good_prompt = "b" * 100
    responses = [MagicMock(), MagicMock()]
    responses[0].content = [MagicMock()]
    responses[0].content[0].text = "too short"
    responses[1].content = [MagicMock()]
    responses[1].content[0].text = good_prompt

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = responses

    eval_results = [{"text_quality": 3, "color_palette": 2, "layout": 3, "label_discipline": 3, "visual_clarity": 3, "icon_quality": 3, "failures": []}]
    result = _mutate_with_fallback(mock_client, "original", eval_results, 7.0, False, None, [], "")
    assert result == good_prompt
    assert mock_client.messages.create.call_count == 2


def test_mutate_with_fallback_returns_original_when_all_fail():
    from autoresearch_skills.train import _mutate_with_fallback

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = ""

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    eval_results = [{"text_quality": 3, "color_palette": 2, "layout": 3, "label_discipline": 3, "visual_clarity": 3, "icon_quality": 3, "failures": []}]
    result = _mutate_with_fallback(mock_client, "original prompt", eval_results, 7.0, False, None, [], "")
    assert result == "original prompt"
    assert mock_client.messages.create.call_count == 3
