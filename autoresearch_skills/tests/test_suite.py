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

# Ensure the package is importable
sys.path.insert(0, str(SKILLS_DIR.parent))

MOD_PREFIX = "autoresearch_skills"


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
    "BASE_DIR", "BEST_PROMPT_FILE", "RESULTS_FILE", "DIAGRAMS_DIR",
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


def test_dashboard_is_self_contained():
    tree = ast.parse((SKILLS_DIR / "dashboard.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("prepare", "train"):
            pytest.fail(f"dashboard.py should not import from {node.module}")


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


def test_eval_prompt_mentions_all_criteria():
    from autoresearch_skills.prepare import EVAL_PROMPT
    for keyword in ["LEGIBLE_AND_GRAMMATICAL", "PASTEL_COLORS", "LINEAR_LAYOUT", "NO_NUMBERS"]:
        assert keyword in EVAL_PROMPT


# ---------------------------------------------------------------------------
# prepare.py: score_batch
# ---------------------------------------------------------------------------

def test_score_batch_all_pass():
    from autoresearch_skills.prepare import score_batch
    results = [
        {"legible_and_grammatical": True, "pastel_colors": True, "linear_layout": True, "no_numbers": True}
        for _ in range(10)
    ]
    scores = score_batch(results)
    assert scores["total"] == 40
    assert scores["legible"] == 10
    assert scores["pastel"] == 10
    assert scores["linear"] == 10
    assert scores["no_numbers"] == 10


def test_score_batch_all_fail():
    from autoresearch_skills.prepare import score_batch
    results = [
        {"legible_and_grammatical": False, "pastel_colors": False, "linear_layout": False, "no_numbers": False}
        for _ in range(10)
    ]
    scores = score_batch(results)
    assert scores["total"] == 0


def test_score_batch_mixed():
    from autoresearch_skills.prepare import score_batch
    results = [
        {"legible_and_grammatical": True, "pastel_colors": False, "linear_layout": True, "no_numbers": False},
        {"legible_and_grammatical": False, "pastel_colors": True, "linear_layout": False, "no_numbers": True},
    ]
    scores = score_batch(results)
    assert scores["legible"] == 1
    assert scores["pastel"] == 1
    assert scores["linear"] == 1
    assert scores["no_numbers"] == 1
    assert scores["total"] == 4


def test_score_batch_empty():
    from autoresearch_skills.prepare import score_batch
    scores = score_batch([])
    assert scores["total"] == 0


def test_score_batch_missing_keys():
    from autoresearch_skills.prepare import score_batch
    results = [{"legible_and_grammatical": True}]
    scores = score_batch(results)
    assert scores["legible"] == 1
    assert scores["pastel"] == 0
    assert scores["total"] == 1


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
    save_state({"best_score": 32, "run_number": 5})
    state = load_state()
    assert state["best_score"] == 32
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
# prepare.py: evaluate_one (mocked API)
# ---------------------------------------------------------------------------

def test_evaluate_one_success(tmp_path):
    from autoresearch_skills.prepare import evaluate_one

    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps({
        "legible_and_grammatical": True,
        "pastel_colors": True,
        "linear_layout": False,
        "no_numbers": True,
        "failures": ["Layout is radial"]
    })

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    result = evaluate_one(mock_client, img_path)
    assert result is not None
    assert result["legible_and_grammatical"] is True
    assert result["linear_layout"] is False
    assert result["failures"] == ["Layout is radial"]


def test_evaluate_one_markdown_fenced(tmp_path):
    from autoresearch_skills.prepare import evaluate_one

    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    fenced = '```json\n{"legible_and_grammatical": true, "pastel_colors": true, "linear_layout": true, "no_numbers": true, "failures": []}\n```'
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = fenced

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    result = evaluate_one(mock_client, img_path)
    assert result is not None
    assert result["legible_and_grammatical"] is True


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
        {"legible_and_grammatical": True, "pastel_colors": False, "linear_layout": True, "no_numbers": False, "failures": ["dark blue fill", "Step 1 label"]},
        {"legible_and_grammatical": True, "pastel_colors": True, "linear_layout": False, "no_numbers": True, "failures": ["circular layout"]},
    ]

    result = mutate_prompt(mock_client, "current prompt", eval_results, 30)
    assert result == "improved prompt text"

    call_args = mock_client.messages.create.call_args
    prompt_text = call_args.kwargs["messages"][0]["content"]
    assert "current prompt" in prompt_text
    assert "30" in prompt_text
    assert "dark blue fill" in prompt_text


def test_mutate_prompt_deduplicates_failures():
    from autoresearch_skills.train import mutate_prompt

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "new prompt"

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    eval_results = [
        {"legible_and_grammatical": False, "pastel_colors": True, "linear_layout": True, "no_numbers": True, "failures": ["garbled text"]},
        {"legible_and_grammatical": False, "pastel_colors": True, "linear_layout": True, "no_numbers": True, "failures": ["garbled text"]},
    ]

    mutate_prompt(mock_client, "prompt", eval_results, 20)
    prompt_text = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert prompt_text.count("garbled text") == 1


# ---------------------------------------------------------------------------
# train.py: MUTATION_TEMPLATE format fields
# ---------------------------------------------------------------------------

def test_mutation_template_format_fields():
    from autoresearch_skills.train import MUTATION_TEMPLATE
    required_fields = ["current_prompt", "score", "leg_rate", "col_rate", "lin_rate", "num_rate", "best_score", "failures"]
    for field in required_fields:
        assert f"{{{field}}}" in MUTATION_TEMPLATE, f"MUTATION_TEMPLATE missing placeholder '{field}'"


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
    for section in ["## Setup", "## Experimentation", "## The experiment loop", "## Eval Criteria"]:
        assert section in content, f"program.md missing section: {section}"


# ---------------------------------------------------------------------------
# File structure sanity
# ---------------------------------------------------------------------------

def test_no_old_autoresearch_py():
    assert not (SKILLS_DIR / "autoresearch.py").exists(), "autoresearch.py should have been removed"


def test_no_old_skill_md():
    assert not (SKILLS_DIR / "SKILL.md").exists(), "SKILL.md should have been replaced by program.md"


def test_data_dir_exists():
    assert (SKILLS_DIR / "data").is_dir()


def test_data_files_present():
    data = SKILLS_DIR / "data"
    assert (data / "prompt.txt").exists()
    assert (data / "state.json").exists()
    assert (data / "results.jsonl").exists()
