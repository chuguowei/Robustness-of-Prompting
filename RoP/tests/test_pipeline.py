"""
tests/test_pipeline.py

Unit tests for core modules (no API calls required).
"""

import pytest
from rop.correction.corrector import _clean_llm_output, _restore_answer_choices
from rop.guidance.answerer import extract_multiple_choice, extract_numeric
from rop.utils.metrics import accuracy


# ---------------------------------------------------------------------------
# Corrector helpers
# ---------------------------------------------------------------------------

class TestCleanLLMOutput:
    def test_strips_fences(self):
        text = "Intro\n---\nClean body.\n---\nOutro"
        assert _clean_llm_output(text) == "Clean body."

    def test_passthrough_plain_text(self):
        text = "Geb is 10 less than half the age of Haley."
        assert _clean_llm_output(text) == text

    def test_collapses_whitespace(self):
        text = "---\nLine one.\nLine two.\n---"
        result = _clean_llm_output(text)
        assert "\n" not in result


class TestRestoreAnswerChoices:
    def test_does_not_duplicate_choices(self):
        corrected = "Question text. Answer Choices: (A) 1 (B) 2"
        result = _restore_answer_choices(corrected, "original")
        assert result == corrected

    def test_appends_missing_choices(self):
        corrected = "Question text."
        original = "Question text. Answer Choices: (A) 1 (B) 2"
        result = _restore_answer_choices(corrected, original)
        assert "Answer Choices:" in result


# ---------------------------------------------------------------------------
# Answerer helpers
# ---------------------------------------------------------------------------

class TestExtractMultipleChoice:
    def test_boxed_letter(self):
        assert extract_multiple_choice(r"The answer is \boxed{B}.", "") == "B"

    def test_last_letter_fallback(self):
        response = "After calculation, the answer is C."
        result = extract_multiple_choice(response, "(A) 10 (B) 20 (C) 30")
        assert result == "C"


class TestExtractNumeric:
    def test_plain_number(self):
        assert extract_numeric("The answer is 42.") == "42"

    def test_fraction(self):
        result = extract_numeric("Therefore 12/4 is the result.")
        assert result == "3.0"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_perfect(self):
        assert accuracy(["A", "B", "C"], ["A", "B", "C"]) == 1.0

    def test_half(self):
        assert accuracy(["A", "X"], ["A", "B"]) == 0.5

    def test_ignores_none(self):
        # None predictions are skipped
        assert accuracy([None, "B"], ["A", "B"], ignore_empty=True) == 1.0

    def test_empty(self):
        assert accuracy([], []) == 0.0
