from unittest.mock import MagicMock
import pytest
from eval.metrics.task_completion import task_completed, JUDGE_TASK_PROMPT


def test_judge_task_prompt_present():
    assert "ground truth" in JUDGE_TASK_PROMPT.lower()


def test_completed_when_all_three_pass(monkeypatch):
    monkeypatch.setattr("eval.metrics.task_completion._judge_pass", lambda **k: True)
    out = task_completed(
        question="q",
        ground_truth="gt",
        answer_text="answer [a.pdf, p. 1].",
        actual_tools=["retrieve_from_docs"],
        expected_tools=["retrieve_from_docs"],
        category="single_hop_factual",
        api_key="x",
        judge_model="gemini-2.5-pro",
    )
    assert out.completed is True
    assert out.judge_pass is True
    assert out.tools_match is True
    assert out.has_citation is True


def test_not_completed_when_judge_fails(monkeypatch):
    monkeypatch.setattr("eval.metrics.task_completion._judge_pass", lambda **k: False)
    out = task_completed(
        question="q", ground_truth="gt",
        answer_text="answer [a.pdf, p. 1].",
        actual_tools=["retrieve_from_docs"], expected_tools=["retrieve_from_docs"],
        category="single_hop_factual",
        api_key="x", judge_model="gemini-2.5-pro",
    )
    assert out.completed is False


def test_not_completed_when_tools_mismatch(monkeypatch):
    monkeypatch.setattr("eval.metrics.task_completion._judge_pass", lambda **k: True)
    out = task_completed(
        question="q", ground_truth="gt",
        answer_text="answer [a.pdf, p. 1].",
        actual_tools=["retrieve_from_docs"],
        expected_tools=["retrieve_from_docs", "calculate"],
        category="numerical",
        api_key="x", judge_model="gemini-2.5-pro",
    )
    assert out.completed is False
    assert out.tools_match is False


def test_not_completed_when_no_citation(monkeypatch):
    monkeypatch.setattr("eval.metrics.task_completion._judge_pass", lambda **k: True)
    out = task_completed(
        question="q", ground_truth="gt",
        answer_text="answer with no citation marker",
        actual_tools=["retrieve_from_docs"], expected_tools=["retrieve_from_docs"],
        category="single_hop_factual",
        api_key="x", judge_model="gemini-2.5-pro",
    )
    assert out.completed is False
    assert out.has_citation is False


def test_meta_questions_dont_require_citation(monkeypatch):
    monkeypatch.setattr("eval.metrics.task_completion._judge_pass", lambda **k: True)
    out = task_completed(
        question="What docs do you have?", ground_truth="lists docs",
        answer_text="I have zomato_fy24.pdf and nykaa_fy24.pdf available.",
        actual_tools=["list_available_documents"],
        expected_tools=["list_available_documents"],
        category="meta",
        api_key="x", judge_model="gemini-2.5-pro",
    )
    assert out.completed is True
