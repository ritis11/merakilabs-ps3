from unittest.mock import MagicMock
import pytest
from eval.metrics.faithfulness import faithfulness_score, JUDGE_PROMPT, FAITHFULNESS_PROMPT_VERSION


def test_judge_prompt_pinned():
    assert FAITHFULNESS_PROMPT_VERSION
    assert "supported" in JUDGE_PROMPT.lower()


def test_faithfulness_score_parses_judge_json(monkeypatch):
    fake_client = MagicMock()
    response = MagicMock()
    response.text = '{"score": 0.5, "unsupported_claims": ["X is 3"]}'
    fake_client.models.generate_content.return_value = response
    monkeypatch.setattr(
        "eval.metrics.faithfulness._build_client",
        lambda key, model: fake_client,
    )
    result = faithfulness_score(
        api_key="test", judge_model="gemini-2.5-pro",
        chunks=[{"text": "X is 5", "doc_name": "a.pdf", "page_number": 1}],
        answer_text="X is 3 [a.pdf, p. 1].",
    )
    assert result.score == 0.5
    assert "X is 3" in result.unsupported_claims


def test_faithfulness_score_handles_malformed_judge_output(monkeypatch):
    fake_client = MagicMock()
    response = MagicMock()
    response.text = "not json"
    fake_client.models.generate_content.return_value = response
    monkeypatch.setattr(
        "eval.metrics.faithfulness._build_client",
        lambda key, model: fake_client,
    )
    result = faithfulness_score(
        api_key="test", judge_model="gemini-2.5-pro",
        chunks=[],
        answer_text="anything",
    )
    assert result.score == 0.0
