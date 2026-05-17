from unittest.mock import MagicMock
import pytest
from eval.metrics.refusal import is_refusal
from eval.metrics.clarification import is_clarification


def test_keyword_refusal_obvious():
    assert is_refusal("I don't have information about that in the uploaded documents.") is True
    assert is_refusal("I cannot answer questions about live stock prices.") is True


def test_keyword_refusal_negative():
    assert is_refusal("Zomato's adjusted revenue in FY24 was 12,114 crore.") is False


def test_clarification_keyword_obvious():
    assert is_clarification("Did you mean FY24 (Apr 2023 - Mar 2024) or calendar year 2024?") is True
    assert is_clarification("Could you clarify whether you mean fiscal year or calendar year?") is True


def test_clarification_keyword_negative():
    assert is_clarification("The revenue was 12,114 crore.") is False
