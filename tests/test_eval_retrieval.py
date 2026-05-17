import pytest
from eval.metrics.retrieval import recall_at_k


def test_recall_at_k_hits_at_top_1():
    retrieved_pages = [12, 5, 8]
    expected = [12]
    assert recall_at_k(retrieved_pages, expected, k=1) == 1.0
    assert recall_at_k(retrieved_pages, expected, k=3) == 1.0


def test_recall_at_k_misses_at_top_1_hits_at_top_5():
    retrieved_pages = [99, 12, 5, 8, 7]
    expected = [12]
    assert recall_at_k(retrieved_pages, expected, k=1) == 0.0
    assert recall_at_k(retrieved_pages, expected, k=5) == 1.0


def test_recall_at_k_partial():
    retrieved_pages = [12, 5, 99]
    expected = [12, 5, 7]
    assert recall_at_k(retrieved_pages, expected, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_handles_empty_expected():
    # No expected pages = N/A; return None so it's excluded from aggregates
    assert recall_at_k([1, 2, 3], [], k=3) is None
