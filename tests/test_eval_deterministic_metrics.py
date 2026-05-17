from eval.metrics.tool_selection import tool_selection_match
from eval.metrics.doc_filter_correct import doc_filter_correct
from eval.metrics.iterative_retrieval_used import iterative_retrieval_used


def test_tool_selection_exact_match():
    assert tool_selection_match(["retrieve_from_docs", "calculate"], ["calculate", "retrieve_from_docs"]) is True


def test_tool_selection_subset_loose_for_multi_hop():
    assert tool_selection_match(
        ["retrieve_from_docs", "retrieve_from_docs"],
        ["retrieve_from_docs"],
        loose_multi_retrieve=True,
    ) is True


def test_tool_selection_missing_required():
    assert tool_selection_match(["retrieve_from_docs"], ["retrieve_from_docs", "calculate"]) is False


def test_tool_selection_unexpected_extra():
    assert tool_selection_match(["retrieve_from_docs", "google_search"], ["retrieve_from_docs"]) is False


def test_doc_filter_correct_when_expected_and_used():
    tool_calls = [{"tool_name": "retrieve_from_docs", "args": {"doc_filter": ["nykaa_fy24.pdf"]}}]
    assert doc_filter_correct(tool_calls, expected_doc_filter=["nykaa_fy24.pdf"]) is True


def test_doc_filter_correct_when_not_expected():
    assert doc_filter_correct([], expected_doc_filter=None) is True


def test_doc_filter_incorrect_when_expected_and_missing():
    tool_calls = [{"tool_name": "retrieve_from_docs", "args": {}}]
    assert doc_filter_correct(tool_calls, expected_doc_filter=["nykaa_fy24.pdf"]) is False


def test_iterative_retrieval_detected():
    tool_calls = [
        {"tool_name": "retrieve_from_docs"},
        {"tool_name": "retrieve_from_docs"},
        {"tool_name": "calculate"},
    ]
    assert iterative_retrieval_used(tool_calls) is True


def test_iterative_retrieval_not_used():
    tool_calls = [{"tool_name": "retrieve_from_docs"}]
    assert iterative_retrieval_used(tool_calls) is False
