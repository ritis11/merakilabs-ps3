"""Tool-selection metric. Match actual tool_calls against expected_tools."""
from collections import Counter


def tool_selection_match(
    actual_tools: list[str],
    expected_tools: list[str],
    loose_multi_retrieve: bool = False,
) -> bool:
    """True if actual matches expected.

    - When `loose_multi_retrieve=True` (multi-hop questions), multiple
      retrieve_from_docs calls collapse to one for the comparison.
    - Otherwise, multisets must match exactly.
    - Extra unexpected tools (e.g. unwarranted google_search) cause a miss.
    """
    a = Counter(actual_tools)
    e = Counter(expected_tools)
    if loose_multi_retrieve:
        if a.get("retrieve_from_docs", 0) >= 1:
            a["retrieve_from_docs"] = 1
        if e.get("retrieve_from_docs", 0) >= 1:
            e["retrieve_from_docs"] = 1
    return a == e
