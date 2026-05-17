"""Did the agent pass the expected doc_filter when it should have?"""


def doc_filter_correct(
    tool_calls: list[dict],
    expected_doc_filter: list[str] | None,
) -> bool:
    """If expected_doc_filter is None, any behavior is OK (returns True).
    Otherwise, at least one retrieve_from_docs call must include
    doc_filter == expected_doc_filter (order-insensitive)."""
    if not expected_doc_filter:
        return True
    expected_set = set(expected_doc_filter)
    for tc in tool_calls:
        if tc.get("tool_name") != "retrieve_from_docs":
            continue
        args = tc.get("args") or {}
        df = args.get("doc_filter")
        if df and set(df) == expected_set:
            return True
    return False
