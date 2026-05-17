"""Retrieval recall@k metric. Computed against expected_pages in the dataset."""


def recall_at_k(retrieved_pages: list[int], expected_pages: list[int], k: int) -> float | None:
    """Fraction of expected_pages present in the top-k retrieved pages.

    Returns None when expected_pages is empty (the metric is N/A for that Q).
    """
    if not expected_pages:
        return None
    top = retrieved_pages[:k]
    hits = sum(1 for p in expected_pages if p in top)
    return hits / len(expected_pages)
