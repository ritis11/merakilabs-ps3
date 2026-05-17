"""Pure retrieval function. Wrapped by the agent tool in backend/agent/tools.py."""
from pydantic import BaseModel


class ChunkResult(BaseModel):
    text: str
    doc_name: str
    page_number: int
    score: float


class RetrievalResult(BaseModel):
    chunks: list[ChunkResult]
    all_below_threshold: bool
    doc_filter_used: list[str] | None


def retrieve(
    store,
    session_id: str,
    query: str,
    k: int = 5,
    threshold: float = 0.3,
    doc_filter: list[str] | None = None,
) -> RetrievalResult:
    """Wrap ChromaStore.query with threshold logic and Pydantic shape.

    If every result is below the cosine threshold, returns
    `all_below_threshold=True` with empty chunks. The agent uses this signal to
    refuse / route to web search rather than hallucinating.
    """
    raw = store.query(session_id, query, k=k, doc_filter=doc_filter)
    above = [(c, s) for c, s in raw if s >= threshold]
    if not above:
        return RetrievalResult(chunks=[], all_below_threshold=True, doc_filter_used=doc_filter)
    return RetrievalResult(
        chunks=[ChunkResult(text=c.text, doc_name=c.doc_name, page_number=c.page_number, score=s) for c, s in above],
        all_below_threshold=False,
        doc_filter_used=doc_filter,
    )
