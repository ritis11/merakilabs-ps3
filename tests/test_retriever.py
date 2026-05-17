from unittest.mock import MagicMock
from backend.ingestion.models import Chunk
from backend.retrieval.retriever import retrieve, RetrievalResult, ChunkResult


def make_store(results):
    s = MagicMock()
    s.query.return_value = results
    return s


def _chunk(text="hello", doc_name="d.pdf", page=1, cid="c1"):
    return Chunk(chunk_id=cid, doc_id="d", doc_name=doc_name, page_number=page, text=text, char_start=0, char_end=len(text))


def test_retrieve_returns_chunks_above_threshold():
    store = make_store([(_chunk(), 0.9), (_chunk(cid="c2"), 0.5)])
    out = retrieve(store, "s1", "q", k=5, threshold=0.3)
    assert isinstance(out, RetrievalResult)
    assert out.all_below_threshold is False
    assert len(out.chunks) == 2
    assert all(isinstance(c, ChunkResult) for c in out.chunks)


def test_retrieve_flags_below_threshold():
    store = make_store([(_chunk(), 0.1), (_chunk(cid="c2"), 0.05)])
    out = retrieve(store, "s1", "q", k=5, threshold=0.3)
    assert out.all_below_threshold is True
    assert out.chunks == []


def test_retrieve_passes_doc_filter():
    store = make_store([(_chunk(doc_name="zomato_fy24.pdf"), 0.8)])
    out = retrieve(store, "s1", "q", k=5, threshold=0.3, doc_filter=["zomato_fy24.pdf"])
    assert out.doc_filter_used == ["zomato_fy24.pdf"]
    store.query.assert_called_once_with("s1", "q", k=5, doc_filter=["zomato_fy24.pdf"])


def test_retrieve_handles_empty_store():
    store = make_store([])
    out = retrieve(store, "s1", "q", k=5, threshold=0.3)
    assert out.all_below_threshold is True
    assert out.chunks == []
