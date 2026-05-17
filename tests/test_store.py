from unittest.mock import MagicMock
import pytest
from backend.ingestion.models import Chunk
from backend.retrieval.store import ChromaStore


def fake_embedder(values_for):
    """Returns an embedder that produces deterministic 4-d vectors keyed by text."""
    def vec(text: str) -> list[float]:
        # Hash-derived but stable for given text
        h = abs(hash(text)) % 1000
        return [h / 1000.0, (h % 100) / 100.0, 0.0, 1.0]

    e = MagicMock()
    e.embed_documents.side_effect = lambda texts: [vec(t) for t in texts]
    e.embed_query.side_effect = lambda text: vec(text)
    return e


@pytest.fixture
def store(tmp_path):
    e = fake_embedder([])
    return ChromaStore(persist_dir=str(tmp_path / "chroma"), embedder=e)


def _chunks(doc_id, doc_name, n_pages=3):
    return [
        Chunk(
            chunk_id=f"{doc_id}_{i}",
            doc_id=doc_id,
            doc_name=doc_name,
            page_number=i,
            text=f"chunk text page {i} of {doc_name}",
            char_start=0,
            char_end=10,
        )
        for i in range(1, n_pages + 1)
    ]


def test_add_and_query_round_trip(store):
    chunks = _chunks("d1", "zomato_fy24.pdf")
    store.add_chunks("s1", chunks)
    results = store.query("s1", "page 2 of zomato", k=2)
    assert len(results) <= 2
    assert all(hasattr(r[0], "page_number") for r in results)
    assert all(0.0 <= r[1] <= 1.0 for r in results)


def test_session_isolation(store):
    store.add_chunks("s1", _chunks("d1", "zomato_fy24.pdf"))
    store.add_chunks("s2", _chunks("d2", "nykaa_fy24.pdf"))
    s1_results = store.query("s1", "anything", k=10)
    assert all(r[0].doc_name == "zomato_fy24.pdf" for r in s1_results)


def test_doc_filter(store):
    store.add_chunks("s1", _chunks("d1", "zomato_fy24.pdf"))
    store.add_chunks("s1", _chunks("d2", "nykaa_fy24.pdf"))
    out = store.query("s1", "anything", k=10, doc_filter=["nykaa_fy24.pdf"])
    assert len(out) > 0
    assert all(r[0].doc_name == "nykaa_fy24.pdf" for r in out)


def test_list_documents(store):
    store.add_chunks("s1", _chunks("d1", "zomato_fy24.pdf"))
    store.add_chunks("s1", _chunks("d2", "nykaa_fy24.pdf"))
    docs = store.list_documents("s1")
    names = sorted(d["doc_name"] for d in docs)
    assert names == ["nykaa_fy24.pdf", "zomato_fy24.pdf"]
    assert all(d["chunk_count"] > 0 for d in docs)


def test_delete_session(store):
    store.add_chunks("s1", _chunks("d1", "x.pdf"))
    store.delete_session("s1")
    # querying a deleted session should return empty without crashing
    assert store.query("s1", "x", k=5) == []
