from unittest.mock import MagicMock
import pytest
from pydantic_ai import RunContext
from backend.agent.deps import AgentDeps
from backend.agent.tools import retrieve_from_docs, list_available_documents
from backend.ingestion.models import Chunk


def _make_ctx(store_results=None, docs=None):
    store = MagicMock()
    store.query.return_value = store_results or []
    store.list_documents.return_value = docs or []
    deps = AgentDeps(
        session_id="s1",
        vector_store=store,
        web_search_client=MagicMock(),
        retrieval_threshold=0.3,
        retrieval_k=5,
    )
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    return ctx


@pytest.mark.asyncio
async def test_retrieve_returns_above_threshold():
    chunk = Chunk(chunk_id="c1", doc_id="d", doc_name="x.pdf", page_number=1, text="hello", char_start=0, char_end=5)
    ctx = _make_ctx(store_results=[(chunk, 0.9)])
    result = await retrieve_from_docs(ctx, "what is hello?")
    assert result.all_below_threshold is False
    assert len(result.chunks) == 1
    assert result.chunks[0].text == "hello"


@pytest.mark.asyncio
async def test_retrieve_below_threshold_flags():
    chunk = Chunk(chunk_id="c1", doc_id="d", doc_name="x.pdf", page_number=1, text="hello", char_start=0, char_end=5)
    ctx = _make_ctx(store_results=[(chunk, 0.1)])
    result = await retrieve_from_docs(ctx, "irrelevant")
    assert result.all_below_threshold is True
    assert result.chunks == []


@pytest.mark.asyncio
async def test_retrieve_passes_doc_filter():
    ctx = _make_ctx(store_results=[])
    await retrieve_from_docs(ctx, "q", doc_filter=["nykaa_fy24.pdf"])
    ctx.deps.vector_store.query.assert_called_once_with(
        "s1", "q", k=5, doc_filter=["nykaa_fy24.pdf"]
    )


@pytest.mark.asyncio
async def test_list_documents():
    ctx = _make_ctx(docs=[
        {"doc_id": "d1", "doc_name": "zomato_fy24.pdf", "chunk_count": 120},
        {"doc_id": "d2", "doc_name": "nykaa_fy24.pdf", "chunk_count": 95},
    ])
    docs = await list_available_documents(ctx)
    assert len(docs) == 2
    assert docs[0].doc_name in {"zomato_fy24.pdf", "nykaa_fy24.pdf"}
    assert docs[0].chunk_count > 0
