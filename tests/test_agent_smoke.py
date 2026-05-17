"""End-to-end smoke test of the agent layer.

Real Pydantic AI agent + real Gemini Flash + mocked vector store and Tavily.
Skips if GEMINI_API_KEY is not set.
"""
import os
import pytest
from unittest.mock import MagicMock
from backend.agent.agent import run_agent
from backend.agent.deps import AgentDeps
from backend.ingestion.models import Chunk


@pytest.mark.asyncio
async def test_agent_uses_retrieve_and_cites():
    if not os.environ.get("GEMINI_API_KEY") or os.environ["GEMINI_API_KEY"] == "dummy":
        pytest.skip("GEMINI_API_KEY not configured for live smoke test")

    chunk = Chunk(
        chunk_id="c1", doc_id="d1", doc_name="zomato_fy24.pdf",
        page_number=12,
        text="Zomato's adjusted revenue for FY24 was INR 12,114 crore, a 71% YoY increase.",
        char_start=0, char_end=80,
    )
    store = MagicMock()
    store.query.return_value = [(chunk, 0.91)]
    store.list_documents.return_value = [
        {"doc_id": "d1", "doc_name": "zomato_fy24.pdf", "chunk_count": 1}
    ]

    deps = AgentDeps(
        session_id="smoke",
        vector_store=store,
        web_search_client=MagicMock(),
    )

    resp = await run_agent(
        message_history=[],
        user_message="What was Zomato's adjusted revenue in FY24?",
        deps=deps,
    )

    assert resp.answer.requires_citation is True
    assert any(c.doc_name == "zomato_fy24.pdf" for c in resp.answer.citations if c.kind == "doc")
    # The agent should have called retrieve_from_docs at least once
    tool_names = [tc["tool_name"] for tc in resp.tool_calls]
    assert "retrieve_from_docs" in tool_names
