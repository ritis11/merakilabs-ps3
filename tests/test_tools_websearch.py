from unittest.mock import MagicMock
import pytest
from pydantic_ai import RunContext
from backend.agent.deps import AgentDeps
from backend.agent.tools import google_search


def _ctx(tavily_response=None):
    web = MagicMock()
    web.search.return_value = tavily_response or {"results": []}
    deps = AgentDeps(
        session_id="s1",
        vector_store=MagicMock(),
        web_search_client=web,
    )
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    return ctx


@pytest.mark.asyncio
async def test_google_search_normalizes_results():
    ctx = _ctx({
        "results": [
            {"title": "Tesla shares", "url": "https://bloomberg.com/x", "content": "Tesla dropped 5%."},
            {"title": "Tesla news", "url": "https://reuters.com/y", "content": "Tesla news ..."},
        ]
    })
    out = await google_search(ctx, "tesla stock", used_because="below_threshold")
    assert len(out.sources) == 2
    assert out.sources[0].url.startswith("https://")
    assert out.used_because == "below_threshold"


@pytest.mark.asyncio
async def test_google_search_handles_empty_results():
    ctx = _ctx({"results": []})
    out = await google_search(ctx, "nonsense xyz", used_because="user_request")
    assert out.sources == []
    assert out.used_because == "user_request"


@pytest.mark.asyncio
async def test_google_search_handles_provider_error():
    ctx = _ctx()
    ctx.deps.web_search_client.search.side_effect = RuntimeError("tavily down")
    out = await google_search(ctx, "anything", used_because="user_request")
    assert out.sources == []
    assert out.error is not None and "tavily" in out.error.lower()
