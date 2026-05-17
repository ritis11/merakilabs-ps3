from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agent.agent import AgentResponse, build_agent
from backend.agent.deps import AgentDeps
from backend.agent.output import Answer, Citation


@pytest.mark.asyncio
async def test_build_agent_returns_pydantic_ai_agent(monkeypatch):
    """Smoke test that the Agent constructs without error."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("TAVILY_API_KEY", "dummy")
    agent = build_agent()
    assert agent is not None


@pytest.mark.asyncio
async def test_run_agent_handles_usage_limit_exceeded(monkeypatch):
    """When UsageLimitExceeded is raised, return graceful AgentResponse."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("TAVILY_API_KEY", "dummy")

    from backend.agent import agent as agent_module
    from pydantic_ai.exceptions import UsageLimitExceeded

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(side_effect=UsageLimitExceeded("too many"))
    monkeypatch.setattr(agent_module, "_AGENT", fake_agent)

    deps = AgentDeps(session_id="s", vector_store=MagicMock(), web_search_client=MagicMock())
    resp = await agent_module.run_agent(message_history=[], user_message="hi", deps=deps)
    assert isinstance(resp, AgentResponse)
    assert "couldn't reach an answer" in resp.answer.text.lower() or "budget" in resp.answer.text.lower()
    assert resp.answer.requires_citation is False


@pytest.mark.asyncio
async def test_run_agent_returns_response_with_tool_calls(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("TAVILY_API_KEY", "dummy")

    from backend.agent import agent as agent_module

    answer = Answer(
        text="The answer is 42 [zomato_fy24.pdf, p. 1].",
        citations=[Citation(doc_name="zomato_fy24.pdf", page_number=1)],
        requires_citation=True,
        retrieval_iterations=1,
    )
    fake_run_result = MagicMock()
    fake_run_result.output = answer
    fake_run_result.all_messages.return_value = ["msg1", "msg2"]
    fake_run_result.usage.return_value = MagicMock(request_tokens=100, response_tokens=20, total_tokens=120)

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(return_value=fake_run_result)
    monkeypatch.setattr(agent_module, "_AGENT", fake_agent)

    deps = AgentDeps(session_id="s", vector_store=MagicMock(), web_search_client=MagicMock())
    resp = await agent_module.run_agent(message_history=[], user_message="q", deps=deps)
    assert resp.answer.text.startswith("The answer is 42")
    assert resp.answer.citations[0].doc_name == "zomato_fy24.pdf"


@pytest.mark.asyncio
async def test_run_agent_tool_calls_use_only_turn_suffix(monkeypatch):
    """Tool traces must not include prior turns' calls (full graph vs. this turn)."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("TAVILY_API_KEY", "dummy")

    from backend.agent import agent as agent_module

    captured: list[list] = []

    def spy_extract(msgs):
        captured.append(list(msgs))
        return []

    monkeypatch.setattr(agent_module, "_extract_tool_calls", spy_extract)

    answer = Answer(
        text="ok",
        citations=[],
        requires_citation=False,
        retrieval_iterations=0,
    )
    fake_run_result = MagicMock()
    fake_run_result.output = answer
    fake_run_result.all_messages.return_value = ["m0", "m1", "m2"]
    fake_run_result.usage.return_value = MagicMock(request_tokens=1, response_tokens=1, total_tokens=2)

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(return_value=fake_run_result)
    monkeypatch.setattr(agent_module, "_AGENT", fake_agent)

    deps = AgentDeps(session_id="s", vector_store=MagicMock(), web_search_client=MagicMock())
    await agent_module.run_agent(message_history=["m0", "m1"], user_message="q", deps=deps)
    assert captured == [["m2"]]

    captured.clear()
    await agent_module.run_agent(message_history=[], user_message="q", deps=deps)
    assert captured == [["m0", "m1", "m2"]]
