from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from backend.agent.agent import AgentResponse
from backend.agent.output import Answer, Citation
from backend.routes.messages import router, get_session_manager, get_vector_store, get_web_search_client, run_agent_dep
from backend.session.store import SessionManager


def _make_response(message_history=None):
    answer = Answer(
        text="Revenue was 12,114 crore [zomato_fy24.pdf, p. 12].",
        citations=[Citation(doc_name="zomato_fy24.pdf", page_number=12)],
        requires_citation=True,
        retrieval_iterations=1,
    )
    return AgentResponse(
        answer=answer,
        message_history=message_history if message_history is not None else [],
        tool_calls=[{"tool_name": "retrieve_from_docs", "args": {"query": "revenue"}, "result": {"chunks": []}}],
        usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )


@pytest.fixture
def app(tmp_path):
    sm = SessionManager(dump_path=str(tmp_path / "s.json"), history_cap=10)
    vs = MagicMock()
    web = MagicMock()

    # Default fake_run returns a response with valid ModelMessages so the
    # serializer in the route can round-trip them without warnings.
    pa_msgs = [
        ModelRequest(parts=[UserPromptPart(content="What was Zomato revenue?")]),
        ModelResponse(parts=[TextPart(content="Revenue was 12,114 crore [zomato_fy24.pdf, p. 12].")]),
    ]
    fake_run = AsyncMock(return_value=_make_response(message_history=pa_msgs))

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_session_manager] = lambda: sm
    app.dependency_overrides[get_vector_store] = lambda: vs
    app.dependency_overrides[get_web_search_client] = lambda: web
    app.dependency_overrides[run_agent_dep] = lambda: fake_run
    app.state._test_sm = sm
    app.state._fake_run = fake_run
    return app


@pytest.mark.asyncio
async def test_send_message(app):
    sm = app.state._test_sm
    s = sm.create_session()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(f"/sessions/{s.session_id}/messages", json={"content": "What was Zomato revenue?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "Revenue was 12,114 crore" in body["answer_text"]
    assert body["citations"][0]["doc_name"] == "zomato_fy24.pdf"
    assert body["tool_calls"][0]["tool_name"] == "retrieve_from_docs"
    assert body["usage"]["total_tokens"] == 120
    # Session has 2 new messages (user + assistant)
    fetched = sm.get_session(s.session_id)
    assert len(fetched.messages) == 2
    assert fetched.messages[0].role == "user"
    assert fetched.messages[1].role == "assistant"


@pytest.mark.asyncio
async def test_send_message_unknown_session_404(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/sessions/no-such/messages", json={"content": "x"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_multi_turn_replays_pa_history(app):
    """Second turn must receive the deserialized pa_messages_json from the first turn
    as message_history. This is the multi-turn memory fix."""
    sm = app.state._test_sm
    fake_run = app.state._fake_run
    s = sm.create_session()

    # First turn — empty pa_messages_json initially.
    assert sm.get_session(s.session_id).pa_messages_json == ""

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r1 = await c.post(f"/sessions/{s.session_id}/messages", json={"content": "Turn 1"})
    assert r1.status_code == 200

    # After turn 1, pa_messages_json should be populated with the serialized graph.
    after_turn_1 = sm.get_session(s.session_id).pa_messages_json
    assert after_turn_1 != ""
    assert "Revenue was 12,114 crore" in after_turn_1

    # First call: runner was invoked with message_history=[] (empty replay).
    first_call_kwargs = fake_run.call_args_list[0].kwargs
    assert first_call_kwargs["message_history"] == []

    # Second turn — runner should now receive the deserialized prior graph.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r2 = await c.post(f"/sessions/{s.session_id}/messages", json={"content": "Turn 2"})
    assert r2.status_code == 200

    second_call_kwargs = fake_run.call_args_list[1].kwargs
    replayed = second_call_kwargs["message_history"]
    # Should be a non-empty list of ModelMessage objects (ModelRequest / ModelResponse).
    assert isinstance(replayed, list)
    assert len(replayed) == 2
    assert replayed[0].__class__.__name__ == "ModelRequest"
    assert replayed[1].__class__.__name__ == "ModelResponse"
