"""Chat message endpoint. Wires session history → agent → response shape."""
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic_ai.messages import ModelMessagesTypeAdapter

from backend.agent.agent import AgentResponse, run_agent
from backend.agent.deps import AgentDeps
from backend.schemas import (
    SendMessageRequest,
    SendMessageResponse,
    ToolCallTrace,
)
from backend.session.models import Message
from backend.session.store import SessionManager

router = APIRouter()
log = structlog.get_logger(__name__)


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def get_vector_store(request: Request):
    return request.app.state.vector_store


def get_web_search_client(request: Request):
    return request.app.state.web_search_client


# Indirection so tests can override the agent run function
def run_agent_dep() -> Callable[..., Awaitable[AgentResponse]]:
    return run_agent


def _load_pa_history(pa_messages_json: str) -> list[Any]:
    """Deserialize the stored Pydantic AI message graph for replay.

    Returns [] on first turn (empty stored JSON) or on parse failure (logged).
    """
    if not pa_messages_json:
        return []
    try:
        return ModelMessagesTypeAdapter.validate_json(pa_messages_json)
    except Exception as e:
        log.warning("pa_history_deserialize_failed", error=str(e))
        return []


def _serialize_pa_history(messages: list[Any]) -> str:
    """Serialize the full Pydantic AI message graph (this turn + prior) for
    persistence. Returns '' on failure (logged) so we don't clobber state."""
    try:
        return ModelMessagesTypeAdapter.dump_json(messages).decode("utf-8")
    except Exception as e:
        log.warning("pa_history_serialize_failed", error=str(e))
        return ""


@router.post("/sessions/{session_id}/messages", response_model=SendMessageResponse)
async def send_message(
    session_id: str,
    body: SendMessageRequest,
    sm: SessionManager = Depends(get_session_manager),
    vs=Depends(get_vector_store),
    web=Depends(get_web_search_client),
    runner: Callable[..., Awaitable[AgentResponse]] = Depends(run_agent_dep),
):
    s = sm.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")

    deps = AgentDeps(
        session_id=session_id,
        vector_store=vs,
        web_search_client=web,
    )

    # Replay prior Pydantic AI message graph (deserialized from session storage)
    # so the agent sees full tool-call context from previous turns, not just text.
    pa_history = _load_pa_history(s.pa_messages_json)

    response = await runner(message_history=pa_history, user_message=body.content, deps=deps)

    # Persist updated message graph for the next turn. response.message_history
    # is the full Pydantic AI graph (prior + this turn).
    new_pa_json = _serialize_pa_history(response.message_history)
    if new_pa_json:
        sm.set_pa_messages_json(session_id, new_pa_json)

    # Plain text history for the /history endpoint and Streamlit UI.
    sm.append_message(
        session_id,
        Message(role="user", content=body.content, timestamp=datetime.now(timezone.utc)),
    )
    sm.append_message(
        session_id,
        Message(
            role="assistant",
            content=response.answer.text,
            timestamp=datetime.now(timezone.utc),
            tool_calls=response.tool_calls,
        ),
    )

    return SendMessageResponse(
        answer_text=response.answer.text,
        citations=response.answer.citations,
        requires_citation=response.answer.requires_citation,
        retrieval_iterations=response.answer.retrieval_iterations,
        tool_calls=[ToolCallTrace(**tc) for tc in response.tool_calls],
        usage=response.usage,
        prompt_version=response.prompt_version,
    )
