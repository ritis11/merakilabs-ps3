"""Pydantic AI Agent definition + run wrapper.

The wrapper is what the API and eval call. It:
- Enforces UsageLimits(request_limit=N) - the hard runtime cap on the agent loop.
- Catches UsageLimitExceeded and returns a graceful AgentResponse.
- Returns structured AgentResponse with tool calls extracted for the API/eval.
"""
from dataclasses import dataclass
from typing import Any

import structlog
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.usage import UsageLimits

from backend.agent.deps import AgentDeps
from backend.agent.output import Answer
from backend.agent.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from backend.agent.tools import (
    calculate,
    google_search,
    list_available_documents,
    retrieve_from_docs,
)
from backend.config import get_settings

log = structlog.get_logger(__name__)


def build_agent() -> Agent[AgentDeps, Answer]:
    settings = get_settings()
    model = GoogleModel(settings.agent_model)
    agent = Agent(
        model=model,
        deps_type=AgentDeps,
        output_type=Answer,
        system_prompt=SYSTEM_PROMPT,
        tools=[retrieve_from_docs, list_available_documents, calculate, google_search],
        retries=2,
    )
    return agent


# Module-level singleton; rebuilt on import. Tests monkeypatch this.
_AGENT: Agent[AgentDeps, Answer] | None = None


def _agent() -> Agent[AgentDeps, Answer]:
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT


@dataclass
class AgentResponse:
    answer: Answer
    message_history: list[Any]
    tool_calls: list[dict]
    usage: dict
    prompt_version: str = PROMPT_VERSION


def _extract_tool_calls(messages: list[Any]) -> list[dict]:
    """Extract tool calls from Pydantic AI message graph for API/eval."""
    out: list[dict] = []
    for m in messages:
        parts = getattr(m, "parts", []) or []
        for p in parts:
            kind = getattr(p, "part_kind", None)
            if kind == "tool-call":
                out.append(
                    {
                        "tool_name": getattr(p, "tool_name", None),
                        "args": getattr(p, "args", None),
                        "tool_call_id": getattr(p, "tool_call_id", None),
                    }
                )
            elif kind == "tool-return":
                # Attach return value to the most recent tool-call with matching id
                tool_call_id = getattr(p, "tool_call_id", None)
                content = getattr(p, "content", None)
                for entry in reversed(out):
                    if entry.get("tool_call_id") == tool_call_id and "result" not in entry:
                        entry["result"] = content
                        break
    return out


async def run_agent(
    message_history: list[Any],
    user_message: str,
    deps: AgentDeps,
) -> AgentResponse:
    """Run the agent for one user turn. Hard caps via UsageLimits."""
    settings = get_settings()
    usage_limits = UsageLimits(request_limit=settings.agent_request_limit)
    try:
        result = await _agent().run(
            user_message,
            message_history=message_history,
            deps=deps,
            usage_limits=usage_limits,
        )
    except UsageLimitExceeded:
        log.warning("usage_limit_exceeded", session_id=deps.session_id)
        graceful = Answer(
            text=(
                "I couldn't reach an answer within the allowed iteration budget. "
                "Try rephrasing the question more specifically."
            ),
            citations=[],
            requires_citation=False,
            retrieval_iterations=0,
        )
        return AgentResponse(
            answer=graceful,
            message_history=message_history,
            tool_calls=[],
            usage={
                "request_tokens": 0,
                "response_tokens": 0,
                "total_tokens": 0,
                "limit_exceeded": True,
            },
        )

    msgs = result.all_messages()
    # `all_messages()` is the full graph (prior turns + this turn). Tool traces
    # for the API/UI should reflect only this user turn, not every call ever.
    prior_len = len(message_history)
    if prior_len > len(msgs):
        log.warning(
            "all_messages_shorter_than_history",
            prior_len=prior_len,
            msgs_len=len(msgs),
        )
        turn_msgs = msgs
    else:
        turn_msgs = msgs[prior_len:]
    usage_obj = result.usage()
    # Pydantic AI 1.96.x: input_tokens/output_tokens. Older versions used request/response.
    # Read both shapes so token tracking survives a future SDK rename.
    input_tokens = getattr(usage_obj, "input_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage_obj, "request_tokens", 0) or 0
    output_tokens = getattr(usage_obj, "output_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage_obj, "response_tokens", 0) or 0
    total_tokens = getattr(usage_obj, "total_tokens", None)
    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return AgentResponse(
        answer=result.output,
        message_history=msgs,
        tool_calls=_extract_tool_calls(turn_msgs),
        usage={
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "total_tokens": total_tokens or 0,
        },
    )
