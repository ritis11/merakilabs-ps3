from unittest.mock import MagicMock
import pytest
from pydantic_ai import RunContext
from backend.agent.deps import AgentDeps
from backend.agent.tools import calculate


def _ctx():
    deps = AgentDeps(
        session_id="s1",
        vector_store=MagicMock(),
        web_search_client=MagicMock(),
    )
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    return ctx


@pytest.mark.asyncio
async def test_calculate_basic():
    r = await calculate(_ctx(), "12114 * 10000000")
    assert r.error is None
    assert r.result == 121140000000.0


@pytest.mark.asyncio
async def test_calculate_handles_division_by_zero():
    r = await calculate(_ctx(), "1 / 0")
    assert r.error is not None
    assert r.result is None


@pytest.mark.asyncio
async def test_calculate_blocks_unsafe_names():
    r = await calculate(_ctx(), "__import__('os').system('ls')")
    assert r.error is not None
    assert r.result is None


@pytest.mark.asyncio
async def test_calculate_supports_sqrt_and_log():
    r = await calculate(_ctx(), "sqrt(16)")
    assert r.error is None
    assert r.result == 4.0
