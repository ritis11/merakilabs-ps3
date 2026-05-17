from unittest.mock import MagicMock
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from backend.routes.sessions import router as sessions_router, get_session_manager, get_vector_store
from backend.session.models import Session
from backend.session.store import SessionManager


@pytest.fixture
def app(tmp_path):
    sm = SessionManager(dump_path=str(tmp_path / "s.json"), history_cap=10)
    vs = MagicMock()
    app = FastAPI()
    app.include_router(sessions_router)
    app.dependency_overrides[get_session_manager] = lambda: sm
    app.dependency_overrides[get_vector_store] = lambda: vs
    app.state._test_sm = sm
    app.state._test_vs = vs
    return app


@pytest.mark.asyncio
async def test_create_session(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/sessions")
    assert r.status_code == 201
    body = r.json()
    assert "session_id" in body
    assert "created_at" in body


@pytest.mark.asyncio
async def test_get_history(app):
    sm = app.state._test_sm
    s = sm.create_session()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get(f"/sessions/{s.session_id}/history")
    assert r.status_code == 200
    assert r.json() == {"messages": []}


@pytest.mark.asyncio
async def test_get_history_unknown_session_404(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/sessions/does-not-exist/history")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_session_drops_chroma_collection(app):
    sm = app.state._test_sm
    vs = app.state._test_vs
    s = sm.create_session()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.delete(f"/sessions/{s.session_id}")
    assert r.status_code == 204
    assert sm.get_session(s.session_id) is None
    vs.delete_session.assert_called_once_with(s.session_id)
