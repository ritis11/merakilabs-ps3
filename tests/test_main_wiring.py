import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_main_app_has_state_after_lifespan_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("TAVILY_API_KEY", "dummy")
    # Keep test artifacts (chroma_db, sessions.json) inside tmp_path
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma_db"))
    monkeypatch.setenv("SESSION_DUMP_PATH", str(tmp_path / "sessions.json"))
    # Force re-import so settings pick up env
    import importlib
    import backend.main as bm
    importlib.reload(bm)

    # NOTE: httpx 0.28's ASGITransport does not drive lifespan events on its own,
    # so we use asgi-lifespan's LifespanManager to trigger startup before requests.
    transport = ASGITransport(app=bm.app)
    async with LifespanManager(bm.app):
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            # State populated by lifespan startup
            r = await c.post("/sessions")
            assert r.status_code == 201

            assert hasattr(bm.app.state, "session_manager")
            assert hasattr(bm.app.state, "vector_store")
            assert hasattr(bm.app.state, "web_search_client")
