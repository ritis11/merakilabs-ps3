import io
from unittest.mock import MagicMock
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from backend.routes.documents import router, get_session_manager, get_vector_store
from backend.session.store import SessionManager


def _tiny_pdf_bytes() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(72, 720, "Some content for testing on page one.")
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def app(tmp_path):
    sm = SessionManager(dump_path=str(tmp_path / "s.json"), history_cap=10)
    vs = MagicMock()
    vs.add_chunks.return_value = None
    vs.list_documents.return_value = []
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_session_manager] = lambda: sm
    app.dependency_overrides[get_vector_store] = lambda: vs
    app.state._test_sm = sm
    app.state._test_vs = vs
    return app


@pytest.mark.asyncio
async def test_upload_document(app):
    sm = app.state._test_sm
    s = sm.create_session()
    files = {"file": ("test.pdf", _tiny_pdf_bytes(), "application/pdf")}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(f"/sessions/{s.session_id}/documents", files=files)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["doc_name"] == "test.pdf"
    assert body["chunk_count"] >= 1
    # vector store received the chunks
    app.state._test_vs.add_chunks.assert_called_once()
    # session tracks the doc
    assert "test.pdf" in sm.get_session(s.session_id).uploaded_docs


@pytest.mark.asyncio
async def test_upload_unknown_session_404(app):
    files = {"file": ("test.pdf", _tiny_pdf_bytes(), "application/pdf")}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/sessions/no-such/documents", files=files)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_list_documents(app):
    sm = app.state._test_sm
    vs = app.state._test_vs
    s = sm.create_session()
    vs.list_documents.return_value = [
        {"doc_id": "d1", "doc_name": "x.pdf", "chunk_count": 5}
    ]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get(f"/sessions/{s.session_id}/documents")
    assert r.status_code == 200
    assert r.json() == {"documents": [{"doc_id": "d1", "doc_name": "x.pdf", "chunk_count": 5}]}
