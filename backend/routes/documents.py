"""Document upload + listing endpoints.

Ingestion is synchronous CPU + I/O heavy work. Exposed as plain ``def`` handlers
so Starlette runs them in a thread pool and the asyncio loop stays responsive
for other requests (avoids stalls when a second upload follows a long ingest).
"""
import shutil
import tempfile
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from backend.ingestion.chunker import chunk_pages
from backend.ingestion.parser import parse_pdf
from backend.schemas import DocumentInfo, DocumentsResponse, UploadDocumentResponse
from backend.session.store import SessionManager

router = APIRouter()
log = structlog.get_logger(__name__)


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def get_vector_store(request: Request):
    return request.app.state.vector_store


@router.post(
    "/sessions/{session_id}/documents",
    response_model=UploadDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    session_id: str,
    file: UploadFile = File(...),
    sm: SessionManager = Depends(get_session_manager),
    vs=Depends(get_vector_store),
):
    s = sm.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")

    doc_id = str(uuid.uuid4())
    doc_name = file.filename or f"{doc_id}.pdf"

    # Persist upload to a temp file so the parser can use a path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        pages = parse_pdf(tmp_path)
        chunks = chunk_pages(pages, doc_id=doc_id, doc_name=doc_name)
        vs.add_chunks(session_id, chunks)
        sm.add_uploaded_doc(session_id, doc_name)
        log.info("document_ingested", session_id=session_id, doc_name=doc_name, chunk_count=len(chunks))
        return UploadDocumentResponse(doc_id=doc_id, doc_name=doc_name, chunk_count=len(chunks))
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/sessions/{session_id}/documents", response_model=DocumentsResponse)
def list_documents(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
    vs=Depends(get_vector_store),
):
    if sm.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="session not found")
    docs = vs.list_documents(session_id)
    return DocumentsResponse(documents=[DocumentInfo(**d) for d in docs])
