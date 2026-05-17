"""HTTP request/response shapes. Distinct from Pydantic AI Answer to allow API
evolution without churning the agent contract."""
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from backend.agent.output import Citation, WebCitation


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: datetime


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]


class UploadDocumentResponse(BaseModel):
    doc_id: str
    doc_name: str
    chunk_count: int


class HistoryMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime
    tool_calls: list[dict] | None = None


class HistoryResponse(BaseModel):
    messages: list[HistoryMessage]


class SendMessageRequest(BaseModel):
    content: str


class ToolCallTrace(BaseModel):
    tool_name: str
    args: Any | None = None
    result: Any | None = None


class SendMessageResponse(BaseModel):
    answer_text: str
    citations: list[Citation | WebCitation]
    requires_citation: bool
    retrieval_iterations: int
    tool_calls: list[ToolCallTrace]
    usage: dict
    prompt_version: str
