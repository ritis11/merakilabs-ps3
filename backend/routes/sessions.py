"""Session lifecycle endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from backend.schemas import (
    CreateSessionResponse,
    HistoryMessage,
    HistoryResponse,
)
from backend.session.store import SessionManager

router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def get_vector_store(request: Request):
    return request.app.state.vector_store


@router.post("/sessions", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(sm: SessionManager = Depends(get_session_manager)):
    s = sm.create_session()
    return CreateSessionResponse(session_id=s.session_id, created_at=s.created_at)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
    vs=Depends(get_vector_store),
):
    sm.delete_session(session_id)
    vs.delete_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str, sm: SessionManager = Depends(get_session_manager)):
    s = sm.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    return HistoryResponse(messages=[
        HistoryMessage(role=m.role, content=m.content, timestamp=m.timestamp, tool_calls=m.tool_calls)
        for m in s.messages
    ])
