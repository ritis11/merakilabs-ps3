"""In-memory session manager with shutdown JSON dump.

History trim: FIFO oldest user/assistant messages dropped when len > history_cap.
The system prompt isn't stored here — it lives in backend/agent/prompts.py.
"""
import json
import uuid
from pathlib import Path

import structlog

from backend.session.models import Message, Session

log = structlog.get_logger(__name__)


class SessionManager:
    def __init__(self, dump_path: str, history_cap: int = 10):
        self._sessions: dict[str, Session] = {}
        self._dump_path = Path(dump_path)
        self._history_cap = history_cap

    def create_session(self) -> Session:
        sid = str(uuid.uuid4())
        s = Session(session_id=sid)
        self._sessions[sid] = s
        return s

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def append_message(self, session_id: str, message: Message) -> None:
        s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(f"unknown session: {session_id}")
        s.messages.append(message)
        if len(s.messages) > self._history_cap:
            s.messages = s.messages[-self._history_cap :]

    def add_uploaded_doc(self, session_id: str, doc_name: str) -> None:
        s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(f"unknown session: {session_id}")
        if doc_name not in s.uploaded_docs:
            s.uploaded_docs.append(doc_name)

    def set_pa_messages_json(self, session_id: str, payload: str) -> None:
        """Persist the serialized Pydantic AI message graph for multi-turn replay."""
        s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(f"unknown session: {session_id}")
        s.pa_messages_json = payload

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def dump(self) -> None:
        try:
            payload = {sid: s.to_dict() for sid, s in self._sessions.items()}
            self._dump_path.write_text(json.dumps(payload, indent=2))
            log.info("session_dump_written", count=len(payload), path=str(self._dump_path))
        except Exception as e:
            log.error("session_dump_failed", error=str(e))

    def load(self) -> None:
        if not self._dump_path.exists():
            log.info("session_dump_missing", path=str(self._dump_path))
            return
        try:
            payload = json.loads(self._dump_path.read_text())
            self._sessions = {sid: Session.from_dict(data) for sid, data in payload.items()}
            log.info("session_dump_loaded", count=len(self._sessions))
        except Exception as e:
            log.error("session_dump_load_failed", error=str(e))
            self._sessions = {}
