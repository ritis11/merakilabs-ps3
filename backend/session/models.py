"""Session and Message dataclasses. JSON-serializable for shutdown dump."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    tool_calls: list[dict] | None = None  # populated for assistant messages

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tool_calls": self.tool_calls,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tool_calls=data.get("tool_calls"),
        )


@dataclass
class Session:
    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    messages: list[Message] = field(default_factory=list)
    uploaded_docs: list[str] = field(default_factory=list)
    # Serialized Pydantic AI message graph for multi-turn conversations.
    # JSON bytes produced by ModelMessagesTypeAdapter.dump_json(); stored as a
    # str so JSON dump/load works. Empty string = no prior turns.
    pa_messages_json: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "messages": [m.to_dict() for m in self.messages],
            "uploaded_docs": list(self.uploaded_docs),
            "pa_messages_json": self.pa_messages_json,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            uploaded_docs=list(data.get("uploaded_docs", [])),
            pa_messages_json=data.get("pa_messages_json", ""),
        )
