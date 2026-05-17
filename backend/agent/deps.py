"""Dependencies injected into every tool via Pydantic AI's RunContext."""
from dataclasses import dataclass
from typing import Protocol

from backend.retrieval.retriever import RetrievalResult


class VectorStoreProto(Protocol):
    def query(
        self,
        session_id: str,
        query_text: str,
        k: int = 5,
        doc_filter: list[str] | None = None,
    ) -> list: ...
    def list_documents(self, session_id: str) -> list[dict]: ...


class WebSearchProto(Protocol):
    def search(self, query: str, max_results: int = 5) -> dict: ...


@dataclass
class AgentDeps:
    session_id: str
    vector_store: VectorStoreProto
    web_search_client: WebSearchProto
    retrieval_threshold: float = 0.3
    retrieval_k: int = 5
