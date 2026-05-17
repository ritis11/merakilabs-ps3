"""ChromaDB wrapper. One collection per session; doc_filter via metadata where-clause."""
import threading
from typing import Protocol, Sequence

import chromadb
from chromadb.errors import NotFoundError

from backend.ingestion.models import Chunk


class EmbedderProto(Protocol):
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class ChromaStore:
    def __init__(self, persist_dir: str, embedder: EmbedderProto):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedder = embedder
        # PersistentClient is not safe for concurrent writes from multiple threads;
        # FastAPI runs sync routes in a thread pool, so serialise mutating calls.
        self._write_lock = threading.Lock()

    def _coll_name(self, session_id: str) -> str:
        return f"session_{session_id}"

    def _get_collection(self, session_id: str, create: bool = True):
        name = self._coll_name(session_id)
        if create:
            return self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        try:
            return self._client.get_collection(name=name)
        except (ValueError, NotFoundError):
            return None

    def add_chunks(self, session_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self._write_lock:
            coll = self._get_collection(session_id, create=True)
            embeddings = self._embedder.embed_documents([c.text for c in chunks])
            coll.add(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=[c.to_metadata() for c in chunks],
            )

    def query(
        self,
        session_id: str,
        query_text: str,
        k: int = 5,
        doc_filter: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        coll = self._get_collection(session_id, create=False)
        if coll is None:
            return []
        q_vec = self._embedder.embed_query(query_text)
        where: dict | None = None
        if doc_filter:
            where = {"doc_name": {"$in": list(doc_filter)}}

        res = coll.query(query_embeddings=[q_vec], n_results=k, where=where)
        out: list[tuple[Chunk, float]] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        for cid, doc, md, dist in zip(ids, docs, metadatas, distances):
            chunk = Chunk(
                chunk_id=cid,
                doc_id=md["doc_id"],
                doc_name=md["doc_name"],
                page_number=int(md["page_number"]),
                text=doc,
                char_start=0,  # not stored in metadata; debug-only field
                char_end=len(doc),
            )
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            out.append((chunk, similarity))
        return out

    def list_documents(self, session_id: str) -> list[dict]:
        coll = self._get_collection(session_id, create=False)
        if coll is None:
            return []
        # Pull all metadatas (small for v1 corpora) and aggregate by doc_id
        all_md = coll.get(include=["metadatas"]).get("metadatas", [])
        by_doc: dict[str, dict] = {}
        for md in all_md:
            doc_id = md["doc_id"]
            entry = by_doc.setdefault(doc_id, {
                "doc_id": doc_id,
                "doc_name": md["doc_name"],
                "chunk_count": 0,
            })
            entry["chunk_count"] += 1
        return list(by_doc.values())

    def delete_session(self, session_id: str) -> None:
        with self._write_lock:
            try:
                self._client.delete_collection(name=self._coll_name(session_id))
            except Exception:
                pass  # already gone — idempotent
