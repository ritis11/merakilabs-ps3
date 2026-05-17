"""Chunk schema. Single source of truth between ingestion and retrieval."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str       # uuid
    doc_id: str         # uuid per uploaded document
    doc_name: str       # original filename, e.g. "zomato_fy24.pdf"
    page_number: int    # 1-indexed page in source PDF (start page if chunk spans pages)
    text: str           # chunk content
    char_start: int     # offset within page text (debug)
    char_end: int

    def to_metadata(self) -> dict[str, str | int]:
        """Per-vector metadata for ChromaDB. Excludes text (stored as document)."""
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
        }
