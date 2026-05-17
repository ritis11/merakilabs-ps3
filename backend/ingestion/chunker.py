"""Recursive character splitter targeting ~800 tokens with overlap.

Each chunk inherits the page number of the page it starts on.
"""
import uuid
from typing import Iterable

import tiktoken

from backend.ingestion.models import Chunk


_ENCODING = tiktoken.get_encoding("cl100k_base")
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    doc_id: str,
    doc_name: str,
    target_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """Split each page's text into chunks; assign 1-indexed page numbers.

    Pages with empty/whitespace-only text are skipped.
    """
    out: list[Chunk] = []
    for page_number, text in pages:
        if not text or not text.strip():
            continue
        page_chunks = _split_text(text, target_tokens, overlap_tokens)
        for sub_text, char_start, char_end in page_chunks:
            out.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                doc_name=doc_name,
                page_number=page_number,
                text=sub_text,
                char_start=char_start,
                char_end=char_end,
            ))
    return out


def _split_text(text: str, target_tokens: int, overlap_tokens: int) -> list[tuple[str, int, int]]:
    """Return list of (text, char_start, char_end) for chunks."""
    tokens = _ENCODING.encode(text)
    if len(tokens) <= target_tokens:
        return [(text, 0, len(text))]

    chunks: list[tuple[str, int, int]] = []
    step = max(1, target_tokens - overlap_tokens)
    cursor = 0
    while cursor < len(tokens):
        window = tokens[cursor : cursor + target_tokens]
        sub_text = _ENCODING.decode(window)
        # snap chunk boundaries to nearest preferred separator within sub_text
        sub_text = _snap_boundary(sub_text)
        char_start = text.find(sub_text[: min(40, len(sub_text))])
        if char_start < 0:
            char_start = 0
        chunks.append((sub_text, char_start, char_start + len(sub_text)))
        cursor += step
    return chunks


def _snap_boundary(text: str) -> str:
    """Trim trailing fragment to the nearest preferred separator."""
    for sep in _SEPARATORS[:-1]:  # don't snap on empty string
        idx = text.rfind(sep)
        # only snap if it's in the back half (preserve at least 50% of the chunk)
        if idx > len(text) // 2:
            return text[: idx + len(sep)]
    return text
