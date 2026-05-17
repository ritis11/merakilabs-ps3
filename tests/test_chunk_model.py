from backend.ingestion.models import Chunk


def test_chunk_has_required_fields():
    c = Chunk(
        chunk_id="c1",
        doc_id="d1",
        doc_name="zomato_fy24.pdf",
        page_number=12,
        text="hello world",
        char_start=0,
        char_end=11,
    )
    assert c.chunk_id == "c1"
    assert c.doc_id == "d1"
    assert c.doc_name == "zomato_fy24.pdf"
    assert c.page_number == 12
    assert c.text == "hello world"
    assert c.char_start == 0
    assert c.char_end == 11


def test_chunk_to_metadata_excludes_text_and_id():
    """ChromaDB metadata is per-vector and shouldn't duplicate the document text."""
    c = Chunk(
        chunk_id="c1", doc_id="d1", doc_name="x.pdf",
        page_number=1, text="t", char_start=0, char_end=1,
    )
    md = c.to_metadata()
    assert md == {"doc_id": "d1", "doc_name": "x.pdf", "page_number": 1, "chunk_id": "c1"}
