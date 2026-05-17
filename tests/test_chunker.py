from backend.ingestion.chunker import chunk_pages


def test_chunker_emits_chunks_with_page_numbers():
    pages = [
        (1, "alpha. " * 200),
        (2, "beta. " * 200),
    ]
    chunks = chunk_pages(
        pages,
        doc_id="d1",
        doc_name="test.pdf",
        target_tokens=100,
        overlap_tokens=20,
    )
    assert len(chunks) > 0
    # Chunks must inherit page numbers from their source page.
    page_1_chunks = [c for c in chunks if c.page_number == 1]
    page_2_chunks = [c for c in chunks if c.page_number == 2]
    assert len(page_1_chunks) > 0
    assert len(page_2_chunks) > 0
    # All chunk_ids unique
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    # Doc identity propagated
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(c.doc_name == "test.pdf" for c in chunks)
    # Chunks should respect approximate token target (loose check: not gigantic)
    for c in chunks:
        assert len(c.text) < 2000  # rough byte cap


def test_chunker_skips_empty_pages():
    pages = [(1, ""), (2, "real content here. " * 50), (3, "  ")]
    chunks = chunk_pages(pages, doc_id="d1", doc_name="x.pdf", target_tokens=100, overlap_tokens=20)
    assert all(c.page_number == 2 for c in chunks)


def test_chunker_short_page_emits_one_chunk():
    pages = [(1, "small page.")]
    chunks = chunk_pages(pages, doc_id="d1", doc_name="x.pdf", target_tokens=100, overlap_tokens=20)
    assert len(chunks) == 1
    assert chunks[0].text == "small page."
