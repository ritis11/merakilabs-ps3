from backend.ingestion.parser import parse_pdf


def test_parser_returns_page_text_tuples(synthetic_pdf):
    pages = parse_pdf(synthetic_pdf)
    assert len(pages) == 3
    assert pages[0][0] == 1  # 1-indexed
    assert pages[1][0] == 2
    assert pages[2][0] == 3
    assert "12,114 crore" in pages[0][1]
    assert "Blinkit" in pages[1][1]
    assert "governance" in pages[2][1]


def test_parser_handles_missing_file(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        parse_pdf(tmp_path / "does_not_exist.pdf")
