from eval.metrics.citation_check import parse_citations, check_citations


def test_parse_citations_extracts_doc_page():
    text = "Revenue was 12,114 crore [zomato_fy24.pdf, p. 12]. Margin was X% [zomato_fy24.pdf, p. 14]."
    cites = parse_citations(text)
    assert ("zomato_fy24.pdf", 12) in cites
    assert ("zomato_fy24.pdf", 14) in cites
    assert len(cites) == 2


def test_check_citations_precision_recall():
    answer = "X is 5 [a.pdf, p. 1]. Y is 3 [b.pdf, p. 2]."
    chunks_by_page = {
        ("a.pdf", 1): "X is 5 in the report.",
        ("b.pdf", 2): "Z is mentioned here.",  # citation present but content doesn't support
    }
    result = check_citations(answer, chunks_by_page, ground_truth="X is 5; Y is 3")
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0


def test_check_citations_no_citations():
    result = check_citations("answer with no markers", {}, ground_truth="anything")
    assert result.precision == 0.0
    assert result.recall == 0.0
