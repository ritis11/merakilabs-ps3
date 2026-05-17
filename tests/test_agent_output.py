import pytest
from pydantic import ValidationError
from backend.agent.output import Answer, Citation, WebCitation


def test_answer_with_citations_validates():
    a = Answer(
        text="Revenue was 12,114 crore [zomato_fy24.pdf, p. 12].",
        citations=[Citation(doc_name="zomato_fy24.pdf", page_number=12)],
        requires_citation=True,
        retrieval_iterations=1,
    )
    assert len(a.citations) == 1


def test_answer_requiring_citation_with_none_fails_validation():
    with pytest.raises(ValidationError) as exc:
        Answer(
            text="Revenue was 12,114 crore.",
            citations=[],
            requires_citation=True,
            retrieval_iterations=1,
        )
    assert "citation" in str(exc.value).lower()


def test_trivial_reply_without_citation_is_ok():
    a = Answer(
        text="Hello! What would you like to know?",
        citations=[],
        requires_citation=False,
        retrieval_iterations=0,
    )
    assert a.text.startswith("Hello")


def test_web_citation_supported():
    a = Answer(
        text="Tesla shares dropped 5% [bloomberg.com].",
        citations=[WebCitation(url="https://bloomberg.com/...", title="Tesla...")],
        requires_citation=True,
        retrieval_iterations=1,
    )
    assert a.citations[0].kind == "web"
