"""Structured output type for the agent.

Pydantic validators here are RUNTIME GATES. Pydantic AI catches ValidationError
from output_type and auto-retries (configured via Agent(retries=2)).
"""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    kind: Literal["doc"] = "doc"
    doc_name: str
    page_number: int


class WebCitation(BaseModel):
    kind: Literal["web"] = "web"
    url: str
    title: str


CitationT = Annotated[Union[Citation, WebCitation], Field(discriminator="kind")]


class Answer(BaseModel):
    text: str
    citations: list[CitationT] = Field(default_factory=list)
    requires_citation: bool = Field(
        description="True if the answer makes any factual claim from the corpus or web."
    )
    retrieval_iterations: int = Field(
        default=0,
        description="Number of retrieve_from_docs calls made; instrumentation for FM-1.",
    )

    @model_validator(mode="after")
    def _enforce_citation_requirement(self) -> "Answer":
        if self.requires_citation and not self.citations:
            raise ValueError(
                "Answer makes a factual claim (requires_citation=True) but has no citations. "
                "Either provide at least one citation, or set requires_citation=False if the "
                "answer is purely conversational."
            )
        return self
