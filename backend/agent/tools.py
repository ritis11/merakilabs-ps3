"""Agent tools. All tools take RunContext[AgentDeps] as first arg.

All tool errors are returned as structured fields (never raised), so the agent
sees them as data and can recover, not as exceptions that crash the run.
"""
from pydantic import BaseModel
from pydantic_ai import RunContext

from backend.agent.deps import AgentDeps
from backend.retrieval.retriever import RetrievalResult, retrieve


async def retrieve_from_docs(
    ctx: RunContext[AgentDeps],
    query: str,
    doc_filter: list[str] | None = None,
    k: int | None = None,
) -> RetrievalResult:
    """Search the user's uploaded documents.

    Args:
        query: search query.
        doc_filter: optional list of doc_names to restrict search to (e.g.
            ["nykaa_fy24.pdf"]). Pass when the question names a specific company.
        k: optional override for top-k (default from deps).
    """
    deps = ctx.deps
    return retrieve(
        deps.vector_store,
        deps.session_id,
        query,
        k=k or deps.retrieval_k,
        threshold=deps.retrieval_threshold,
        doc_filter=doc_filter,
    )


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    chunk_count: int


async def list_available_documents(ctx: RunContext[AgentDeps]) -> list[DocumentInfo]:
    """List documents uploaded in this session. Use for meta-questions like
    "what reports do you have?" — do not fabricate document names."""
    raw = ctx.deps.vector_store.list_documents(ctx.deps.session_id)
    return [DocumentInfo(**d) for d in raw]


from asteval import Interpreter


class CalculationResult(BaseModel):
    result: float | None
    error: str | None


async def calculate(ctx: RunContext[AgentDeps], expression: str) -> CalculationResult:
    """Evaluate a numeric expression in a restricted namespace.

    Supported: + - * / ** %, sqrt(), log(), abs(), parentheses.
    Numbers only — convert all values to absolute integers BEFORE calling. For
    "INR 12,114 crore", pass "12114 * 10000000".

    All errors (syntax, name lookup, division by zero) are returned as `error`
    strings, never raised — the agent sees them as data and can recover.
    """
    interp = Interpreter(minimal=True)
    try:
        value = interp(expression)
        # asteval stores errors in interp.error
        if interp.error:
            err = "; ".join(str(e.get_error()[1]) for e in interp.error)
            return CalculationResult(result=None, error=err)
        if value is None:
            return CalculationResult(result=None, error="Expression evaluated to None")
        return CalculationResult(result=float(value), error=None)
    except Exception as e:
        return CalculationResult(result=None, error=str(e))


from typing import Literal


class Source(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchResult(BaseModel):
    sources: list[Source]
    used_because: Literal["below_threshold", "user_request"]
    error: str | None = None


async def google_search(
    ctx: RunContext[AgentDeps],
    query: str,
    used_because: Literal["below_threshold", "user_request"],
) -> WebSearchResult:
    """Search the public web via Tavily.

    REQUIRED: only call when (a) retrieve_from_docs returned
    all_below_threshold=True for this question, OR (b) the user explicitly asked
    to look something up online. The `used_because` arg justifies the call and
    is checked post-hoc in eval.
    """
    try:
        raw = ctx.deps.web_search_client.search(query=query, max_results=5)
    except Exception as e:
        return WebSearchResult(sources=[], used_because=used_because, error=str(e))

    sources = [
        Source(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content", ""))
        for r in raw.get("results", [])
    ]
    return WebSearchResult(sources=sources, used_because=used_because)
