"""Eval runner. Orchestrates per-Q runs, applies all applicable metrics,
writes JSONL per result, prints aggregate report with SLO red/green.

Vector store (eval only): all questions share one Chroma collection
``session_eval_shared_corpus``. PDFs are parsed + embedded only the first time
each ``doc_name`` appears; later rows reuse the same vectors. Each question still
uses ``AgentDeps(session_id="eval_<qid>", ...)`` with ``message_history=[]`` so
the agent run is a fresh slate; retrieval is redirected to the shared corpus via
``_EvalCorpusVectorStore``.

Usage:
  uv run python -m eval.runner \
      --dataset eval/dataset/questions.jsonl \
      --adversarial eval/dataset/adversarial.jsonl \
      --output eval/results/<timestamp>.jsonl
"""
from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()
import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rich import print as rich_print
from rich.markup import escape
from tavily import TavilyClient

from backend.agent.agent import run_agent
from backend.agent.deps import AgentDeps
from backend.config import get_settings
from backend.ingestion.chunker import chunk_pages
from backend.ingestion.parser import parse_pdf
from backend.retrieval.embedder import GeminiEmbedder
from backend.retrieval.store import ChromaStore
from eval.metrics.citation_check import check_citations, parse_citations
from eval.metrics.clarification import is_clarification
from eval.metrics.doc_filter_correct import doc_filter_correct
from eval.metrics.faithfulness import faithfulness_score
from eval.metrics.iterative_retrieval_used import iterative_retrieval_used
from eval.metrics.refusal import is_refusal
from eval.metrics.retrieval import recall_at_k
from eval.metrics.task_completion import task_completed
from eval.metrics.tool_selection import tool_selection_match


# Single Chroma "session" holding all eval PDF chunks (dedupe by doc_name).
EVAL_SHARED_CORPUS_SESSION_ID = "eval_shared_corpus"


class _EvalCorpusVectorStore:
    """Use one shared collection for retrieval; ignore per-question ``session_id``."""

    __slots__ = ("_inner", "_corpus_sid")

    def __init__(self, inner: ChromaStore, corpus_session_id: str) -> None:
        self._inner = inner
        self._corpus_sid = corpus_session_id

    def query(
        self,
        session_id: str,
        query_text: str,
        k: int = 5,
        doc_filter: list[str] | None = None,
    ):
        return self._inner.query(self._corpus_sid, query_text, k=k, doc_filter=doc_filter)

    def list_documents(self, session_id: str) -> list[dict]:
        return self._inner.list_documents(self._corpus_sid)


def _eval_log(qid: str, step: str, detail: str = "", *, style: str = "cyan") -> None:
    """One line per major step; ``detail`` is escaped for user-controlled text."""
    suffix = f" — {escape(detail)}" if detail else ""
    rich_print(f"[bold]{escape(qid)}[/bold] [{style}]{escape(step)}[/{style}]{suffix}")


def _eval_log_done(qid: str, step: str, seconds: float) -> None:
    rich_print(f"[bold]{escape(qid)}[/bold] [green]✓ {escape(step)}[/green] [dim]({seconds:.1f}s)[/dim]")


SLO = {
    "task_completion_overall": 0.70,
    "task_completion_multi_hop": 0.50,
    "task_completion_numerical": 0.50,
    "faithfulness": 0.80,
    "citation_recall": 0.85,
    "retrieval_recall_at_5": 0.80,
    "refusal_rate": 0.90,
    "clarification_rate": 0.80,
}


@dataclass
class QResult:
    id: str
    category: str
    question: str
    answer_text: str
    actual_tools: list[str]
    retrieval_iterations: int
    retrieved_pages: list[int]
    retrieved_chunks: list[dict]
    citations_parsed: list[tuple[str, int]]
    metrics: dict
    duration_s: float
    usage: dict


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _indexed_doc_names(store: ChromaStore, corpus_session_id: str) -> set[str]:
    rows = store.list_documents(corpus_session_id)
    return {str(d.get("doc_name", "")) for d in rows if d.get("doc_name")}


def _ensure_eval_docs_indexed(
    store: ChromaStore, corpus_session_id: str, pdf_dir: Path, doc_names: list[str]
) -> tuple[list[str], list[str]]:
    """Index only PDFs not yet present in the shared eval corpus. Returns (added, skipped)."""
    present = _indexed_doc_names(store, corpus_session_id)
    added: list[str] = []
    skipped: list[str] = []
    for name in doc_names:
        if name in present:
            skipped.append(name)
            continue
        path = pdf_dir / name
        if not path.exists():
            raise FileNotFoundError(f"PDF missing for eval: {path}")
        pages = parse_pdf(path)
        chunks = chunk_pages(pages, doc_id=name, doc_name=name)
        store.add_chunks(corpus_session_id, chunks)
        present.add(name)
        added.append(name)
    return added, skipped


def _retrieved_pages_from_tool_calls(tool_calls: list[dict]) -> list[int]:
    pages: list[int] = []
    for tc in tool_calls:
        if tc.get("tool_name") != "retrieve_from_docs":
            continue
        result = tc.get("result")
        if not result:
            continue
        chunks = result.get("chunks") if isinstance(result, dict) else getattr(result, "chunks", None)
        if not chunks:
            continue
        for c in chunks:
            page = c.get("page_number") if isinstance(c, dict) else getattr(c, "page_number", None)
            if page is not None:
                pages.append(int(page))
    return pages


def _retrieved_chunks_from_tool_calls(tool_calls: list[dict]) -> list[dict]:
    out: list[dict] = []
    for tc in tool_calls:
        if tc.get("tool_name") != "retrieve_from_docs":
            continue
        result = tc.get("result")
        if not result:
            continue
        chunks = result.get("chunks") if isinstance(result, dict) else getattr(result, "chunks", None) or []
        for c in chunks:
            if isinstance(c, dict):
                out.append(c)
            else:
                out.append({
                    "text": getattr(c, "text", ""),
                    "doc_name": getattr(c, "doc_name", ""),
                    "page_number": getattr(c, "page_number", 0),
                    "score": getattr(c, "score", 0.0),
                })
    return out


async def _run_one(
    q: dict,
    settings,
    store: ChromaStore,
    pdf_dir: Path,
    web_client: TavilyClient,
    is_adversarial: bool,
) -> QResult:
    qid = q["id"]
    cat = q.get("category", "adversarial")
    _eval_log(qid, "start", f"category={cat}" + (" [adv]" if is_adversarial else ""), style="magenta")

    logical_session_id = f"eval_{qid}"
    _eval_log(
        qid,
        "session",
        f"logical id={logical_session_id} (fresh run); vectors → {EVAL_SHARED_CORPUS_SESSION_ID}",
        style="dim",
    )

    docs = q.get("expected_docs") or []
    if docs:
        t_idx = time.time()
        _eval_log(qid, "index", f"candidates: {', '.join(docs)}")
        added, skipped = _ensure_eval_docs_indexed(store, EVAL_SHARED_CORPUS_SESSION_ID, pdf_dir, docs)
        detail = []
        if added:
            detail.append(f"indexed: {', '.join(added)}")
        if skipped:
            detail.append(f"reuse: {', '.join(skipped)}")
        _eval_log_done(qid, "corpus " + ("; ".join(detail) if detail else "unchanged"), time.time() - t_idx)
    else:
        _eval_log(qid, "index", "skipped (no expected_docs)", style="dim")

    vector_store = _EvalCorpusVectorStore(store, EVAL_SHARED_CORPUS_SESSION_ID)
    deps = AgentDeps(
        session_id=logical_session_id,
        vector_store=vector_store,
        web_search_client=web_client,
        retrieval_threshold=settings.retrieval_threshold,
        retrieval_k=settings.retrieval_k,
    )

    _eval_log(qid, "agent", "run_agent (LLM + tools)")
    t0 = time.time()
    response = await run_agent(message_history=[], user_message=q["question"], deps=deps)
    duration = time.time() - t0
    _eval_log_done(qid, "run_agent", duration)

    tool_calls = response.tool_calls or []
    actual_tools = [tc["tool_name"] for tc in tool_calls]
    retrieved_pages = _retrieved_pages_from_tool_calls(tool_calls)
    retrieved_chunks = _retrieved_chunks_from_tool_calls(tool_calls)
    citations = parse_citations(response.answer.text)

    t_struct = time.time()
    _eval_log(qid, "metrics", "tool selection, doc filter, iterative retrieval, recall@k")
    metrics: dict = {
        "tool_selection_match": tool_selection_match(
            actual_tools, q.get("expected_tools", []),
            loose_multi_retrieve=(q.get("category") == "multi_hop"),
        ),
        "doc_filter_correct": doc_filter_correct(tool_calls, q.get("expected_doc_filter")),
        "iterative_retrieval_used": iterative_retrieval_used(tool_calls),
    }

    if q.get("expected_pages"):
        metrics["recall_at_1"] = recall_at_k(retrieved_pages, q["expected_pages"], 1)
        metrics["recall_at_3"] = recall_at_k(retrieved_pages, q["expected_pages"], 3)
        metrics["recall_at_5"] = recall_at_k(retrieved_pages, q["expected_pages"], 5)

    recall_note = ""
    if q.get("expected_pages"):
        recall_note = (
            f" R@1/3/5={metrics['recall_at_1']:.2f}/{metrics['recall_at_3']:.2f}/{metrics['recall_at_5']:.2f}"
        )
    _eval_log_done(
        qid,
        f"tool_match={metrics['tool_selection_match']} doc_filter={metrics['doc_filter_correct']} "
        f"iter_retrieve={metrics['iterative_retrieval_used']}{recall_note}",
        time.time() - t_struct,
    )

    if not is_adversarial:
        t_cite = time.time()
        _eval_log(qid, "metrics", "citation check (precision / recall vs chunks + ground truth)")
        chunks_by_page = {(c.get("doc_name", ""), int(c.get("page_number", 0))): c.get("text", "") for c in retrieved_chunks}
        cite_check = check_citations(response.answer.text, chunks_by_page, q.get("ground_truth", ""))
        metrics["citation_precision"] = cite_check.precision
        metrics["citation_recall"] = cite_check.recall
        _eval_log_done(
            qid,
            f"citation precision/recall = {cite_check.precision:.2f} / {cite_check.recall:.2f}",
            time.time() - t_cite,
        )

        t_faith = time.time()
        _eval_log(qid, "judge", f"faithfulness ({settings.judge_model})")
        faith = faithfulness_score(
            api_key=settings.gemini_api_key,
            judge_model=settings.judge_model,
            chunks=retrieved_chunks,
            answer_text=response.answer.text,
        )
        metrics["faithfulness"] = faith.score
        metrics["faithfulness_unsupported"] = faith.unsupported_claims
        _eval_log_done(qid, f"faithfulness score = {faith.score:.2f}", time.time() - t_faith)

        t_task = time.time()
        _eval_log(qid, "judge", f"task completion ({settings.judge_model})")
        completion = task_completed(
            question=q["question"],
            ground_truth=q.get("ground_truth", ""),
            answer_text=response.answer.text,
            actual_tools=actual_tools,
            expected_tools=q.get("expected_tools", []),
            category=q.get("category", ""),
            api_key=settings.gemini_api_key,
            judge_model=settings.judge_model,
        )
        metrics["task_completion"] = completion.completed
        metrics["task_completion_judge_pass"] = completion.judge_pass
        _eval_log_done(
            qid,
            f"task_completion={completion.completed} (judge_pass={completion.judge_pass}, tools_ok={completion.tools_match}, has_cite={completion.has_citation})",
            time.time() - t_task,
        )
    else:
        t_adv = time.time()
        subset = q.get("subset")
        _eval_log(qid, "adversarial", f"subset={subset}")
        if subset == "should_refuse":
            metrics["refusal"] = is_refusal(response.answer.text)
        elif subset == "should_clarify":
            metrics["clarification"] = is_clarification(response.answer.text)
        _eval_log_done(qid, "adversarial heuristics", time.time() - t_adv)

    rich_print(
        f"[bold]{escape(qid)}[/bold] [yellow]summary[/yellow] "
        f"tools=[dim]{escape(', '.join(actual_tools) or 'none')}[/dim] "
        f"iterations={response.answer.retrieval_iterations} "
        f"[dim]agent {duration:.1f}s[/dim]"
    )

    return QResult(
        id=q["id"], category=q.get("category", "adversarial"), question=q["question"],
        answer_text=response.answer.text,
        actual_tools=actual_tools,
        retrieval_iterations=response.answer.retrieval_iterations,
        retrieved_pages=retrieved_pages,
        retrieved_chunks=retrieved_chunks,
        citations_parsed=citations,
        metrics=metrics,
        duration_s=duration,
        usage=response.usage,
    )


def _aggregate(results: list[QResult]) -> dict:
    by_cat: dict[str, list[QResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    def _mean(values, default=None):
        clean = [v for v in values if isinstance(v, (int, float))]
        return sum(clean) / len(clean) if clean else default

    agg: dict = {"per_category": {}}
    for cat, rs in by_cat.items():
        agg["per_category"][cat] = {
            "n": len(rs),
            "task_completion_rate": _mean([1 if r.metrics.get("task_completion") else 0 for r in rs if "task_completion" in r.metrics]),
            "faithfulness": _mean([r.metrics.get("faithfulness") for r in rs]),
            "citation_recall": _mean([r.metrics.get("citation_recall") for r in rs]),
            "recall_at_5": _mean([r.metrics.get("recall_at_5") for r in rs]),
            "iterative_retrieval_rate": _mean([1 if r.metrics.get("iterative_retrieval_used") else 0 for r in rs]),
        }
    agg["overall"] = {
        "task_completion_rate": _mean([1 if r.metrics.get("task_completion") else 0 for r in results if "task_completion" in r.metrics]),
        "faithfulness": _mean([r.metrics.get("faithfulness") for r in results]),
        "citation_recall": _mean([r.metrics.get("citation_recall") for r in results]),
        "recall_at_5": _mean([r.metrics.get("recall_at_5") for r in results]),
        "refusal_rate": _mean([1 if r.metrics.get("refusal") else 0 for r in results if "refusal" in r.metrics]),
        "clarification_rate": _mean([1 if r.metrics.get("clarification") else 0 for r in results if "clarification" in r.metrics]),
    }
    return agg


def _slo_status(metric_value, target) -> str:
    if metric_value is None:
        return "N/A"
    return "GREEN" if metric_value >= target else "RED"


async def _run_all(args) -> None:
    settings = get_settings()
    pdf_dir = Path(args.pdf_dir).resolve()
    chroma_dir = Path(args.chroma_dir).resolve()
    embedder = GeminiEmbedder(api_key=settings.gemini_api_key, model=settings.embedding_model)
    web_client = TavilyClient(api_key=settings.tavily_api_key)
    store = ChromaStore(persist_dir=str(chroma_dir), embedder=embedder)

    dataset = _load_jsonl(Path(args.dataset))
    adversarial = _load_jsonl(Path(args.adversarial)) if args.adversarial else []

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rich_print()
    rich_print(
        "[bold white on blue] eval run [/bold white on blue] "
        f"[dim]dataset[/dim] {escape(str(Path(args.dataset)))} "
        f"([cyan]{len(dataset)}[/cyan] questions)"
        + (f" + [cyan]{len(adversarial)}[/cyan] adversarial" if adversarial else "")
    )
    rich_print(f"[dim]pdf_dir[/dim] {escape(str(pdf_dir))}  [dim]chroma_dir[/dim] {escape(str(chroma_dir))}")
    rich_print(
        f"[dim]shared eval corpus[/dim] [cyan]{escape(EVAL_SHARED_CORPUS_SESSION_ID)}[/cyan] "
        f"(collection session_{escape(EVAL_SHARED_CORPUS_SESSION_ID)})"
    )
    rich_print(f"[dim]output[/dim] {escape(str(output_path))}")
    rich_print()

    results: list[QResult] = []
    with output_path.open("w") as fout:
        for q in dataset:
            r = await _run_one(q, settings, store, pdf_dir, web_client, is_adversarial=False)
            fout.write(json.dumps({**asdict(r), "citations_parsed": [list(c) for c in r.citations_parsed]}) + "\n")
            results.append(r)
            rich_print(f"[bold green]✓ wrote[/bold green] {escape(r.id)} → JSONL  [dim]agent {r.duration_s:.1f}s[/dim]\n")
        for q in adversarial:
            r = await _run_one(q, settings, store, pdf_dir, web_client, is_adversarial=True)
            fout.write(json.dumps({**asdict(r), "citations_parsed": [list(c) for c in r.citations_parsed]}) + "\n")
            results.append(r)
            rich_print(
                f"[bold green]✓ wrote[/bold green] {escape(r.id)} adv/{escape(str(q.get('subset', '')))} "
                f"[dim]agent {r.duration_s:.1f}s[/dim]\n"
            )

    agg = _aggregate(results)
    rich_print("[bold]═══ AGGREGATE ═══[/bold]")
    rich_print(json.dumps(agg, indent=2))

    rich_print()
    rich_print("[bold]═══ SLO COMPARISON ═══[/bold]")
    overall = agg["overall"]
    by_cat = agg["per_category"]
    rows = [
        ("task_completion_rate (overall)", overall.get("task_completion_rate"), SLO["task_completion_overall"]),
        ("task_completion_rate (multi_hop)", by_cat.get("multi_hop", {}).get("task_completion_rate"), SLO["task_completion_multi_hop"]),
        ("task_completion_rate (numerical)", by_cat.get("numerical", {}).get("task_completion_rate"), SLO["task_completion_numerical"]),
        ("faithfulness", overall.get("faithfulness"), SLO["faithfulness"]),
        ("citation_recall", overall.get("citation_recall"), SLO["citation_recall"]),
        ("retrieval_recall_at_5", overall.get("recall_at_5"), SLO["retrieval_recall_at_5"]),
        ("refusal_rate", overall.get("refusal_rate"), SLO["refusal_rate"]),
        ("clarification_rate", overall.get("clarification_rate"), SLO["clarification_rate"]),
    ]
    for name, value, target in rows:
        status = _slo_status(value, target)
        v = f"{value:.3f}" if isinstance(value, (int, float)) else "N/A"
        st_style = "green" if status == "GREEN" else ("yellow" if status == "N/A" else "red")
        rich_print(f"  [{st_style}]{status}[/{st_style}] {escape(name)}: {v} [dim](target {target})[/dim]")

    rich_print()
    rich_print(f"[bold]Results[/bold] {escape(str(output_path))}")
    rich_print(f"[dim]Run timestamp[/dim] {escape(datetime.now(timezone.utc).isoformat())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/dataset/questions.jsonl")
    parser.add_argument("--adversarial", default="eval/dataset/adversarial.jsonl")
    parser.add_argument("--pdf-dir", default="data/pdfs")
    parser.add_argument("--chroma-dir", default="./chroma_db_eval")
    parser.add_argument("--output", default=f"eval/results/run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl")
    args = parser.parse_args()
    asyncio.run(_run_all(args))


if __name__ == "__main__":
    main()
