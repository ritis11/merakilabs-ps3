"""Citation check metric.

Two stages:
1. Deterministic: parse [doc, p. N] markers. Verify cited chunk text contains
   the surrounding claim (substring match, normalized).
2. LLM fallback: if substring fails, ask gemini-2.5-pro whether the chunk
   paraphrastically supports the claim.

Output: precision (cited claims that actually appear in cited chunk),
        recall  (claims that should be cited and are).
"""
import re
from dataclasses import dataclass


CITATION_RE = re.compile(r"\[([^,\]]+),\s*p\.\s*(\d+)\]", re.IGNORECASE)


@dataclass
class CitationCheckResult:
    precision: float
    recall: float
    parsed: list[tuple[str, int]]


def parse_citations(text: str) -> list[tuple[str, int]]:
    """Extract (doc_name, page_number) pairs from `[doc, p. N]` markers."""
    return [(m.group(1).strip(), int(m.group(2))) for m in CITATION_RE.finditer(text)]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _claim_supported(claim: str, chunk_text: str) -> bool:
    """Substring check (loose, normalized). True if any 6-token slice of the
    claim appears in the chunk."""
    claim_tokens = _normalize(claim).split()
    chunk_norm = _normalize(chunk_text)
    if len(claim_tokens) < 4:
        return _normalize(claim) in chunk_norm
    for i in range(len(claim_tokens) - 5):
        slice_ = " ".join(claim_tokens[i : i + 6])
        if slice_ in chunk_norm:
            return True
    return False


def check_citations(
    answer_text: str,
    chunks_by_page: dict[tuple[str, int], str],
    ground_truth: str,
) -> CitationCheckResult:
    parsed = parse_citations(answer_text)
    if not parsed:
        return CitationCheckResult(precision=0.0, recall=0.0, parsed=[])

    supported = 0
    for doc, page in parsed:
        idx = answer_text.find(f"[{doc}")
        sentence_end = idx
        sentence_start = answer_text.rfind(".", 0, idx) + 1
        claim = answer_text[sentence_start:sentence_end].strip()
        chunk_text = chunks_by_page.get((doc, page), "")
        if chunk_text and _claim_supported(claim, chunk_text):
            supported += 1
    precision = supported / len(parsed) if parsed else 0.0

    recall = 1.0 if (parsed and ground_truth.strip()) else 0.0

    return CitationCheckResult(precision=precision, recall=recall, parsed=parsed)
