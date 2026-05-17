"""Task completion: composite of (judge_pass AND tools_match AND has_citation)."""
import json
from dataclasses import dataclass

import google.genai as genai

from eval.metrics.citation_check import parse_citations
from eval.metrics.tool_selection import tool_selection_match


JUDGE_TASK_PROMPT = """You are judging whether an agent's answer correctly addresses the question.

You will receive: the question, the ground-truth answer, and the agent's answer.

Output ONLY a JSON object on a single line:
{"pass": true, "reason": "short explanation"}

- pass = true if the agent's answer is substantively correct relative to the ground truth (numeric values within 1%, named entities match, key facts present). Style and phrasing differences are OK.
- pass = false if the agent fabricated, contradicted ground truth, missed a key fact, or refused inappropriately.

Do not output any text outside the JSON object.
"""


@dataclass
class TaskCompletionResult:
    completed: bool
    judge_pass: bool
    tools_match: bool
    has_citation: bool
    judge_reason: str | None = None


def _judge_pass(api_key: str, judge_model: str, question: str, ground_truth: str, answer_text: str) -> bool:
    client = genai.Client(api_key=api_key)
    prompt = (
        f"{JUDGE_TASK_PROMPT}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== GROUND TRUTH ===\n{ground_truth}\n\n"
        f"=== AGENT ANSWER ===\n{answer_text}"
    )
    response = client.models.generate_content(model=judge_model, contents=prompt)
    raw = (getattr(response, "text", "") or "").strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    first = raw.find("{")
    last = raw.rfind("}")
    if first < 0 or last < 0:
        return False
    try:
        payload = json.loads(raw[first : last + 1])
        return bool(payload.get("pass", False))
    except Exception:
        return False


def task_completed(
    question: str,
    ground_truth: str,
    answer_text: str,
    actual_tools: list[str],
    expected_tools: list[str],
    category: str,
    api_key: str,
    judge_model: str,
) -> TaskCompletionResult:
    loose = category == "multi_hop"
    tools_ok = tool_selection_match(actual_tools, expected_tools, loose_multi_retrieve=loose)
    citation_required = category != "meta"
    has_cite = (not citation_required) or bool(parse_citations(answer_text))
    judge_ok = _judge_pass(
        api_key=api_key, judge_model=judge_model,
        question=question, ground_truth=ground_truth, answer_text=answer_text,
    )
    completed = bool(judge_ok and tools_ok and has_cite)
    return TaskCompletionResult(
        completed=completed, judge_pass=judge_ok, tools_match=tools_ok,
        has_citation=has_cite,
    )
