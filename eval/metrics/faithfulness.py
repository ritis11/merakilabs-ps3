"""Faithfulness metric via gemini-2.5-pro judge.

Judge prompt is PINNED — every change requires bumping FAITHFULNESS_PROMPT_VERSION
and re-running v0_baseline. Eval results without a recorded prompt version are
not comparable.
"""
import json
from dataclasses import dataclass

import google.genai as genai


FAITHFULNESS_PROMPT_VERSION = "judge-v0.1.0"

JUDGE_PROMPT = """You are evaluating whether an agent's answer is faithful to the retrieved chunks.

You will receive:
- A list of chunks the agent retrieved from a document corpus.
- The agent's answer.

Your job: determine whether every factual claim in the answer is directly supported by the chunks. Paraphrase is OK; fabrication is not. Numeric values must match (allow rounding within 1%). Citations like `[doc, p. N]` are not claims themselves — focus on the prose.

Output ONLY a JSON object on a single line:
{"score": 1.0, "unsupported_claims": []}

- score = 1.0 if every claim is supported.
- score = 0.5 if most claims are supported but at least one is unsupported, or one is contradicted.
- score = 0.0 if the answer is mostly fabricated or contradicts the chunks.
- unsupported_claims = list of short strings naming each unsupported/contradicted claim, empty if score=1.0.

Do not output any text outside the JSON object.
"""


@dataclass
class FaithfulnessResult:
    score: float
    unsupported_claims: list[str]
    raw: str


def _build_client(api_key: str, model: str):
    return genai.Client(api_key=api_key)


def faithfulness_score(
    api_key: str,
    judge_model: str,
    chunks: list[dict],
    answer_text: str,
) -> FaithfulnessResult:
    """Run the judge. Returns FaithfulnessResult."""
    client = _build_client(api_key, judge_model)
    chunks_block = "\n\n".join(
        f"[{c.get('doc_name')}, p. {c.get('page_number')}]\n{c.get('text','')}"
        for c in chunks
    )
    prompt = (
        f"{JUDGE_PROMPT}\n\n"
        f"=== CHUNKS ===\n{chunks_block or '(no chunks retrieved)'}\n\n"
        f"=== ANSWER ===\n{answer_text}"
    )
    response = client.models.generate_content(model=judge_model, contents=prompt)
    raw = (getattr(response, "text", "") or "").strip()
    try:
        clean = raw.strip("`")
        if clean.lower().startswith("json"):
            clean = clean[4:]
        clean = clean.strip()
        first_brace = clean.find("{")
        last_brace = clean.rfind("}")
        payload = json.loads(clean[first_brace : last_brace + 1])
        return FaithfulnessResult(
            score=float(payload.get("score", 0.0)),
            unsupported_claims=list(payload.get("unsupported_claims", [])),
            raw=raw,
        )
    except Exception:
        return FaithfulnessResult(score=0.0, unsupported_claims=[raw], raw=raw)
