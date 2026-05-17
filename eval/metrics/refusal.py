"""Refusal detection. Keyword heuristic; can be augmented with LLM judge later."""
import re

REFUSAL_PATTERNS = [
    r"don'?t have information",
    r"not (?:in|present in|available in) (?:the )?(?:uploaded )?(?:documents|corpus|reports)",
    r"cannot answer",
    r"i (?:can'?t|cannot) (?:provide|tell|give)",
    r"outside (?:the |my )?(?:scope|corpus)",
    r"no (?:information|data|details) (?:about|on)",
]


def is_refusal(answer_text: str) -> bool:
    text = answer_text.lower()
    return any(re.search(p, text) for p in REFUSAL_PATTERNS)
