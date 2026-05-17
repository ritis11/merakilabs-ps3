"""Clarification detection. Looks for question marks combined with disambiguation words."""
import re

CLARIFICATION_PATTERNS = [
    r"did you mean",
    r"could you (?:clarify|specify)",
    r"do you mean (?:fy|fiscal|calendar)",
    r"which (?:fiscal year|calendar year|year)",
    r"are you asking about",
]


def is_clarification(answer_text: str) -> bool:
    text = answer_text.lower()
    has_question = "?" in answer_text
    has_keyword = any(re.search(p, text) for p in CLARIFICATION_PATTERNS)
    return has_question and has_keyword
