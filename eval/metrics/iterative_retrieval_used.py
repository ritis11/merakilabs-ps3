"""FM-1 instrumentation: did the agent issue >1 retrieve_from_docs call?"""


def iterative_retrieval_used(tool_calls: list[dict]) -> bool:
    return sum(1 for tc in tool_calls if tc.get("tool_name") == "retrieve_from_docs") >= 2
