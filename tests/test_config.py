import os
from backend.config import Settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    s = Settings()
    assert s.gemini_api_key == "test-gemini-key"
    assert s.tavily_api_key == "test-tavily-key"
    assert s.chroma_persist_dir == "./chroma_db"
    assert s.session_dump_path == "./sessions.json"
    assert s.retrieval_threshold == 0.3
    assert s.retrieval_k == 5
    assert s.history_cap == 10
    assert s.agent_request_limit == 5
    assert s.agent_model == "gemini-3-flash-preview"
    assert s.judge_model == "gemini-3.1-pro-preview"
    assert s.embedding_model == "gemini-embedding-001"
