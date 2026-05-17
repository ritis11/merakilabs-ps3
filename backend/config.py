"""Application settings loaded from environment via Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Required secrets
    gemini_api_key: str
    tavily_api_key: str

    # Storage paths
    chroma_persist_dir: str = "./chroma_db"
    session_dump_path: str = "./sessions.json"

    # Retrieval
    retrieval_threshold: float = 0.3
    retrieval_k: int = 5

    # Session memory
    history_cap: int = 10

    # Agent
    agent_request_limit: int = 5
    agent_model: str = "gemini-3-flash-preview"
    judge_model: str = "gemini-3.1-pro-preview"
    embedding_model: str = "gemini-embedding-001"

    # CORS (Streamlit default port)
    cors_origins: list[str] = ["http://localhost:8501"]


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
