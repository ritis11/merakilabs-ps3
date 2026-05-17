"""FastAPI application entrypoint."""
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tavily import TavilyClient

from backend.config import get_settings
from backend.logging_config import configure_logging
from backend.retrieval.embedder import GeminiEmbedder
from backend.retrieval.store import ChromaStore
from backend.routes.documents import router as documents_router
from backend.routes.messages import router as messages_router
from backend.routes.sessions import router as sessions_router
from backend.session.store import SessionManager
from dotenv import load_dotenv
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: build clients and SessionManager onto app.state.
    Shutdown: dump session state."""
    configure_logging(json_output=True)
    settings = get_settings()

    embedder = GeminiEmbedder(api_key=settings.gemini_api_key, model=settings.embedding_model)
    vector_store = ChromaStore(persist_dir=settings.chroma_persist_dir, embedder=embedder)
    session_manager = SessionManager(dump_path=settings.session_dump_path, history_cap=settings.history_cap)
    session_manager.load()
    web_search_client = TavilyClient(api_key=settings.tavily_api_key)

    app.state.session_manager = session_manager
    app.state.vector_store = vector_store
    app.state.web_search_client = web_search_client
    app.state.embedder = embedder
    app.state.settings = settings

    try:
        yield
    finally:
        session_manager.dump()


def create_app() -> FastAPI:
    app = FastAPI(title="Meraki 10-K Agent", lifespan=lifespan)

    # CORS - Streamlit default port + any user-overridden origins
    settings_origins = ["http://localhost:8501"]
    try:
        settings_origins = get_settings().cors_origins
    except Exception:
        pass

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(sessions_router)
    app.include_router(documents_router)
    app.include_router(messages_router)
    return app


app = create_app()
