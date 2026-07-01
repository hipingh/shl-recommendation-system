"""
app/main.py

FastAPI application entry point.

Lifespan:
  On startup:
    1. Load settings from .env.
    2. Load catalog (from disk or remote).
    3. Initialise embedding model.
    4. Initialise ChromaDB and ingest catalog (idempotent).
    5. Wire all services together.
    6. Store AgentOrchestrator in app.state.

  On shutdown:
    Graceful cleanup (log shutdown event).

The app is fully dependency-injected — no global singletons outside
of app.state.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.services.agent import AgentOrchestrator
from app.services.clarification_service import ClarificationService
from app.services.comparison_service import ComparisonService
from app.services.conversation_service import ConversationService
from app.services.intent_service import IntentService
from app.services.llm_service import LLMService
from app.services.recommendation_service import RecommendationService
from app.services.refusal_service import RefusalService
from app.services.retrieval_service import RetrievalService
from app.services.scraper import CatalogLoader
from app.services.session_service import SessionService
from app.vectorstore.chroma import ChromaStore
from app.vectorstore.embedding import EmbeddingService

# ── Logging Setup ─────────────────────────────────────────────


def _configure_logging(log_level: str) -> None:
    """Configure structured logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy third-party loggers
    for noisy in ("chromadb", "sentence_transformers", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Handles startup initialisation and graceful shutdown.
    All expensive objects are created once here and stored in app.state.
    """
    settings = get_settings()
    _configure_logging(settings.log_level)

    logger.info("=" * 60)
    logger.info("SHL Assessment Recommender — Starting up")
    logger.info("Model: %s | Port: %s", settings.groq_model, settings.port)
    logger.info("=" * 60)

    # ── 1. Load Catalog ───────────────────────────────────────
    logger.info("Loading SHL catalog from: %s", settings.catalog_abs_path)
    catalog_loader = CatalogLoader(catalog_path=settings.catalog_abs_path)
    try:
        assessments = catalog_loader.load()
        logger.info("Catalog loaded: %d assessments.", len(assessments))
    except Exception as exc:
        logger.critical("Failed to load catalog: %s", exc)
        raise RuntimeError(f"Catalog load failed: {exc}") from exc

    if not assessments:
        logger.critical("Catalog is empty — cannot start without assessment data.")
        raise RuntimeError("Empty catalog. Add data to catalog.json before starting.")

    # ── 2. Embedding Model ────────────────────────────────────
    logger.info("Loading embedding model: %s", settings.embedding_model)
    try:
        embedding_service = EmbeddingService(model_name=settings.embedding_model)
        logger.info(
            "Embedding model ready. Dimension: %d", embedding_service.dimension
        )
    except Exception as exc:
        logger.critical("Failed to load embedding model: %s", exc)
        raise RuntimeError(f"Embedding model load failed: {exc}") from exc

    # ── 3. ChromaDB ───────────────────────────────────────────
    logger.info("Initialising ChromaDB at: %s", settings.chroma_db_abs_path)
    try:
        chroma_store = ChromaStore(
            embedding_service=embedding_service,
            db_path=settings.chroma_db_abs_path,
            collection_name=settings.chroma_collection_name,
        )
        chroma_store.ingest(assessments)
        logger.info("ChromaDB ready. Documents: %d", chroma_store.count())
    except Exception as exc:
        logger.critical("ChromaDB initialisation failed: %s", exc)
        raise RuntimeError(f"ChromaDB init failed: {exc}") from exc

    # ── 4. SQLite Session DB ──────────────────────────────────
    logger.info("Initialising SQLite session database at: %s", settings.sqlite_db_abs_path)
    try:
        session_service = SessionService(db_path=settings.sqlite_db_abs_path)
        session_service.init_db()
    except Exception as exc:
        logger.critical("SQLite session database initialization failed: %s", exc)
        raise RuntimeError(f"SQLite init failed: {exc}") from exc

    # ── 5. LLM Service ────────────────────────────────────────
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        logger.critical(
            "GROQ_API_KEY is not set. Please add it to your .env file."
        )
        raise RuntimeError("GROQ_API_KEY is required. Set it in .env")

    logger.info("Initialising LLM service: %s", settings.groq_model)
    llm_service = LLMService(settings=settings)

    # ── 6. Wire All Services ──────────────────────────────────
    intent_service = IntentService(llm_service=llm_service)
    conversation_service = ConversationService(llm_service=llm_service)
    retrieval_service = RetrievalService(
        chroma_store=chroma_store,
        top_k=settings.retrieval_top_k,
    )
    recommendation_service = RecommendationService(llm_service=llm_service)
    comparison_service = ComparisonService(
        llm_service=llm_service,
        retrieval_service=retrieval_service,
    )
    clarification_service = ClarificationService(llm_service=llm_service)
    refusal_service = RefusalService()

    agent = AgentOrchestrator(
        intent_service=intent_service,
        conversation_service=conversation_service,
        retrieval_service=retrieval_service,
        recommendation_service=recommendation_service,
        comparison_service=comparison_service,
        clarification_service=clarification_service,
        refusal_service=refusal_service,
    )

    # ── 7. Store in app.state ─────────────────────────────────
    app.state.agent = agent
    app.state.settings = settings

    logger.info("All services initialised. Ready to serve requests.")
    logger.info("=" * 60)

    yield  # ── Server is running ─────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("SHL Assessment Recommender — Shutting down gracefully.")


# ── Application Factory ───────────────────────────────────────


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="SHL Assessment Recommender",
        description=(
            "Conversational AI agent for recommending SHL Individual Test Solutions. "
            "Powered by RAG with ChromaDB and Groq LLaMA-3.3-70B."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global Exception Handler ──────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An unexpected error occurred. Please try again later."
            },
        )

    # ── Register Routes ───────────────────────────────────────
    app.include_router(router)

    return app


# ── App Instance ──────────────────────────────────────────────

app = create_app()


# ── Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
