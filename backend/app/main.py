"""Entry point for the Hybrid RAG backend API."""

import logging

from fastapi import FastAPI

from app.api import routes_documents, routes_search
from app.core.config import get_settings
from app.core.dependencies import get_graph_builder, get_neo4j_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="Hybrid RAG API",
    description="Grounded question answering over uploaded documents.",
    version="0.3.0",
)

app.include_router(routes_documents.router)
app.include_router(routes_search.router)


@app.on_event("startup")
def prepare_graph() -> None:
    """Create the graph schema at startup, if Neo4j is available."""
    client = get_neo4j_client()
    if client.is_configured and client.verify():
        try:
            client.ensure_schema()
            builder = get_graph_builder()
            if builder is not None:
                builder.ensure_schema()
        except Exception:
            logger.exception("Could not prepare the Neo4j schema.")
    else:
        logger.warning("Starting without a graph connection.")


@app.get("/health", tags=["health"])
def health() -> dict:
    """Check the server is running and show which features are switched on."""
    client = get_neo4j_client()
    return {
        "status": "ok",
        "retrieval": {
            "vector": settings.enable_vector_retrieval,
            "bm25": settings.enable_bm25,
            "knowledge_graph": settings.enable_kg_retrieval,
            "reranker": settings.enable_reranker,
        },
        "chunk_size": settings.chunk_size,
        "fusion_method": settings.fusion_method,
        "neo4j_configured": client.is_configured,
        "neo4j_connected": client.verify() if client.is_configured else False,
        "llm_configured": bool(settings.openai_api_key.get_secret_value()),
    }