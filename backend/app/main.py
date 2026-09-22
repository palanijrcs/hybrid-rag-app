"""Entry point for the Hybrid RAG backend API."""

from fastapi import FastAPI

from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Hybrid RAG API",
    description="Grounded question answering over uploaded documents.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    """Check the server is running and show which features are switched on."""
    return {
        "status": "ok",
        "retrieval": {
            "vector": settings.enable_vector_retrieval,
            "bm25": settings.enable_bm25,
            "knowledge_graph": settings.enable_kg_retrieval,
            "reranker": settings.enable_reranker,
        },
        "chunk_size": settings.chunk_size,
        "neo4j_configured": bool(settings.neo4j_uri),
        "llm_configured": bool(settings.openai_api_key.get_secret_value()),
    }