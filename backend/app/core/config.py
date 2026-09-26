"""Central application settings, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = hybrid-rag-app/ (three folders above this file)
def _find_project_root() -> Path:
    """Works both on the host (repo root) and in Docker (/app)."""
    here = Path(__file__).resolve()
    # backend/app/core/config.py -> repo root is 3 levels up
    candidate = here.parents[3]
    if (candidate / ".env").exists() or (candidate / "docker-compose.yml").exists():
        return candidate
    # In the container the app is copied to /app
    return Path("/app")


PROJECT_ROOT = _find_project_root()


class Settings(BaseSettings):
    """All configurable values for the app. Defaults are safe for local dev."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: str = "openai"
    llm_model: str = ""
    openai_api_key: SecretStr = SecretStr("")

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Neo4j AuraDB
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: SecretStr = SecretStr("")
    neo4j_database: str = "neo4j"

    # Storage
    upload_dir: str = "data/uploads"
    faiss_index_path: str = "data/faiss"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    vector_top_k: int = 10
    bm25_top_k: int = 10
    kg_top_k: int = 10
    rerank_top_k: int = 5
    rerank_candidates: int = 30  # how many fused results the re-ranker scores

    enable_vector_retrieval: bool = True
    enable_bm25: bool = True
    enable_kg_retrieval: bool = True
    enable_reranker: bool = True
    fusion_method: str = "rrf"
    min_relevance_score: float = 0.0  # re-ranker threshold, 0..1

    # Context builder (evidence given to the LLM)
    context_max_chars: int = 6000
    context_max_sources: int = 8

    # Ingestion quality: don't index text that was extracted incorrectly
    skip_garbled_chunks: bool = True


    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    def resolve_path(self, relative: str) -> Path:
        """Turn a path from .env (like data/uploads) into a full path."""
        return PROJECT_ROOT / relative


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them everywhere."""
    return Settings()