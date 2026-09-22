"""Central application settings, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = hybrid-rag-app/ (three folders above this file)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


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

    enable_vector_retrieval: bool = True
    enable_bm25: bool = True
    enable_kg_retrieval: bool = True
    enable_reranker: bool = True

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