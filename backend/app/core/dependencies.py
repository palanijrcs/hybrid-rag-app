"""Shared objects handed to the API routes."""

from functools import lru_cache

from app.core.config import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.store import DocumentStore


@lru_cache
def get_document_store() -> DocumentStore:
    settings = get_settings()
    return DocumentStore(
        upload_dir=settings.resolve_path(settings.upload_dir),
        registry_path=settings.resolve_path("data/processed") / "registry.json",
    )


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    settings = get_settings()
    return IngestionPipeline(
        store=get_document_store(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )