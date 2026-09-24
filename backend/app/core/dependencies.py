"""Shared objects handed to the API routes."""

from functools import lru_cache
from app.retrieval.hybrid_retriever import HybridRetriever

from app.core.config import get_settings
from app.core.indexes import BM25Index
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.store import DocumentStore
from app.vectorstore.base import VectorStore
from app.vectorstore.embeddings import get_embedding_model
from app.vectorstore.faiss_store import FaissVectorStore


@lru_cache
def get_document_store() -> DocumentStore:
    settings = get_settings()
    return DocumentStore(
        upload_dir=settings.resolve_path(settings.upload_dir),
        registry_path=settings.resolve_path("data/processed") / "registry.json",
    )


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return FaissVectorStore(
        index_dir=settings.resolve_path(settings.faiss_index_path),
        embedding_model=get_embedding_model(),
    )


@lru_cache
def get_bm25_index() -> BM25Index:
    return BM25Index(get_document_store())


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    settings = get_settings()
    return IngestionPipeline(
        store=get_document_store(),
        vector_store=get_vector_store(),
        bm25_index=get_bm25_index(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    settings = get_settings()
    return HybridRetriever(
        vector_retriever=get_vector_store() if settings.enable_vector_retrieval else None,
        bm25_retriever=(
            get_bm25_index().retriever if settings.enable_bm25 else None
        ),
        graph_retriever=None,  # Neo4j arrives in a later phase
        vector_top_k=settings.vector_top_k,
        bm25_top_k=settings.bm25_top_k,
        kg_top_k=settings.kg_top_k,
        fusion_method=settings.fusion_method,
    )