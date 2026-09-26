"""Shared test setup."""

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import (
    get_bm25_index,
    get_document_store,
    get_graph_builder,
    get_graph_indexer,
    get_hybrid_retriever,
    get_ingestion_pipeline,
    get_vector_store,
)
from app.core.indexes import BM25Index
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.store import DocumentStore
from app.main import app
from app.retrieval.hybrid_retriever import HybridRetriever
from app.vectorstore.embeddings import get_embedding_model
from app.vectorstore.faiss_store import FaissVectorStore


@pytest.fixture
def client(tmp_path):
    """A test client whose data is stored in a temporary folder."""
    store = DocumentStore(
        upload_dir=tmp_path / "uploads",
        registry_path=tmp_path / "processed" / "registry.json",
    )
    vector_store = FaissVectorStore(
        index_dir=tmp_path / "faiss", embedding_model=get_embedding_model()
    )
    bm25_index = BM25Index(store)
    hybrid = HybridRetriever(
        vector_retriever=vector_store,
        bm25_retriever=bm25_index.retriever,
        vector_top_k=10,
        bm25_top_k=10,
    )
    pipeline = IngestionPipeline(
        store=store,
        vector_store=vector_store,
        bm25_index=bm25_index,
        chunk_size=400,
        chunk_overlap=60,
    )

    app.dependency_overrides[get_document_store] = lambda: store
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_bm25_index] = lambda: bm25_index
    app.dependency_overrides[get_hybrid_retriever] = lambda: hybrid
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline
    # Never call the LLM or write to Neo4j from API tests
    app.dependency_overrides[get_graph_indexer] = lambda: None
    app.dependency_overrides[get_graph_builder] = lambda: None

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ---------------------------------------------------------------- Phase 19: markers
SLOW_TEST_FILES = {
    "test_embeddings.py", "test_faiss_store.py", "test_api_documents.py",
    "test_api_search.py", "test_hybrid_retriever.py", "test_reranker.py",
}


def pytest_collection_modifyitems(config, items):
    """Tag model-loading tests as slow so the quick loop can skip them."""
    for item in items:
        if item.path.name in SLOW_TEST_FILES:
            item.add_marker(pytest.mark.slow)
