"""Shared test setup."""

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import get_document_store, get_ingestion_pipeline
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.store import DocumentStore
from app.main import app


@pytest.fixture
def client(tmp_path):
    """A test client whose data is stored in a temporary folder."""
    store = DocumentStore(
        upload_dir=tmp_path / "uploads",
        registry_path=tmp_path / "processed" / "registry.json",
    )
    pipeline = IngestionPipeline(store=store, chunk_size=400, chunk_overlap=60)

    app.dependency_overrides[get_document_store] = lambda: store
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()