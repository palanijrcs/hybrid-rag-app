"""Vector search endpoint (a stepping stone towards full hybrid retrieval)."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.config import get_settings
from app.core.dependencies import get_document_store, get_vector_store
from app.ingestion.store import DocumentStore
from app.models.schemas import IndexStats, RetrievedChunk, SearchResponse
from app.vectorstore.base import VectorStore

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Your question or keywords"),
    top_k: int | None = Query(None, ge=1, le=50),
    vector_store: VectorStore = Depends(get_vector_store),
) -> SearchResponse:
    """Find the chunks closest in meaning to the query."""
    if not q.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Query cannot be empty.")

    limit = top_k or get_settings().vector_top_k
    hits = vector_store.search(q, top_k=limit)

    return SearchResponse(
        query=q,
        results=[RetrievedChunk(**hit.__dict__) for hit in hits],
    )


@router.get("/index/stats", response_model=IndexStats)
def index_stats(
    store: DocumentStore = Depends(get_document_store),
    vector_store: VectorStore = Depends(get_vector_store),
) -> IndexStats:
    """How much content is currently indexed."""
    documents = store.list_documents()
    return IndexStats(
        documents=len(documents),
        chunks=sum(d.chunk_count for d in documents),
        vectors=vector_store.count(),
    )