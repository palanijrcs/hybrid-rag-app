"""Search endpoints (a stepping stone towards full hybrid retrieval)."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.config import get_settings
from app.core.dependencies import (
    get_bm25_index,
    get_document_store,
    get_vector_store,
)
from app.core.indexes import BM25Index
from app.ingestion.store import DocumentStore
from app.models.schemas import (
    ComparisonResponse,
    IndexStats,
    RetrievedChunk,
    SearchResponse,
)
from app.vectorstore.base import VectorStore

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Your question or keywords"),
    method: str = Query("vector", pattern="^(vector|bm25)$"),
    top_k: int | None = Query(None, ge=1, le=50),
    vector_store: VectorStore = Depends(get_vector_store),
    bm25_index: BM25Index = Depends(get_bm25_index),
) -> SearchResponse:
    """Search using one retrieval method."""
    if not q.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Query cannot be empty.")

    settings = get_settings()
    if method == "vector":
        hits = vector_store.search(q, top_k=top_k or settings.vector_top_k)
    else:
        hits = bm25_index.retriever.search(q, top_k=top_k or settings.bm25_top_k)

    return SearchResponse(
        query=q,
        method=method,
        results=[RetrievedChunk(**hit.__dict__) for hit in hits],
    )


@router.get("/search/compare", response_model=ComparisonResponse)
def compare(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    vector_store: VectorStore = Depends(get_vector_store),
    bm25_index: BM25Index = Depends(get_bm25_index),
) -> ComparisonResponse:
    """Run both retrievers on the same query, to see how they differ."""
    return ComparisonResponse(
        query=q,
        vector=[
            RetrievedChunk(**h.__dict__) for h in vector_store.search(q, top_k=top_k)
        ],
        bm25=[
            RetrievedChunk(**h.__dict__)
            for h in bm25_index.retriever.search(q, top_k=top_k)
        ],
    )


@router.get("/index/stats", response_model=IndexStats)
def index_stats(
    store: DocumentStore = Depends(get_document_store),
    vector_store: VectorStore = Depends(get_vector_store),
    bm25_index: BM25Index = Depends(get_bm25_index),
) -> IndexStats:
    """How much content is currently indexed."""
    documents = store.list_documents()
    return IndexStats(
        documents=len(documents),
        chunks=sum(d.chunk_count for d in documents),
        vectors=vector_store.count(),
        bm25_chunks=bm25_index.count(),
    )