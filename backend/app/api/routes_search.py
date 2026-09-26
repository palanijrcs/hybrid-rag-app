"""Search endpoints (a stepping stone towards full hybrid retrieval)."""

from app.core.dependencies import get_hybrid_retriever
from app.models.schemas import FusedResult, HybridSearchResponse
from app.retrieval.hybrid_retriever import HybridRetriever
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.config import get_settings
from app.core.dependencies import (
    get_bm25_index,
    get_context_builder,
    get_graph_retriever,
    get_document_store,
    get_vector_store,
)
from app.core.indexes import BM25Index
from app.ingestion.store import DocumentStore
from app.knowledge_graph.graph_retriever import GraphRetriever
from app.retrieval.context_builder import ContextBuilder
from app.models.schemas import (
    ComparisonResponse,
    ContextResponse,
    GraphSearchResponse,
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
@router.get("/search/hybrid", response_model=HybridSearchResponse)
def hybrid_search(
    q: str = Query(..., min_length=1),
    top_k: int | None = Query(None, ge=1, le=50),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
) -> HybridSearchResponse:
    """Search every enabled retriever and return one fused, ranked list."""
    per_retriever = retriever.retrieve_each(q)
    from app.retrieval.fusion import fuse

    fused = fuse(per_retriever, method=retriever.fusion_method)
    if retriever.reranker is not None:
        limited = retriever.rerank(q, fused, top_k)
    else:
        limited = fused[:top_k] if top_k else fused

    return HybridSearchResponse(
        query=q,
        fusion_method=retriever.fusion_method,
        retrievers_used=retriever.enabled,
        counts={
            **{name: len(hits) for name, hits in per_retriever.items()},
            "fused": len(fused),
            "returned": len(limited),
        },
        results=[FusedResult(**hit.__dict__) for hit in limited],
    )


@router.get("/search/graph", response_model=GraphSearchResponse)
def graph_search(
    q: str = Query(..., min_length=1),
    top_k: int | None = Query(None, ge=1, le=50),
    retriever: GraphRetriever | None = Depends(get_graph_retriever),
) -> GraphSearchResponse:
    """Knowledge-graph retrieval only: matched entities, facts and source chunks."""
    if not q.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Query cannot be empty.")
    if retriever is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Neo4j is not configured.")
    try:
        result = retriever.retrieve(q, top_k=top_k or get_settings().kg_top_k)
    except Exception:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Graph retrieval failed.")
    return GraphSearchResponse(
        query=q,
        matched_entities=[m.__dict__ for m in result.matched_entities],
        evidence=[
            {**{k: v for k, v in e.__dict__.items() if k != "facts"},
             "facts": [f.__dict__ for f in e.facts]}
            for e in result.evidence
        ],
    )


@router.get("/search/context", response_model=ContextResponse)
def build_context(
    q: str = Query(..., min_length=1),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    graph: GraphRetriever | None = Depends(get_graph_retriever),
    builder: ContextBuilder = Depends(get_context_builder),
) -> ContextResponse:
    """Preview the exact evidence block the LLM will receive for a question."""
    if not q.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Query cannot be empty.")
    hits = retriever.retrieve(q)  # fused + re-ranked + thresholded
    graph_evidence = []
    if graph is not None and get_settings().enable_kg_retrieval and hits:
        try:
            graph_evidence = graph.retrieve(q, top_k=get_settings().kg_top_k).evidence
        except Exception:
            graph_evidence = []  # graph facts are optional extras
    ctx = builder.build(q, hits, graph_evidence)
    return ContextResponse(
        query=q,
        is_empty=ctx.is_empty,
        char_count=ctx.char_count,
        dropped_chunks=ctx.dropped_chunks,
        removed_duplicate_sentences=ctx.removed_duplicate_sentences,
        sources=[{**s.__dict__, "label": s.label} for s in ctx.sources],
        facts=[f.__dict__ for f in ctx.facts],
        context=ctx.text,
    )
