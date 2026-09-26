"""Runs every enabled retriever and fuses their results into one list."""

import logging
from typing import Protocol

from app.retrieval.fusion import FusedHit, fuse
from app.retrieval.reranker import Reranker
from app.vectorstore.base import VectorHit

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    """Anything that can return ranked chunks for a query."""

    def search(self, query: str, top_k: int) -> list[VectorHit]: ...


class HybridRetriever:
    """Combines vector, keyword and (later) graph retrieval."""

    def __init__(
        self,
        vector_retriever: Retriever | None = None,
        bm25_retriever: Retriever | None = None,
        graph_retriever: Retriever | None = None,
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        kg_top_k: int = 10,
        fusion_method: str = "rrf",
        reranker: Reranker | None = None,
        rerank_candidates: int = 30,
        rerank_top_k: int = 5,
    ) -> None:
        self.retrievers: dict[str, tuple[Retriever, int]] = {}
        if vector_retriever is not None:
            self.retrievers["vector"] = (vector_retriever, vector_top_k)
        if bm25_retriever is not None:
            self.retrievers["bm25"] = (bm25_retriever, bm25_top_k)
        if graph_retriever is not None:
            self.retrievers["knowledge_graph"] = (graph_retriever, kg_top_k)

        self.fusion_method = fusion_method
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.rerank_top_k = rerank_top_k

    @property
    def enabled(self) -> list[str]:
        return sorted(self.retrievers)

    def retrieve_each(self, query: str) -> dict[str, list[VectorHit]]:
        """Run every enabled retriever separately, tolerating failures."""
        results: dict[str, list[VectorHit]] = {}

        for name, (retriever, top_k) in self.retrievers.items():
            try:
                hits = retriever.search(query, top_k=top_k)
            except Exception:
                # One retriever failing must not take down the whole query
                logger.exception("Retriever '%s' failed; continuing without it.", name)
                hits = []
            results[name] = hits
            logger.info("Retriever '%s' returned %d hits.", name, len(hits))

        return results

    def retrieve(self, query: str, top_k: int | None = None) -> list[FusedHit]:
        """Retrieve from every source and return one fused, ranked list."""
        if not query.strip():
            return []
        if not self.retrievers:
            logger.warning("No retrievers are enabled.")
            return []

        per_retriever = self.retrieve_each(query)
        fused = fuse(per_retriever, method=self.fusion_method)
        if self.reranker is not None:
            return self.rerank(query, fused, top_k)
        return fused[:top_k] if top_k else fused

    def rerank(
        self, query: str, fused: list[FusedHit], top_k: int | None = None
    ) -> list[FusedHit]:
        """Re-score the best fused candidates. Falls back to fusion order on failure."""
        limit = top_k or self.rerank_top_k
        if self.reranker is None:
            return fused[:limit]
        candidates = fused[: self.rerank_candidates]
        try:
            return self.reranker.rerank(query, candidates, top_k=limit)
        except Exception:
            logger.exception("Re-ranker failed; using fusion order instead.")
            return fused[:limit]