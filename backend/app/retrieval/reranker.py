"""Phase 12: re-rank fused candidates with a cross-encoder.

Retrievers score the question and each chunk separately (vectors, keywords, graph
links). A cross-encoder reads the question and the chunk *together* and scores how
well the chunk answers it. That is slower, so it runs only on the fused top-N.

Scores are passed through a sigmoid to get 0..1 relevance. Chunks below
MIN_RELEVANCE_SCORE are dropped: weak evidence is removed before the LLM sees it,
and if nothing survives, later phases answer "not enough information".
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Any, Protocol

from app.retrieval.fusion import FusedHit

logger = logging.getLogger(__name__)


class ScoringModel(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> Any: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[FusedHit], top_k: int) -> list[FusedHit]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class CrossEncoderReranker:
    """Wraps a sentence-transformers CrossEncoder (loaded on first use)."""

    def __init__(
        self,
        model_name: str,
        min_score: float = 0.0,
        max_chars: int = 2000,
        batch_size: int = 16,
        model: ScoringModel | None = None,
    ) -> None:
        self.model_name = model_name
        self.min_score = min_score
        self.max_chars = max_chars
        self.batch_size = batch_size
        self._model = model

    @property
    def model(self) -> ScoringModel:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading re-ranker model %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, hits: list[FusedHit], top_k: int) -> list[FusedHit]:
        if not hits or not query.strip():
            return []
        pairs = [(query, hit.text[: self.max_chars]) for hit in hits]
        raw = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scored = [
            replace(hit, rerank_score=round(_sigmoid(float(s)), 4))
            for hit, s in zip(hits, list(raw))
        ]
        scored.sort(key=lambda h: h.rerank_score, reverse=True)
        kept = [h for h in scored if h.rerank_score >= self.min_score]
        logger.info(
            "Re-ranked %d candidates: %d above %.2f, returning %d.",
            len(hits), len(kept), self.min_score, min(len(kept), top_k),
        )
        return kept[:top_k]
