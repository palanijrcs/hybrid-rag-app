"""Combine results from several retrievers into one ranked list."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from app.vectorstore.base import VectorHit

logger = logging.getLogger(__name__)

RRF_K = 60  # standard constant from the Reciprocal Rank Fusion paper


@dataclass
class FusedHit:
    """A chunk after fusion, remembering which retrievers found it."""

    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_number: int | None
    score: float
    retrievers: list[str] = field(default_factory=list)
    original_scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    rerank_score: float | None = None  # set by the re-ranker (0..1)

    @property
    def found_by_multiple(self) -> bool:
        return len(self.retrievers) > 1


def _normalise(scores: list[float]) -> list[float]:
    """Scale a list of scores to 0-1 so different retrievers are comparable."""
    if not scores:
        return []
    lowest, highest = min(scores), max(scores)
    if highest == lowest:
        return [1.0] * len(scores)
    span = highest - lowest
    return [(score - lowest) / span for score in scores]


def reciprocal_rank_fusion(
    results: dict[str, list[VectorHit]], k: int = RRF_K
) -> list[FusedHit]:
    """Fuse by position, not score. Robust when scales differ wildly."""
    contributions: dict[str, float] = defaultdict(float)
    seen: dict[str, VectorHit] = {}
    retrievers: dict[str, list[str]] = defaultdict(list)
    originals: dict[str, dict[str, float]] = defaultdict(dict)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for retriever_name, hits in results.items():
        for position, hit in enumerate(hits, start=1):
            contributions[hit.chunk_id] += 1.0 / (k + position)
            seen.setdefault(hit.chunk_id, hit)
            retrievers[hit.chunk_id].append(retriever_name)
            originals[hit.chunk_id][retriever_name] = hit.score
            ranks[hit.chunk_id][retriever_name] = position

    return _build(contributions, seen, retrievers, originals, ranks)


def weighted_score_fusion(
    results: dict[str, list[VectorHit]], weights: dict[str, float] | None = None
) -> list[FusedHit]:
    """Fuse by normalised score, with an optional weight per retriever."""
    weights = weights or {}
    contributions: dict[str, float] = defaultdict(float)
    seen: dict[str, VectorHit] = {}
    retrievers: dict[str, list[str]] = defaultdict(list)
    originals: dict[str, dict[str, float]] = defaultdict(dict)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for retriever_name, hits in results.items():
        weight = weights.get(retriever_name, 1.0)
        normalised = _normalise([hit.score for hit in hits])
        for position, (hit, score) in enumerate(zip(hits, normalised), start=1):
            contributions[hit.chunk_id] += weight * score
            seen.setdefault(hit.chunk_id, hit)
            retrievers[hit.chunk_id].append(retriever_name)
            originals[hit.chunk_id][retriever_name] = hit.score
            ranks[hit.chunk_id][retriever_name] = position

    return _build(contributions, seen, retrievers, originals, ranks)


def _build(
    contributions: dict[str, float],
    seen: dict[str, VectorHit],
    retrievers: dict[str, list[str]],
    originals: dict[str, dict[str, float]],
    ranks: dict[str, dict[str, int]],
) -> list[FusedHit]:
    """Turn the accumulated scores into a sorted list of FusedHits."""
    fused = [
        FusedHit(
            chunk_id=chunk_id,
            document_id=seen[chunk_id].document_id,
            document_name=seen[chunk_id].document_name,
            text=seen[chunk_id].text,
            page_number=seen[chunk_id].page_number,
            score=score,
            retrievers=retrievers[chunk_id],
            original_scores=originals[chunk_id],
            ranks=ranks[chunk_id],
        )
        for chunk_id, score in contributions.items()
    ]
    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused


FUSION_METHODS = {
    "rrf": reciprocal_rank_fusion,
    "weighted": weighted_score_fusion,
}


def fuse(
    results: dict[str, list[VectorHit]], method: str = "rrf"
) -> list[FusedHit]:
    """Fuse results using the named method."""
    if method not in FUSION_METHODS:
        raise ValueError(
            f"Unknown fusion method '{method}'. "
            f"Available: {', '.join(sorted(FUSION_METHODS))}"
        )
    fused = FUSION_METHODS[method](results)
    logger.info(
        "Fused %d lists into %d unique chunks using %s.",
        len(results),
        len(fused),
        method,
    )
    return fused