"""Tests for result fusion."""

import pytest

from app.retrieval.fusion import fuse, reciprocal_rank_fusion, weighted_score_fusion
from app.vectorstore.base import VectorHit


def hit(chunk_id: str, score: float, retriever: str = "vector") -> VectorHit:
    return VectorHit(
        chunk_id=chunk_id,
        document_id="doc1",
        document_name="doc1.pdf",
        text=f"text of {chunk_id}",
        page_number=1,
        score=score,
        retriever=retriever,
    )


def test_chunk_found_by_both_retrievers_ranks_first():
    """Agreement between methods is the strongest signal."""
    results = {
        "vector": [hit("a", 0.9), hit("b", 0.8)],
        "bm25": [hit("c", 4.6, "bm25"), hit("a", 3.0, "bm25")],
    }
    fused = reciprocal_rank_fusion(results)
    assert fused[0].chunk_id == "a"
    assert fused[0].found_by_multiple


def test_duplicates_appear_only_once():
    results = {
        "vector": [hit("a", 0.9), hit("b", 0.5)],
        "bm25": [hit("a", 4.0, "bm25"), hit("b", 1.0, "bm25")],
    }
    fused = reciprocal_rank_fusion(results)
    assert len(fused) == 2
    assert sorted(h.chunk_id for h in fused) == ["a", "b"]


def test_rrf_ignores_score_scale():
    """BM25's huge scores must not swamp the vector list."""
    results = {
        "vector": [hit("a", 0.3)],
        "bm25": [hit("b", 900.0, "bm25")],
    }
    fused = reciprocal_rank_fusion(results)
    # Both are rank 1 in their own list, so they tie
    assert fused[0].score == pytest.approx(fused[1].score)


def test_weighted_fusion_can_favour_one_retriever():
    results = {
        "vector": [hit("a", 0.9), hit("b", 0.1)],
        "bm25": [hit("c", 5.0, "bm25"), hit("d", 1.0, "bm25")],
    }
    vector_heavy = weighted_score_fusion(results, {"vector": 3.0, "bm25": 0.5})
    bm25_heavy = weighted_score_fusion(results, {"vector": 0.5, "bm25": 3.0})
    assert vector_heavy[0].chunk_id == "a"
    assert bm25_heavy[0].chunk_id == "c"


def test_provenance_is_kept_for_transparency():
    results = {
        "vector": [hit("a", 0.42)],
        "bm25": [hit("a", 4.61, "bm25")],
    }
    fused = reciprocal_rank_fusion(results)[0]
    assert set(fused.retrievers) == {"vector", "bm25"}
    assert fused.original_scores["vector"] == 0.42
    assert fused.original_scores["bm25"] == 4.61
    assert fused.ranks == {"vector": 1, "bm25": 1}


def test_results_are_sorted_by_score():
    results = {"vector": [hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]}
    scores = [h.score for h in reciprocal_rank_fusion(results)]
    assert scores == sorted(scores, reverse=True)


def test_empty_and_partial_inputs_are_safe():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"vector": [], "bm25": []}) == []
    only_bm25 = reciprocal_rank_fusion({"vector": [], "bm25": [hit("a", 1.0, "bm25")]})
    assert len(only_bm25) == 1


def test_metadata_survives_fusion():
    fused = reciprocal_rank_fusion({"vector": [hit("a", 0.5)]})[0]
    assert fused.document_name == "doc1.pdf"
    assert fused.page_number == 1
    assert fused.text == "text of a"


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        fuse({"vector": []}, method="magic")


def test_fuse_dispatches_to_both_methods():
    results = {"vector": [hit("a", 0.9)], "bm25": [hit("b", 2.0, "bm25")]}
    assert len(fuse(results, "rrf")) == 2
    assert len(fuse(results, "weighted")) == 2