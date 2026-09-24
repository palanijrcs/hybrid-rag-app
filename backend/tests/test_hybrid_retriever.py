"""Tests for the hybrid retriever."""

from app.retrieval.hybrid_retriever import HybridRetriever
from app.vectorstore.base import VectorHit


class FakeRetriever:
    def __init__(self, hits: list[VectorHit]) -> None:
        self.hits = hits
        self.last_top_k: int | None = None

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        self.last_top_k = top_k
        return self.hits[:top_k]


class BrokenRetriever:
    def search(self, query: str, top_k: int) -> list[VectorHit]:
        raise RuntimeError("connection lost")


def hit(chunk_id: str, score: float, retriever: str) -> VectorHit:
    return VectorHit(
        chunk_id=chunk_id,
        document_id="doc1",
        document_name="doc1.pdf",
        text=f"text of {chunk_id}",
        page_number=1,
        score=score,
        retriever=retriever,
    )


def test_runs_every_enabled_retriever():
    retriever = HybridRetriever(
        vector_retriever=FakeRetriever([hit("a", 0.9, "vector")]),
        bm25_retriever=FakeRetriever([hit("b", 3.0, "bm25")]),
    )
    assert retriever.enabled == ["bm25", "vector"]
    assert len(retriever.retrieve("query")) == 2


def test_disabled_retriever_is_skipped():
    retriever = HybridRetriever(
        vector_retriever=FakeRetriever([hit("a", 0.9, "vector")]),
        bm25_retriever=None,
    )
    assert retriever.enabled == ["vector"]
    assert retriever.retrieve_each("query").keys() == {"vector"}


def test_agreement_between_retrievers_wins():
    shared = hit("shared", 0.5, "vector")
    retriever = HybridRetriever(
        vector_retriever=FakeRetriever([hit("a", 0.9, "vector"), shared]),
        bm25_retriever=FakeRetriever([hit("b", 9.0, "bm25"), shared]),
    )
    top = retriever.retrieve("query")[0]
    assert top.chunk_id == "shared"
    assert set(top.retrievers) == {"vector", "bm25"}


def test_a_failing_retriever_does_not_break_the_query():
    retriever = HybridRetriever(
        vector_retriever=FakeRetriever([hit("a", 0.9, "vector")]),
        bm25_retriever=BrokenRetriever(),
    )
    results = retriever.retrieve("query")
    assert len(results) == 1
    assert results[0].chunk_id == "a"


def test_each_retriever_gets_its_own_top_k():
    vector = FakeRetriever([hit(f"v{i}", 0.5, "vector") for i in range(20)])
    bm25 = FakeRetriever([hit(f"b{i}", 1.0, "bm25") for i in range(20)])
    HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
        vector_top_k=3,
        bm25_top_k=7,
    ).retrieve("query")
    assert vector.last_top_k == 3
    assert bm25.last_top_k == 7


def test_top_k_limits_the_fused_list():
    retriever = HybridRetriever(
        vector_retriever=FakeRetriever([hit(f"v{i}", 0.5, "vector") for i in range(10)])
    )
    assert len(retriever.retrieve("query", top_k=4)) == 4


def test_no_retrievers_or_empty_query_is_safe():
    assert HybridRetriever().retrieve("query") == []
    with_one = HybridRetriever(vector_retriever=FakeRetriever([hit("a", 1.0, "vector")]))
    assert with_one.retrieve("   ") == []