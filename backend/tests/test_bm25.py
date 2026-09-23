"""Tests for BM25 keyword retrieval."""

import pytest

from app.bm25.retriever import BM25Retriever, tokenize


def make_chunk(chunk_id: str, text: str, document_id: str = "doc1", page: int = 1):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "document_name": f"{document_id}.pdf",
        "page_number": page,
        "text": text,
    }


CHUNKS = [
    make_chunk("c0", "Under PM-KMY a fixed pension of Rs.3,000 is provided."),
    make_chunk("c1", "Farmers contribute between Rs.55 and Rs.200 per month."),
    make_chunk("c2", "The petrol bunk opens at six in the morning.", "doc2"),
]


@pytest.fixture
def retriever():
    bm25 = BM25Retriever()
    bm25.build(CHUNKS)
    return bm25


def test_tokenizer_keeps_compound_terms_whole():
    assert "pm-kmy" in tokenize("Under PM-KMY the scheme")
    # Reference numbers stay intact rather than shattering into digits
    assert "r.c.no.4395/2026" in tokenize("R.C.No.4395/2026 dated today")
    assert tokenize("Rs.3,000 per month") == ["rs.3", "000", "per", "month"]


def test_exact_term_is_found(retriever):
    hits = retriever.search("PM-KMY", top_k=5)
    assert hits[0].chunk_id == "c0"
    assert hits[0].retriever == "bm25"


def test_hits_carry_citation_metadata(retriever):
    hit = retriever.search("petrol bunk", top_k=1)[0]
    assert hit.document_id == "doc2"
    assert hit.page_number == 1
    assert hit.score > 0


def test_unmatched_query_returns_nothing(retriever):
    assert retriever.search("helicopter aviation turbine", top_k=5) == []


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("   ", top_k=5) == []


def test_empty_corpus_is_safe():
    bm25 = BM25Retriever()
    bm25.build([])
    assert bm25.search("anything", top_k=5) == []
    assert bm25.count() == 0


def test_rebuild_replaces_the_corpus(retriever):
    retriever.build([make_chunk("c9", "A new chunk about cooperative societies.")])
    assert retriever.count() == 1
    assert retriever.search("PM-KMY", top_k=5) == []
    assert retriever.search("cooperative", top_k=5)[0].chunk_id == "c9"


def test_bm25_beats_semantics_on_rare_exact_terms(retriever):
    """The whole reason BM25 exists alongside vectors."""
    hits = retriever.search("Rs.55", top_k=3)
    assert hits[0].chunk_id == "c1"