"""Tests for the FAISS vector store."""

import pytest

from app.vectorstore.embeddings import get_embedding_model
from app.vectorstore.faiss_store import FaissVectorStore


def make_chunk(chunk_id: str, document_id: str, text: str, page: int | None = 1):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "document_name": f"{document_id}.pdf",
        "page_number": page,
        "text": text,
    }


CHUNKS = [
    make_chunk("c0", "doc1", "A fixed pension of Rs.3,000 is paid to small farmers."),
    make_chunk("c1", "doc1", "The scheme is voluntary and contribution based.", 2),
    make_chunk("c2", "doc2", "The petrol bunk opens at six in the morning."),
]


@pytest.fixture
def store(tmp_path):
    return FaissVectorStore(tmp_path / "faiss", get_embedding_model())


def test_add_and_count(store):
    assert store.add_chunks(CHUNKS) == 3
    assert store.count() == 3


def test_search_finds_the_relevant_chunk(store):
    store.add_chunks(CHUNKS)
    hits = store.search("How much pension do farmers get?", top_k=2)
    assert hits[0].chunk_id == "c0"
    assert hits[0].score > 0


def test_hits_carry_citation_metadata(store):
    store.add_chunks(CHUNKS)
    hit = store.search("voluntary contribution scheme", top_k=1)[0]
    assert hit.document_id == "doc1"
    assert hit.document_name == "doc1.pdf"
    assert hit.page_number == 2
    assert hit.retriever == "vector"


def test_search_on_empty_store_returns_nothing(store):
    assert store.search("anything", top_k=5) == []


def test_top_k_larger_than_store_is_safe(store):
    store.add_chunks(CHUNKS)
    assert len(store.search("pension", top_k=50)) == 3


def test_delete_removes_only_that_document(store):
    store.add_chunks(CHUNKS)
    assert store.delete_document("doc1") == 2
    assert store.count() == 1
    assert all(h.document_id == "doc2" for h in store.search("anything", top_k=5))


def test_deleting_unknown_document_is_harmless(store):
    store.add_chunks(CHUNKS)
    assert store.delete_document("nope") == 0
    assert store.count() == 3


def test_index_survives_a_restart(tmp_path):
    model = get_embedding_model()
    first = FaissVectorStore(tmp_path / "faiss", model)
    first.add_chunks(CHUNKS)

    reopened = FaissVectorStore(tmp_path / "faiss", model)
    assert reopened.count() == 3
    assert reopened.search("pension for farmers", top_k=1)[0].chunk_id == "c0"


def test_ids_do_not_clash_after_delete_and_readd(tmp_path):
    model = get_embedding_model()
    store = FaissVectorStore(tmp_path / "faiss", model)
    store.add_chunks(CHUNKS)
    store.delete_document("doc1")
    store.add_chunks([make_chunk("c9", "doc3", "A new chunk about cooperatives.")])

    hits = store.search("cooperatives", top_k=5)
    assert {h.chunk_id for h in hits} == {"c2", "c9"}