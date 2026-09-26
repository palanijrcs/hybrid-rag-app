"""Phase 12 tests: re-ranking and the unreadable-text filter."""
from __future__ import annotations

import pytest

from app.ingestion.quality import assess_text, split_by_quality
from app.retrieval.fusion import FusedHit
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.vectorstore.base import VectorHit

# Real text from your PM-KMY PDF (page 3), extracted with a broken Hindi font
GARBLED_HINDI = (
    "लघु एवं सीमांत किसान पति त-प् नी इस स िीम िो ्लग -्लग ्पनाने िे ललए पात्र होंगे तथा\n"
    "वे जु 60 विि िी आयु पूर िर लेंगे तो 3000 रु. प्रति तमाह ्लग-्लग पेंशन प्राप् त िरने िे\n"
    "हिदार होंगे।"
)
PROPER_HINDI = (
    "प्रधान मंत्री किसान मान-धन योजना देश के सभी छोटे और सीमांत किसानों को सामाजिक "
    "सुरक्षा प्रदान करने के लिए शुरू की गई है।"
)
ENGLISH = (
    "Under this scheme, a fixed pension of Rs.3,000/- will be provided to all eligible "
    "small and marginal farmers."
)


# ---------------------------------------------------------------- text quality
def test_broken_hindi_font_is_detected():
    q = assess_text(GARBLED_HINDI)
    assert q.garbled and q.reason == "broken_indic_font"


@pytest.mark.parametrize("text", [PROPER_HINDI, ENGLISH, ENGLISH + " " + PROPER_HINDI])
def test_good_text_passes(text):
    assert not assess_text(text).garbled


def test_unmapped_pdf_characters_detected():
    assert assess_text("Pension (cid:12)(cid:45)(cid:3) scheme (cid:9)").garbled
    assert assess_text("Pen��sion sch�eme").garbled


def test_split_by_quality_works_on_dicts():
    good, bad = split_by_quality([{"text": ENGLISH}, {"text": GARBLED_HINDI}])
    assert [c["text"] for c in good] == [ENGLISH]
    assert bad[0][1].reason == "broken_indic_font"


def test_pipeline_skips_garbled_chunks(tmp_path):
    from app.ingestion.pipeline import IngestionPipeline
    from app.ingestion.store import DocumentStore

    store = DocumentStore(tmp_path / "u", tmp_path / "p" / "r.json")
    pipeline = IngestionPipeline(store, None, None, chunk_size=300, chunk_overlap=20)
    content = (ENGLISH + " " + ENGLISH + "\n\n" + GARBLED_HINDI).encode()
    record = pipeline.ingest("mixed.txt", content)
    texts = [c["text"] for c in store.get_chunks(record.document_id)]
    assert texts and all(not assess_text(t).garbled for t in texts)
    assert record.chunk_count == len(texts)  # registry counts only indexed chunks


def test_pipeline_rejects_fully_garbled_file(tmp_path):
    from app.ingestion.loaders import DocumentLoadError
    from app.ingestion.pipeline import IngestionPipeline
    from app.ingestion.store import DocumentStore

    store = DocumentStore(tmp_path / "u", tmp_path / "p" / "r.json")
    pipeline = IngestionPipeline(store, None, None, chunk_size=400, chunk_overlap=40)
    with pytest.raises(DocumentLoadError, match="OCR"):
        pipeline.ingest("broken.txt", GARBLED_HINDI.encode())
    assert store.list_documents() == []


# ---------------------------------------------------------------- re-ranker
def hit(cid: str, text: str, score: float = 0.0) -> FusedHit:
    return FusedHit(chunk_id=cid, document_id="d", document_name="d.pdf", text=text,
                    page_number=1, score=score, retrievers=["vector"])


class KeywordModel:
    """Fake cross-encoder: logit +4 if the chunk contains 'pension', else -4."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs, **kwargs):
        self.calls += 1
        return [4.0 if "pension" in text.lower() else -4.0 for _, text in pairs]


def test_reranker_reorders_and_scores():
    hits = [hit("a", "Enrollment is free at CSC centres", 0.9),
            hit("b", "A pension of Rs.3000 per month", 0.5)]
    out = CrossEncoderReranker("fake", model=KeywordModel()).rerank("pension amount?", hits, 5)
    assert [h.chunk_id for h in out] == ["b", "a"]
    assert out[0].rerank_score > 0.95 and out[1].rerank_score < 0.05
    assert out[0].retrievers == ["vector"]  # fusion info preserved


def test_reranker_drops_weak_evidence():
    hits = [hit("a", "Enrollment is free"), hit("b", "pension details")]
    out = CrossEncoderReranker("fake", min_score=0.5, model=KeywordModel()).rerank("q", hits, 5)
    assert [h.chunk_id for h in out] == ["b"]


def test_reranker_can_return_nothing():
    out = CrossEncoderReranker("fake", min_score=0.5, model=KeywordModel()).rerank(
        "q", [hit("a", "unrelated")], 5)
    assert out == []


def test_reranker_top_k_and_empty_input():
    model = KeywordModel()
    rr = CrossEncoderReranker("fake", model=model)
    assert len(rr.rerank("q", [hit(str(i), "pension") for i in range(10)], 3)) == 3
    assert rr.rerank("q", [], 3) == [] and model.calls == 1


class ListRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k):
        return self.hits[:top_k]


def _vhit(cid, text):
    return VectorHit(chunk_id=cid, document_id="d", document_name="d.pdf", text=text,
                     page_number=1, score=1.0)


def test_hybrid_retriever_reranks_candidates():
    vector = ListRetriever([_vhit("a", "CSC enrollment"), _vhit("b", "pension of Rs.3000")])
    rr = CrossEncoderReranker("fake", model=KeywordModel())
    hybrid = HybridRetriever(vector_retriever=vector, reranker=rr, rerank_top_k=1)
    out = hybrid.retrieve("How much pension?")
    assert [h.chunk_id for h in out] == ["b"] and out[0].rerank_score is not None


def test_hybrid_retriever_falls_back_when_reranker_fails():
    class Broken:
        def rerank(self, query, hits, top_k):
            raise RuntimeError("model download failed")

    vector = ListRetriever([_vhit("a", "x"), _vhit("b", "y")])
    out = HybridRetriever(vector_retriever=vector, reranker=Broken(), rerank_top_k=2).retrieve("q")
    assert [h.chunk_id for h in out] == ["a", "b"] and out[0].rerank_score is None


# ---------------------------------------------------------------- real model
def test_real_cross_encoder_ranks_answer_first():
    """Uses RERANKER_MODEL from .env; skipped if the model can't be loaded."""
    from app.core.config import get_settings

    rr = CrossEncoderReranker(get_settings().reranker_model)
    try:
        rr.model
    except Exception as exc:  # no internet / model not cached
        pytest.skip(f"re-ranker model unavailable: {type(exc).__name__}")
    hits = [
        hit("csc", "The eligible farmers desirous of joining the scheme will visit nearest "
                   "Common Service Centre (CSC) along with their Aadhaar number."),
        hit("pension", ENGLISH),
        hit("hindi", GARBLED_HINDI),
    ]
    out = rr.rerank("How much pension will farmers get per month?", hits, 3)
    assert out[0].chunk_id == "pension"
    assert out[0].rerank_score > out[-1].rerank_score
