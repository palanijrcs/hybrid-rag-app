"""Phase 17 tests: answer formatting and the Streamlit page (fake backend, no server).

Run from the frontend folder:  python -m pytest tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))

from components.formatting import (  # noqa: E402
    link_citations,
    relevance_label,
    retrieval_method,
    status_banners,
)

SOURCES = [
    {"ref": 1, "label": "PM-KMY - Salient Features.pdf, page 1", "document_name": "PM-KMY.pdf",
     "page_number": 1, "chunk_id": "c1", "retrievers": ["vector", "bm25"],
     "rerank_score": 0.9993, "text": "• separate pension of Rs.3000/ when they reach the age of 60 years."},
]
ANSWER = {
    "question": "How much pension will farmers get?",
    "answer": "Farmers will receive a fixed pension of Rs.3,000/- when they reach the age of 60 years [1].",
    "grounded": True, "insufficient_evidence": False, "sources": SOURCES,
    "retrieval": {"vector": 10, "bm25": 10, "knowledge_graph": 0, "fused": 15, "reranked": 3},
    "model": "gpt-4o-mini", "timings_ms": {"retrieval": 700, "generation": 3200, "total": 4000},
    "notes": ["Corrected citations in 1 sentence(s)."], "coverage": "full",
    "input_guardrail": {"allowed": True, "category": "ok"},
    "output_guardrail": {"enabled": True, "passed": True, "verifier_used": True,
                         "regenerated": False, "removed_sentences": 0, "corrected_citations": 1,
                         "issues": ["... [3]. -> numbers ['60'] are not in the cited source; "
                                    "citation changed to [1]"]},
}


# ---------------------------------------------------------------- formatting
def test_citations_become_links_with_page_tooltip():
    md = link_citations("Rs.3,000/- [1]. Unknown [7].", SOURCES)
    assert '[\\[1\\]](#source-1 "PM-KMY - Salient Features.pdf, page 1")' in md
    assert "[7]" in md and "#source-7" not in md


def test_dollar_signs_are_not_rendered_as_latex():
    assert "\\$5" in link_citations("It costs $5 and $6 [1].", SOURCES)


def test_retrieval_method_summary():
    assert retrieval_method({"vector": 10, "bm25": 10, "knowledge_graph": 0}) == "Hybrid (Vector + BM25)"
    assert retrieval_method({"vector": 0, "bm25": 3}) == "BM25"
    assert retrieval_method({}) == "None"


def test_relevance_labels():
    assert relevance_label(0.9993) == "very high (1.00)"
    assert relevance_label(0.45) == "medium (0.45)"
    assert relevance_label(None) == "not re-ranked"


def test_banners_for_grounded_corrected_answer():
    kinds = [(b.kind, b.message) for b in status_banners(ANSWER)]
    assert ("info", "Citations were corrected in 1 sentence(s).") in kinds
    assert ("success", "Grounded in 1 source(s) and verified.") in kinds


def test_banners_for_insufficient_evidence():
    banners = status_banners({"insufficient_evidence": True, "grounded": True, "sources": []})
    assert [b.kind for b in banners] == ["warning"]


def test_banners_for_mentioned_only():
    resp = {**ANSWER, "coverage": "mentioned_only", "question_subject": "PM-KISAN scheme",
            "sources_subject": "PM-KMY", "output_guardrail": {"verifier_used": True}}
    msgs = [b.message for b in status_banners(resp)]
    assert any("only mention **PM-KISAN scheme**" in m for m in msgs)


def test_banners_for_blocked_question():
    resp = {"answer": "I can only answer...", "sources": [],
            "input_guardrail": {"allowed": False, "category": "prompt_injection"}}
    banners = status_banners(resp)
    assert banners[0].kind == "error" and "prompt injection" in banners[0].message
    greeting = {"input_guardrail": {"allowed": False, "category": "chit_chat"}}
    assert status_banners(greeting) == []


def test_banners_for_removed_sentences_hide_success():
    resp = {**ANSWER, "output_guardrail": {"removed_sentences": 1, "verifier_used": True}}
    kinds = [b.kind for b in status_banners(resp)]
    assert "warning" in kinds and "success" not in kinds


# ---------------------------------------------------------------- whole page (Streamlit AppTest)
class FakeResponse:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status
        self.content = b"x"
        self.text = str(data)

    def json(self):
        return self._data


@pytest.fixture
def fake_backend(monkeypatch):
    import requests

    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs.get("json")))
        if url.endswith("/documents"):
            return FakeResponse([{"document_id": "d1", "filename": "PM-KMY.pdf",
                                  "file_type": "pdf", "uploaded_at": "", "chunk_count": 5,
                                  "page_count": 5}])
        if url.endswith("/graph"):
            return FakeResponse({"kg_status": "done"})
        if url.endswith("/query"):
            return FakeResponse(ANSWER)
        return FakeResponse({})

    monkeypatch.setattr(requests, "request", fake_request)
    return calls


def test_page_answers_question_with_sources(fake_backend):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert any("graph ready" in c.value for c in at.sidebar.caption)

    at.chat_input[0].set_value("How much pension will farmers get?").run()
    assert not at.exception
    assert ("POST", "http://localhost:8000/query",
            {"question": "How much pension will farmers get?"}) in fake_backend
    markdown = " ".join(m.value for m in at.markdown)
    assert "#source-1" in markdown                       # clickable citation
    assert "PM-KMY - Salient Features.pdf, page 1" in markdown
    assert any("verified" in s.value for s in at.success)
    assert any("corrected" in i.value for i in at.info)


def test_mentioned_only_banner_not_repeated_when_answer_has_note():
    resp = {**ANSWER, "coverage": "mentioned_only", "question_subject": "PM-KISAN",
            "sources_subject": "PM-KMY",
            "answer": "Note: the uploaded documents are about PM-KMY. They mention PM-KISAN "
                      "only in passing.\n\nContributions can be paid ... [1]."}
    assert not any("only mention" in b.message for b in status_banners(resp))


def test_public_mode_hides_upload_delete_and_settings(fake_backend, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PUBLIC_MODE", "true")
    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert len(at.text_input) == 0                         # no Backend URL box
    assert not any(b.label == "Delete" for b in at.button)
    assert not any("Upload" in s.value for s in at.sidebar.subheader)
    assert any("PM-KMY.pdf" in m.value for m in at.sidebar.markdown)   # documents still listed

    at.chat_input[0].set_value("How much pension will farmers get?").run()
    assert ("POST", "http://localhost:8000/query",
            {"question": "How much pension will farmers get?"}) in fake_backend
