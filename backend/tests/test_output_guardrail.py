"""Phase 16 tests: output guardrail. Cases come from real answers the system produced."""
from __future__ import annotations

import json

import pytest

from app.guardrails.output_guardrail import (
    OutputGuardrail,
    normalize_number,
    number_periods,
    numbers_in,
)
from app.llm.grounded_generation import GroundedQA
from app.llm.prompts import INSUFFICIENT_EVIDENCE_ANSWER
from app.retrieval.context_builder import ContextBuilder, ContextSource
from app.retrieval.hybrid_retriever import HybridRetriever
from app.vectorstore.base import VectorHit

# The three real sources from "How much pension will farmers get?"
S1 = ("• Farmers will have to contribute an amount between Rs.55 to Rs.200 per month in the "
      "Pension Fund till they reach the retirement date i.e. the age of 60 years.\n"
      "• Spouses of the Small and Marginal farmers are also eligible to join the scheme "
      "separately and they will also get separate pension of Rs.3000/ when they reach the "
      "age of 60 years.")
S2 = ("• If the farmer dies after the retirement date, the spouse will receive 50% of the "
      "pension i.e. Rs.1500 per month as Family Pension.\n"
      "• If the farmer is a beneficiary of the PM-KISAN Scheme , he/she may allow the "
      "contribution to be directly paid from the same bank account in which he / she "
      "receives the PM-Kisan benefit.")
S3 = ("• Pradhan Mantri Kisan Maan -Dhan Yojana has been started to provide social security "
      "to all landholding Small and Marginal Farmers in the country.\n"
      "• Under this scheme, a fixed pension of Rs.3,000/ - will be provided to all eligible "
      "small and marginal farmers.")


def src(ref, text, page=1):
    return ContextSource(ref=ref, chunk_id=f"c{ref}", document_id="d1",
                         document_name="PM-KMY.pdf", page_number=page,
                         retrievers=["vector"], rerank_score=0.99, text=text)


SOURCES = [src(1, S1), src(2, S2, 2), src(3, S3)]
guard = OutputGuardrail()  # deterministic only


# ---------------------------------------------------------------- number helpers
def test_numbers_normalised():
    assert numbers_in("Rs.3,000/- and Rs.3000/ [3]") == {"3000"}
    assert normalize_number("1,500.00") == "1500"
    assert number_periods("Rs.1500 per month") == {("1500", "per month")}
    assert number_periods("Rs.3,000 monthly") == {("3000", "per month")}


# ---------------------------------------------------------------- deterministic layer
def test_correct_answer_passes():
    check = guard.check("q", "Eligible farmers get a fixed pension of Rs.3,000/- [3].", SOURCES)
    assert check.passed and check.sentences[0].status == "supported"


def test_wrong_citation_is_corrected():
    # real answer: "age of 60 years" is in [1], not [3]
    check = guard.check("q", "Farmers will receive a fixed pension of Rs.3,000/- when they "
                             "reach the age of 60 years [3].", SOURCES)
    s = check.sentences[0]
    assert check.passed and s.status == "corrected"
    assert s.fixed_text.endswith("[1].")
    assert check.cleaned_answer.endswith("age of 60 years [1].")


def test_invented_period_is_caught():
    # real answer from Phase 14: "per month" is not stated for Rs.3,000 in the sources
    check = guard.check("q", "Farmers will receive a fixed pension of Rs.3,000 per month [3].",
                        SOURCES)
    assert not check.passed
    assert "3000 per month" in check.unsupported[0].issues[0]


def test_real_period_is_accepted():
    check = guard.check("q", "The spouse receives Rs.1500 per month as Family Pension [2].",
                        SOURCES)
    assert check.passed


def test_invented_number_is_removed():
    check = guard.check("q", "Farmers get Rs.3,000/- [3]. The pension rises to Rs.5,000 "
                             "after 70 years [3].", SOURCES)
    assert len(check.unsupported) == 1 and "5000" in check.unsupported[0].issues[0]
    assert check.cleaned_answer == "Farmers get Rs.3,000/- [3]."


def test_uncited_number_is_unsupported_but_meta_sentence_is_kept():
    check = guard.check("q", "Farmers get Rs.3,000/- [3]. It is about 36000 a year. "
                             "The documents do not describe other schemes.", SOURCES)
    statuses = [s.status for s in check.sentences]
    assert statuses == ["supported", "unsupported", "meta"]


# ---------------------------------------------------------------- LLM verifier layer
class FakeVerifier:
    model = "fake-verifier"

    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = 0

    def complete_json(self, system, user):
        self.calls += 1
        if isinstance(self.verdicts, Exception):
            raise self.verdicts
        return json.dumps({"sentences": self.verdicts})


PM_KISAN_BAD = (
    "The PM-KISAN scheme provides financial assistance to small and marginal farmers [3]. "
    "Contributions can be paid from the bank account used for PM-Kisan benefits [2]."
)


def test_verifier_removes_wrong_subject_sentence():
    verifier = FakeVerifier([
        {"id": 1, "verdict": "unsupported", "sources": [],
         "issue": "describes PM-KMY, not PM-KISAN; 'financial assistance' not stated"},
        {"id": 2, "verdict": "supported", "sources": [2], "issue": ""},
    ])
    check = OutputGuardrail(verifier).check("Tell me about PM-KISAN", PM_KISAN_BAD, SOURCES)
    assert check.verifier_used and len(check.unsupported) == 1
    assert "financial assistance" not in check.cleaned_answer
    assert check.cleaned_answer.startswith("Contributions can be paid")


def test_verifier_failure_falls_back_to_deterministic():
    check = OutputGuardrail(FakeVerifier(RuntimeError("timeout"))).check(
        "q", "Farmers get Rs.3,000/- [3].", SOURCES)
    assert check.passed and not check.verifier_used and "timeout" in check.verifier_error


def test_verifier_not_called_when_everything_already_failed():
    verifier = FakeVerifier([])
    check = OutputGuardrail(verifier).check("q", "Farmers get Rs.9,999 [3].", SOURCES)
    assert not check.passed and verifier.calls == 0


def test_mentioned_only_notice_is_preserved():
    answer = ("Note: the uploaded documents are about PM-KMY. They mention PM-KISAN only in "
              "passing and do not describe it in detail.\n\nContributions can be paid from the "
              "PM-Kisan bank account [2].")
    check = guard.check("q", answer, SOURCES)
    assert check.cleaned_answer.startswith("Note: the uploaded documents are about PM-KMY.")
    assert check.passed


# ---------------------------------------------------------------- full pipeline
def _vhit(cid, text, page=1):
    return VectorHit(chunk_id=cid, document_id="d1", document_name="PM-KMY.pdf",
                     text=text, page_number=page, score=1.0)


class ListRetriever:
    def search(self, query, top_k):
        return [_vhit("c1", S1), _vhit("c2", S2, 2), _vhit("c3", S3)]


class ScriptedLLM:
    """Returns answers in order; any call carrying the verifier prompt gets `verdicts`."""
    model = "fake"

    def __init__(self, answers, verdicts=None):
        self.answers, self.verdicts = list(answers), verdicts
        self.generation_prompts: list[str] = []

    def complete_json(self, system, user):
        if "strict fact-checker" in system:
            return json.dumps({"sentences": self.verdicts or []})
        self.generation_prompts.append(user)
        return json.dumps(self.answers.pop(0))


def pipeline(llm, verifier=None, max_regenerations=1):
    retriever = HybridRetriever(vector_retriever=ListRetriever(), rerank_top_k=5)
    return GroundedQA(retriever, ContextBuilder(), llm,
                      output_guardrail=OutputGuardrail(verifier),
                      max_regenerations=max_regenerations)


def test_bad_answer_is_regenerated_with_feedback():
    llm = ScriptedLLM([
        {"answer": "Farmers get Rs.3,000 per month [3].", "coverage": "full"},
        {"answer": "Farmers get a fixed pension of Rs.3,000/- [3].", "coverage": "full"},
    ])
    result = pipeline(llm).answer("How much pension will farmers get?")
    assert result.answer == "Farmers get a fixed pension of Rs.3,000/- [3]."
    assert result.output_guardrail["regenerated"] is True
    assert result.output_guardrail["passed"] is True
    assert "3000 per month" in llm.generation_prompts[1]  # feedback reached the model


def test_still_bad_after_regeneration_removes_unsupported_part():
    bad = {"answer": "Farmers get Rs.3,000/- [3]. It is paid from age 70 [1].",
           "coverage": "full"}
    result = pipeline(ScriptedLLM([bad, bad])).answer("How much pension?")
    assert result.answer == "Farmers get Rs.3,000/- [3]."
    assert result.output_guardrail["removed_sentences"] == 1
    assert [s.ref for s in result.sources] == [3]


def test_nothing_verifiable_means_insufficient():
    bad = {"answer": "Farmers get Rs.9,000 [3].", "coverage": "full"}
    result = pipeline(ScriptedLLM([bad, bad])).answer("How much pension?")
    assert result.insufficient_evidence and result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert "could be verified" in result.notes[0]


def test_citation_fix_updates_returned_sources():
    llm = ScriptedLLM([{"answer": "Farmers receive Rs.3,000/- when they reach the age of "
                                  "60 years [3].", "coverage": "full"}])
    result = pipeline(llm).answer("How much pension?")
    assert result.answer.endswith("[1].") and [s.ref for s in result.sources] == [1]
    assert result.output_guardrail["corrected_citations"] == 1


def test_guardrail_can_be_disabled():
    retriever = HybridRetriever(vector_retriever=ListRetriever(), rerank_top_k=5)
    llm = ScriptedLLM([{"answer": "Farmers get Rs.3,000 per month [3].", "coverage": "full"}])
    result = GroundedQA(retriever, ContextBuilder(), llm).answer("q")
    assert result.answer == "Farmers get Rs.3,000 per month [3]."  # unchecked
    assert result.output_guardrail["enabled"] is False


def test_real_number_in_wrong_sense_needs_the_verifier():
    """'55' is in the sources (Rs.55), so the number check can't flag 'age 55' — the LLM
    verifier is the layer that catches a real number used with the wrong meaning."""
    answer = "Farmers get Rs.3,000/- [3]. The pension starts at age 55 [1]."
    assert guard.check("q", answer, SOURCES).passed  # deterministic alone misses it
    verifier = FakeVerifier([
        {"id": 1, "verdict": "supported", "sources": [3]},
        {"id": 2, "verdict": "unsupported", "sources": [],
         "issue": "55 is a contribution amount (Rs.55), not an age"},
    ])
    check = OutputGuardrail(verifier).check("q", answer, SOURCES)
    assert not check.passed and check.cleaned_answer == "Farmers get Rs.3,000/- [3]."


def test_verifier_cannot_undo_a_certain_citation_fix():
    """Real bug: the number check moved '60 years' to [1]; the verifier then said [3]."""
    answer = ("Farmers will receive a fixed pension of Rs.3,000/- when they reach the age of "
              "60 years [3].")
    verifier = FakeVerifier([{"id": 1, "verdict": "supported", "sources": [3], "issue": ""}])
    check = OutputGuardrail(verifier).check("q", answer, SOURCES)
    assert check.cleaned_answer.endswith("age of 60 years [1].")
    assert check.passed


def test_verifier_citation_accepted_when_numbers_match():
    answer = "The spouse gets a family pension [1]."   # no numbers, wrong source
    verifier = FakeVerifier([{"id": 1, "verdict": "supported", "sources": [2], "issue": ""}])
    check = OutputGuardrail(verifier).check("q", answer, SOURCES)
    assert check.cleaned_answer == "The spouse gets a family pension [2]."
