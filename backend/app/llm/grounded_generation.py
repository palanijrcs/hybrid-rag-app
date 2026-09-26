"""Phase 14: the full question-answering pipeline.

    question
      -> hybrid retrieval (FAISS + BM25 + graph) -> fusion -> re-rank + threshold
      -> context builder (numbered sources)
      -> empty?  yes -> "not enough information" (the LLM is never called)
      -> grounded LLM (JSON, temperature 0)
      -> citation check: only real source numbers; no valid citation -> "not enough information"
      -> answer + cited sources + retrieval stats

Phase 15/16 guardrails plug in before retrieval and after generation.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.context_builder import BuiltContext, ContextBuilder, ContextSource
from app.retrieval.fusion import fuse
from app.retrieval.hybrid_retriever import HybridRetriever

from .client import ChatClient, LLMError
from .prompts import (
    COVERAGE_VALUES,
    INSUFFICIENT_EVIDENCE_ANSWER,
    build_messages,
    mentioned_only_notice,
)

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d{1,2})\]")


@dataclass
class QAResult:
    question: str
    answer: str
    grounded: bool
    insufficient_evidence: bool
    sources: list[ContextSource] = field(default_factory=list)       # cited sources
    context_sources: list[ContextSource] = field(default_factory=list)  # all given to the LLM
    retrieval: dict[str, int] = field(default_factory=dict)
    model: str | None = None
    timings_ms: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    coverage: str | None = None          # full | partial | mentioned_only | none
    question_subject: str | None = None
    sources_subject: str | None = None
    output_guardrail: dict[str, Any] = field(default_factory=dict)


def _citations(text: str) -> list[int]:
    return [int(n) for n in _CITATION.findall(text)]


def parse_llm_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        raise ValueError("response is not an object with an 'answer' string")
    return data


def check_citations(answer: str, valid_refs: set[int]) -> tuple[str, list[int], list[int]]:
    """Remove citations to sources that don't exist. Returns (answer, used, invalid)."""
    used: list[int] = []
    invalid: list[int] = []

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if n in valid_refs:
            if n not in used:
                used.append(n)
            return m.group(0)
        invalid.append(n)
        return ""

    cleaned = _CITATION.sub(repl, answer)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), used, invalid


class GroundedQA:
    def __init__(
        self,
        retriever: HybridRetriever,
        context_builder: ContextBuilder,
        llm: ChatClient | None,
        graph_retriever: Any | None = None,
        kg_top_k: int = 10,
        max_attempts: int = 2,
        output_guardrail: Any | None = None,
        max_regenerations: int = 1,
    ) -> None:
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm = llm
        self.graph_retriever = graph_retriever
        self.kg_top_k = kg_top_k
        self.max_attempts = max_attempts
        self.output_guardrail = output_guardrail
        self.max_regenerations = max_regenerations if output_guardrail is not None else 0

    # ---------------------------------------------------------------- pipeline
    def answer(self, question: str) -> QAResult:
        timings: dict[str, int] = {}
        t0 = time.perf_counter()

        per_retriever = self.retriever.retrieve_each(question)
        fused = fuse(per_retriever, method=self.retriever.fusion_method)
        hits = self.retriever.rerank(question, fused)
        timings["retrieval"] = _ms(t0)

        graph_evidence = self._graph_evidence(question) if hits else []
        context = self.context_builder.build(question, hits, graph_evidence)
        retrieval = {
            **{name: len(h) for name, h in per_retriever.items()},
            "fused": len(fused),
            "reranked": len(hits),
            "context_sources": len(context.sources),
            "graph_facts": len(context.facts),
        }

        if context.is_empty:
            logger.info("qa.insufficient reason=no_relevant_evidence")
            return self._insufficient(question, context, retrieval, timings,
                                      "No retrieved passage was relevant enough.")
        if self.llm is None:
            raise LLMError("No language model is configured (check OPENAI_API_KEY).")

        t1 = time.perf_counter()
        feedback: str | None = None
        guard_info: dict[str, Any] = {"enabled": self.output_guardrail is not None,
                                      "regenerated": False}
        notes: list[str] = []
        by_ref = {s.ref: s for s in context.sources}
        valid_refs = set(by_ref)

        for round_no in range(self.max_regenerations + 1):
            data = self._generate(question, context, feedback)
            coverage = str(data.get("coverage", "full")).strip().lower()
            if coverage not in COVERAGE_VALUES:
                coverage = "full"
            q_subject = str(data.get("question_subject") or "").strip() or None
            s_subject = str(data.get("sources_subject") or "").strip() or None
            subject_info = {"coverage": coverage, "question_subject": q_subject,
                            "sources_subject": s_subject}

            if coverage == "none":
                timings["generation"] = _ms(t1)
                return self._insufficient(question, context, retrieval, timings,
                                          "The sources do not cover the question's subject.",
                                          model=self.llm.model, subject=subject_info,
                                          guard=guard_info)

            answer, used, invalid = check_citations(data["answer"], valid_refs)
            notes = [f"Removed citations to non-existent sources: {sorted(set(invalid))}"] \
                if invalid else []
            if bool(data.get("insufficient_evidence")) or not answer:
                timings["generation"] = _ms(t1)
                return self._insufficient(question, context, retrieval, timings,
                                          "The model found the evidence insufficient.",
                                          model=self.llm.model, extra=notes,
                                          subject=subject_info, guard=guard_info)
            if not used:
                logger.warning("qa.uncited_answer_rejected")
                timings["generation"] = _ms(t1)
                return self._insufficient(question, context, retrieval, timings,
                                          "The model's answer cited no source, so it was not "
                                          "returned.", model=self.llm.model, extra=notes,
                                          subject=subject_info, guard=guard_info)
            if coverage == "mentioned_only":
                answer = f"{mentioned_only_notice(q_subject, s_subject)}\n\n{answer}"

            if self.output_guardrail is None:
                break
            check = self.output_guardrail.check(question, answer, context.sources)
            guard_info.update({
                "passed": check.passed,
                "verifier_used": check.verifier_used,
                "verifier_error": check.verifier_error,
                "issues": [f"{s.text} -> {'; '.join(s.issues)}"
                           for s in check.sentences if s.issues],
            })
            if check.passed or round_no == self.max_regenerations:
                break
            # one more attempt, telling the model exactly what was wrong
            logger.info("qa.regenerating unsupported=%d", len(check.unsupported))
            guard_info["regenerated"] = True
            feedback = check.feedback()

        timings["generation"] = _ms(t1)

        if self.output_guardrail is not None:
            removed = len(check.unsupported)
            guard_info.update({"removed_sentences": removed,
                               "corrected_citations": len(check.corrected)})
            if removed:
                notes.append(f"Removed {removed} sentence(s) not supported by the sources.")
            if check.corrected:
                notes.append(f"Corrected citations in {len(check.corrected)} sentence(s).")
            if not check.has_supported_facts:
                return self._insufficient(question, context, retrieval, timings,
                                          "No part of the answer could be verified against "
                                          "the sources.", model=self.llm.model, extra=notes,
                                          subject=subject_info, guard=guard_info)
            answer = check.cleaned_answer
            used = sorted({r for c in check.sentences if c.status in ("supported", "corrected")
                           for r in _citations(c.fixed_text or c.text)} & valid_refs)
        if coverage == "mentioned_only":
            notes.append("The question's subject is only mentioned in passing in the documents.")

        timings["total"] = _ms(t0)
        logger.info("qa.answered sources=%s timings=%s", used, timings)
        return QAResult(
            question=question,
            answer=answer,
            grounded=True,
            insufficient_evidence=False,
            sources=[by_ref[r] for r in sorted(used)],
            context_sources=context.sources,
            retrieval=retrieval,
            model=self.llm.model,
            timings_ms=timings,
            notes=notes,
            output_guardrail=guard_info,
            **subject_info,
        )

    # ---------------------------------------------------------------- helpers
    def _generate(self, question: str, context: BuiltContext,
                  feedback: str | None = None) -> dict[str, Any]:
        system, user = build_messages(question, context.text)
        if feedback:
            user += (
                "\n\nYour previous answer contained statements that the sources do not "
                "support:\n" + feedback + "\nWrite the answer again. Include only facts "
                "stated in the sources, with correct [n] citations."
            )
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            raw = self.llm.complete_json(system, user)  # LLMError propagates
            try:
                return parse_llm_json(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("qa.bad_llm_json attempt=%d err=%s", attempt, exc)
        raise LLMError(f"The model did not return valid JSON: {last_error}")

    def _graph_evidence(self, question: str) -> list[Any]:
        if self.graph_retriever is None:
            return []
        try:
            return self.graph_retriever.retrieve(question, top_k=self.kg_top_k).evidence
        except Exception:
            logger.exception("qa.graph_evidence_failed")
            return []

    @staticmethod
    def _insufficient(question, context, retrieval, timings, reason, model=None, extra=None,
                      subject=None, guard=None):
        return QAResult(
            question=question,
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            grounded=True,  # refusing is always grounded
            insufficient_evidence=True,
            sources=[],
            context_sources=context.sources,
            retrieval=retrieval,
            model=model,
            timings_ms=timings,
            notes=[reason, *(extra or [])],
            output_guardrail=guard or {},
            **(subject or {}),
        )


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
