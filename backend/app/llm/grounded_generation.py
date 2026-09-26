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
    ) -> None:
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm = llm
        self.graph_retriever = graph_retriever
        self.kg_top_k = kg_top_k
        self.max_attempts = max_attempts

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
        data = self._generate(question, context)
        timings["generation"] = _ms(t1)

        coverage = str(data.get("coverage", "full")).strip().lower()
        if coverage not in COVERAGE_VALUES:
            coverage = "full"
        q_subject = str(data.get("question_subject") or "").strip() or None
        s_subject = str(data.get("sources_subject") or "").strip() or None
        subject_info = {"coverage": coverage, "question_subject": q_subject,
                        "sources_subject": s_subject}

        if coverage == "none":
            return self._insufficient(question, context, retrieval, timings,
                                      "The sources do not cover the question's subject.",
                                      model=self.llm.model, subject=subject_info)

        valid_refs = {s.ref for s in context.sources}
        answer, used, invalid = check_citations(data["answer"], valid_refs)
        notes = []
        if invalid:
            notes.append(f"Removed citations to non-existent sources: {sorted(set(invalid))}")

        if bool(data.get("insufficient_evidence")) or not answer:
            return self._insufficient(question, context, retrieval, timings,
                                      "The model found the evidence insufficient.",
                                      model=self.llm.model, extra=notes, subject=subject_info)
        if not used:
            logger.warning("qa.uncited_answer_rejected")
            return self._insufficient(question, context, retrieval, timings,
                                      "The model's answer cited no source, so it was not returned.",
                                      model=self.llm.model, extra=notes, subject=subject_info)

        if coverage == "mentioned_only":
            # Always tell the user plainly that the documents are about something else
            answer = f"{mentioned_only_notice(q_subject, s_subject)}\n\n{answer}"
            notes.append("The question's subject is only mentioned in passing in the documents.")

        by_ref = {s.ref: s for s in context.sources}
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
            **subject_info,
        )

    # ---------------------------------------------------------------- helpers
    def _generate(self, question: str, context: BuiltContext) -> dict[str, Any]:
        system, user = build_messages(question, context.text)
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
                      subject=None):
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
            **(subject or {}),
        )


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
