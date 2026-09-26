"""Phase 18: evaluation metrics.

Two groups:

* DeepEval (LLM-judged, need OPENAI_API_KEY)
    answer_relevancy      does the answer address the question?
    faithfulness          is every claim in the answer supported by the retrieved context?
    contextual_relevancy  is the retrieved context about the question?
    contextual_recall     does the retrieved context contain what the expected answer needs?
    contextual_precision  are the relevant chunks ranked above irrelevant ones?

* Deterministic (free, exact)
    behavior_correct   answered / refused / mentioned-only / blocked, as expected
    source_hit         a cited source is the expected document + page
    key_facts          expected numbers/phrases appear in the answer
"""
from __future__ import annotations

import re
from typing import Any, Callable

DEEPEVAL_METRICS = (
    "answer_relevancy",
    "faithfulness",
    "contextual_relevancy",
    "contextual_recall",
    "contextual_precision",
)


def build_deepeval_metrics(model: str, threshold: float) -> dict[str, Callable[[], Any]]:
    """Factories, because DeepEval metric objects keep per-test-case state."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )

    common = {"model": model, "threshold": threshold, "include_reason": True,
              "async_mode": False}
    return {
        "answer_relevancy": lambda: AnswerRelevancyMetric(**common),
        "faithfulness": lambda: FaithfulnessMetric(**common),
        "contextual_relevancy": lambda: ContextualRelevancyMetric(**common),
        "contextual_recall": lambda: ContextualRecallMetric(**common),
        "contextual_precision": lambda: ContextualPrecisionMetric(**common),
    }


# ---------------------------------------------------------------- deterministic
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().replace(",", "")).strip()


def behavior_correct(expected: str, observed: str) -> float:
    return 1.0 if expected == observed else 0.0


def source_hit(expected_document: str | None, expected_pages: list[int],
               cited_sources: list[dict]) -> float | None:
    """1.0 if any cited source is from the expected document (and page, when given)."""
    if not expected_document:
        return None
    want = expected_document.casefold()
    for s in cited_sources:
        if want not in (s.get("document_name") or "").casefold():
            continue
        if not expected_pages or s.get("page_number") in expected_pages:
            return 1.0
    return 0.0


def key_facts(expected_facts: list[str], answer: str) -> float | None:
    """Share of expected facts ("3,000", "18", "general body") present in the answer."""
    if not expected_facts:
        return None
    ans = _norm(answer)
    found = sum(1 for f in expected_facts if _norm(f) in ans)
    return round(found / len(expected_facts), 3)
