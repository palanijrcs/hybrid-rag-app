"""Phase 18: run the golden dataset through the full pipeline and score it.

Runs in-process (not through the HTTP API) so it is independent from production
traffic, and uses exactly the same guardrails, retrieval, re-ranking, context
builder, LLM and output checks as POST /query.

Each run is saved to evaluation/results/<timestamp>_<label>.json with a snapshot
of the settings, so runs with different chunk sizes, top-k values, thresholds,
re-rankers, prompts or models can be compared.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from app.evaluation.metrics import (
    DEEPEVAL_METRICS,
    behavior_correct,
    key_facts,
    source_hit,
)

logger = logging.getLogger(__name__)

DETERMINISTIC_METRICS = ("behavior_correct", "source_hit", "key_facts")


@dataclass
class EvalCase:
    id: str
    question: str
    expected_behavior: str          # answer | refuse | mentioned_only | blocked
    expected_answer: str = ""
    expected_facts: list[str] = field(default_factory=list)
    expected_document: str | None = None
    expected_pages: list[int] = field(default_factory=list)


@dataclass
class CaseResult:
    id: str
    question: str
    expected_behavior: str
    observed_behavior: str
    answer: str
    expected_answer: str
    cited_sources: list[dict]
    retrieval_context: list[str]
    scores: dict[str, float | None] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None


def load_dataset(path: Path) -> list[EvalCase]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            cases.append(EvalCase(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{path.name} line {line_no}: {exc}") from exc
    return cases


def observed_behavior(guard_allowed: bool, result: Any | None) -> str:
    if not guard_allowed:
        return "blocked"
    if result is None or result.insufficient_evidence:
        return "refuse"
    if result.coverage == "mentioned_only":
        return "mentioned_only"
    return "answer"


class Evaluator:
    def __init__(
        self,
        guardrail: Any,
        qa: Any,
        deepeval_metrics: dict[str, Callable[[], Any]] | None = None,
        settings_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.guardrail = guardrail
        self.qa = qa
        self.deepeval_metrics = deepeval_metrics or {}
        self.settings_snapshot = settings_snapshot or {}

    # ---------------------------------------------------------------- one case
    def run_case(self, case: EvalCase) -> CaseResult:
        start = time.perf_counter()
        check = self.guardrail.check(case.question)
        result, error = None, None
        if check.allowed:
            try:
                result = self.qa.answer(check.sanitized_query)
            except Exception as exc:  # keep evaluating the other cases
                error = f"{type(exc).__name__}: {exc}"[:300]
                logger.error("eval.case_failed id=%s %s", case.id, error)

        behavior = "error" if error else observed_behavior(check.allowed, result)
        cited = [
            {"ref": s.ref, "document_name": s.document_name, "page_number": s.page_number,
             "chunk_id": s.chunk_id}
            for s in (result.sources if result else [])
        ]
        record = CaseResult(
            id=case.id, question=case.question,
            expected_behavior=case.expected_behavior, observed_behavior=behavior,
            answer=(result.answer if result else check.message) or "",
            expected_answer=case.expected_answer, cited_sources=cited,
            retrieval_context=[s.text for s in (result.context_sources if result else [])],
            error=error,
        )

        record.scores["behavior_correct"] = behavior_correct(case.expected_behavior, behavior)
        if case.expected_behavior in ("answer", "mentioned_only") and behavior == case.expected_behavior:
            record.scores["source_hit"] = source_hit(case.expected_document,
                                                     case.expected_pages, cited)
            record.scores["key_facts"] = key_facts(case.expected_facts, record.answer)
            if case.expected_behavior == "answer":
                self._score_with_deepeval(case, record)

        record.seconds = round(time.perf_counter() - start, 2)
        logger.info("eval.case id=%s behavior=%s scores=%s", case.id, behavior, record.scores)
        return record

    def _score_with_deepeval(self, case: EvalCase, record: CaseResult) -> None:
        if not self.deepeval_metrics or not record.retrieval_context:
            return
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=case.question,
            actual_output=record.answer,
            expected_output=case.expected_answer,
            retrieval_context=record.retrieval_context,
        )
        for name, factory in self.deepeval_metrics.items():
            try:
                metric = factory()
                metric.measure(test_case)
                record.scores[name] = round(float(metric.score), 3)
                record.reasons[name] = str(metric.reason or "")[:500]
            except Exception as exc:  # one failing metric must not stop the run
                record.scores[name] = None
                record.reasons[name] = f"metric failed: {type(exc).__name__}: {exc}"[:300]

    # ---------------------------------------------------------------- whole run
    def run(self, cases: list[EvalCase], label: str = "run",
            progress: Callable[[int, int, CaseResult], None] | None = None) -> dict[str, Any]:
        started = datetime.now()
        results = []
        for i, case in enumerate(cases, 1):
            record = self.run_case(case)
            results.append(record)
            if progress:
                progress(i, len(cases), record)
        return {
            "label": label,
            "started_at": started.isoformat(timespec="seconds"),
            "settings": self.settings_snapshot,
            "summary": summarize(results),
            "cases": [asdict(r) for r in results],
        }


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    names = [*DETERMINISTIC_METRICS, *DEEPEVAL_METRICS]
    metrics = {}
    for name in names:
        values = [r.scores[name] for r in results if r.scores.get(name) is not None]
        if values:
            metrics[name] = {"mean": round(mean(values), 3), "n": len(values)}
    by_behavior: dict[str, dict[str, int]] = {}
    for r in results:
        b = by_behavior.setdefault(r.expected_behavior, {"total": 0, "correct": 0})
        b["total"] += 1
        b["correct"] += int(r.scores.get("behavior_correct") == 1.0)
    return {
        "cases": len(results),
        "errors": sum(1 for r in results if r.error),
        "metrics": metrics,
        "behavior": by_behavior,
        "avg_seconds": round(mean(r.seconds for r in results), 2) if results else 0,
    }


# ---------------------------------------------------------------- saving / comparing
def save_run(run: dict[str, Any], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in run["label"])[:40]
    path = results_dir / f"{stamp}_{safe_label}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["id", "expected_behavior", "observed_behavior",
                *DETERMINISTIC_METRICS, *DEEPEVAL_METRICS, "seconds", "question", "answer"]
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for case in run["cases"]:
            writer.writerow({**case, **case["scores"],
                             "answer": case["answer"].replace("\n", " ")})
    return path


def list_runs(results_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(results_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        runs.append({"file": path.name, "label": data.get("label"),
                     "started_at": data.get("started_at"),
                     "settings": data.get("settings", {}), "summary": data.get("summary", {})})
    return runs


def settings_snapshot(settings: Any, kg_settings: Any, eval_model: str | None) -> dict[str, Any]:
    """The knobs worth comparing between runs (never secrets)."""
    keys = ["chunk_size", "chunk_overlap", "vector_top_k", "bm25_top_k", "kg_top_k",
            "rerank_top_k", "rerank_candidates", "min_relevance_score", "fusion_method",
            "enable_vector_retrieval", "enable_bm25", "enable_kg_retrieval", "enable_reranker",
            "reranker_model", "embedding_model", "llm_model", "enable_output_guardrail",
            "enable_llm_verification", "max_regenerations", "context_max_chars"]
    snap = {k: getattr(settings, k, None) for k in keys}
    snap["kg_extraction_model"] = getattr(kg_settings, "kg_extraction_model", None)
    snap["eval_model"] = eval_model
    return snap


def prepare_environment(api_key: str) -> None:
    """DeepEval reads the key from the environment; also turn off its telemetry."""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("ERROR_REPORTING", "NO")
