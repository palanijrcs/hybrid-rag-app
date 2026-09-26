"""Phase 19: end-to-end smoke test against the REAL indexes, OpenAI and Neo4j.

Skipped by default. It costs a few cents (about 6 gpt-4o-mini calls). Run from backend/:
    $env:RUN_LIVE_TESTS="1"; python -m pytest -m live -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.evaluation.deepeval_runner import Evaluator, load_dataset

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1"),
]

GOLDEN = Path(__file__).resolve().parents[2] / "evaluation" / "datasets" / "golden.jsonl"
SMOKE_IDS = {"kmy-01", "mo-01", "rf-01", "gr-01"}   # answer, mentioned-only, refuse, blocked


def test_pipeline_behaves_on_key_questions():
    from app.core.dependencies import get_grounded_qa, get_input_guardrail

    cases = [c for c in load_dataset(GOLDEN) if c.id in SMOKE_IDS]
    run = Evaluator(get_input_guardrail(), get_grounded_qa()).run(cases, label="live-smoke")
    wrong = [(c["id"], c["expected_behavior"], c["observed_behavior"], c["error"])
             for c in run["cases"] if c["scores"]["behavior_correct"] != 1.0]
    assert not wrong, f"unexpected behaviour: {wrong}"
    answer = next(c for c in run["cases"] if c["id"] == "kmy-01")
    assert answer["scores"]["source_hit"] == 1.0
