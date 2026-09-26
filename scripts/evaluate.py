"""Phase 18: evaluate the RAG pipeline on the golden dataset.

Usage (from backend/, with the venv active):
    python ../scripts/evaluate.py --label baseline            # full run, DeepEval scoring
    python ../scripts/evaluate.py --limit 3 --label smoke     # quick check
    python ../scripts/evaluate.py --no-deepeval               # free: behaviour/source/fact checks only
    python ../scripts/evaluate.py --compare                   # table of saved runs

Change a setting in .env (e.g. RERANK_TOP_K=3), run again with a new --label,
then --compare to see which configuration scores best.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.deepeval_runner import (  # noqa: E402
    Evaluator,
    list_runs,
    load_dataset,
    prepare_environment,
    save_run,
    settings_snapshot,
)
from app.evaluation.metrics import build_deepeval_metrics  # noqa: E402

SHORT = {"behavior_correct": "behav", "source_hit": "src", "key_facts": "facts",
         "answer_relevancy": "ans_rel", "faithfulness": "faith",
         "contextual_relevancy": "ctx_rel", "contextual_recall": "recall",
         "contextual_precision": "prec"}


def print_summary(summary: dict) -> None:
    print(f"\nCases: {summary['cases']}   errors: {summary['errors']}   "
          f"avg time: {summary['avg_seconds']}s")
    print("\nBehaviour (correct / total):")
    for name, b in summary["behavior"].items():
        print(f"  {name:<15} {b['correct']}/{b['total']}")
    print("\nMetric means:")
    for name, m in summary["metrics"].items():
        print(f"  {name:<22} {m['mean']:.3f}   (n={m['n']})")


def compare(results_dir: Path) -> None:
    runs = list_runs(results_dir) if results_dir.exists() else []
    if not runs:
        print("No saved runs yet.")
        return
    cols = list(SHORT)
    print(f"{'run':<34}" + "".join(f"{SHORT[c]:>8}" for c in cols))
    for run in runs:
        metrics = run["summary"].get("metrics", {})
        cells = "".join(f"{metrics[c]['mean']:>8.2f}" if c in metrics else f"{'-':>8}" for c in cols)
        print(f"{run['file'][:33]:<34}{cells}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", help="JSONL golden dataset")
    parser.add_argument("--label", default="run", help="name for this run")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--no-deepeval", action="store_true", help="skip LLM-judged metrics")
    parser.add_argument("--compare", action="store_true", help="compare saved runs and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()
    results_dir = settings.resolve_path(settings.evaluation_results_dir)
    if args.compare:
        compare(results_dir)
        return 0

    dataset = Path(args.dataset) if args.dataset else settings.resolve_path(settings.evaluation_dataset)
    cases = load_dataset(dataset)[: args.limit]
    api_key = settings.openai_api_key.get_secret_value()
    if not api_key:
        print("OPENAI_API_KEY is not set in .env.")
        return 1
    prepare_environment(api_key)

    from app.core.dependencies import get_grounded_qa, get_input_guardrail
    from app.knowledge_graph.extraction_config import get_kg_settings

    metrics = {} if args.no_deepeval else build_deepeval_metrics(
        settings.evaluation_model, settings.evaluation_threshold)
    evaluator = Evaluator(
        get_input_guardrail(), get_grounded_qa(), metrics,
        settings_snapshot(settings, get_kg_settings(),
                          None if args.no_deepeval else settings.evaluation_model),
    )

    print(f"Evaluating {len(cases)} question(s) from {dataset.name}"
          f"{'' if metrics else ' (no DeepEval)'}...\n")

    def progress(i: int, n: int, r) -> None:
        mark = "OK " if r.scores.get("behavior_correct") == 1.0 else "BAD"
        scores = " ".join(f"{SHORT.get(k, k)}={v:.2f}" for k, v in r.scores.items()
                          if v is not None and k != "behavior_correct")
        print(f"[{i:>2}/{n}] {mark} {r.id:<7} expected={r.expected_behavior:<14} "
              f"got={r.observed_behavior:<14} {scores}")

    run = evaluator.run(cases, label=args.label, progress=progress)
    path = save_run(run, results_dir)
    print_summary(run["summary"])
    print(f"\nSaved: {path}\n       {path.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
