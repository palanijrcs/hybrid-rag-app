"""Phase 18: list saved evaluation runs (runs are started with scripts/evaluate.py)."""
from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.evaluation.deepeval_runner import list_runs

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("")
def evaluation_runs(limit: int = 20) -> list[dict]:
    """Summaries of the most recent runs, newest first."""
    results_dir = get_settings().resolve_path(get_settings().evaluation_results_dir)
    if not results_dir.exists():
        return []
    return list_runs(results_dir)[: max(1, min(limit, 100))]


@router.get("/{file_name}")
def evaluation_run(file_name: str) -> dict:
    """Full per-question results of one run."""
    import json

    if "/" in file_name or "\\" in file_name or not file_name.endswith(".json"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid run file name.")
    path = get_settings().resolve_path(get_settings().evaluation_results_dir) / file_name
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found.")
    return json.loads(path.read_text(encoding="utf-8"))
