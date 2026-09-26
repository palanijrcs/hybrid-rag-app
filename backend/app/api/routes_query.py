"""The question-answering endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_grounded_qa
from app.llm.client import LLMError
from app.llm.grounded_generation import GroundedQA
from app.models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, qa: GroundedQA = Depends(get_grounded_qa)) -> QueryResponse:
    """Answer a question using only the uploaded documents, with citations."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The question is empty.")
    logger.info("query.received chars=%d", len(question))
    try:
        result = qa.answer(question)
    except LLMError as error:
        logger.error("query.llm_failed %s", error)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The language model is unavailable right now. Please try again.",
        )
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        grounded=result.grounded,
        insufficient_evidence=result.insufficient_evidence,
        sources=[{**s.__dict__, "label": s.label} for s in result.sources],
        retrieval=result.retrieval,
        model=result.model,
        timings_ms=result.timings_ms,
        notes=result.notes,
        coverage=result.coverage,
        question_subject=result.question_subject,
        sources_subject=result.sources_subject,
    )
