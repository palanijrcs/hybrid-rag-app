"""The question-answering endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_grounded_qa, get_input_guardrail
from app.guardrails.input_guardrail import InputGuardrail
from app.llm.client import LLMError
from app.llm.grounded_generation import GroundedQA
from app.models.schemas import GuardrailInfo, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    guardrail: InputGuardrail = Depends(get_input_guardrail),
    qa: GroundedQA = Depends(get_grounded_qa),
) -> QueryResponse:
    """Answer a question using only the uploaded documents, with citations."""
    check = guardrail.check(request.question)
    info = GuardrailInfo(
        allowed=check.allowed,
        category=check.category,
        reason=check.reason,
        sanitized_query=check.sanitized_query,
        transformations=check.transformations,
    )
    # log the decision, never the question text itself
    logger.info("input_guardrail allowed=%s category=%s reason=%s changes=%s",
                check.allowed, check.category, check.reason, check.transformations)

    if not check.allowed:
        if check.category == "invalid" and check.reason == "empty question":
            raise HTTPException(status.HTTP_400_BAD_REQUEST, check.message)
        return QueryResponse(
            question=request.question,
            answer=check.message,
            grounded=True,               # nothing was claimed
            insufficient_evidence=False,
            notes=[f"Blocked by input guardrail: {check.reason}"],
            input_guardrail=info,
        )

    try:
        result = qa.answer(check.sanitized_query)
    except LLMError as error:
        logger.error("query.llm_failed %s", type(error).__name__)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The language model is unavailable right now. Please try again.",
        )
    return QueryResponse(
        question=request.question,
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
        input_guardrail=info,
        output_guardrail=result.output_guardrail,
    )
