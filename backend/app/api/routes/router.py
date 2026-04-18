from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.rag.pipeline import DEFAULT_GENERATION_MODEL, run_rag_pipeline
from backend.app.rag.retriever import DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K

router = APIRouter()
_logger = logging.getLogger(__name__)


@router.get("/health")
def health() -> dict[str, str]:
    """Lightweight liveness probe used to wake the server from idle."""
    return {"status": "ok"}


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(DEFAULT_TOP_K)
    score_threshold: float = Field(DEFAULT_SCORE_THRESHOLD)
    period_label_key: str | None = None
    publication_date: str | None = None
    publication_date_from: str | None = None
    publication_date_to: str | None = None
    model: str = Field(DEFAULT_GENERATION_MODEL)


class QueryResponse(BaseModel):
    answer: str
    sources: list


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Execute the RAG pipeline for a user question.

    Args:
        request: The query parameters including the question and optional filters.

    Returns:
        A QueryResponse with the grounded answer and cited sources.

    Raises:
        HTTPException: 500 if the pipeline raises an unexpected error.
    """
    start = time.monotonic()
    _logger.info(
        "query.received",
        extra={
            "query": request.query,
            "top_k": request.top_k,
            "score_threshold": request.score_threshold,
            "period_label_key": request.period_label_key,
            "publication_date": request.publication_date,
            "publication_date_from": request.publication_date_from,
            "publication_date_to": request.publication_date_to,
            "model": request.model,
        },
    )
    try:
        result = run_rag_pipeline(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            period_label_key=request.period_label_key,
            publication_date=request.publication_date,
            publication_date_from=request.publication_date_from,
            publication_date_to=request.publication_date_to,
            model=request.model,
        )
        _logger.info(
            "query.completed",
            extra={
                "answer_length": len(result.get("answer", "")),
                "source_count": len(result.get("sources", [])),
                "duration_ms": round((time.monotonic() - start) * 1000, 1),
            },
        )
        return result
    except Exception as e:
        _logger.error(
            "query.failed",
            exc_info=True,
            extra={
                "error_type": type(e).__name__,
                "duration_ms": round((time.monotonic() - start) * 1000, 1),
            },
        )
        raise HTTPException(status_code=500, detail=str(e)) from e
