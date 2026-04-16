from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.rag.pipeline import run_rag_pipeline, DEFAULT_GENERATION_MODEL
from backend.app.rag.retriever import DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K

router = APIRouter()


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
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
