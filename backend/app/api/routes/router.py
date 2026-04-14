from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.rag.pipeline import run_rag_pipeline, DEFAULT_GENERATION_MODEL
from app.rag.retriever import DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(DEFAULT_TOP_K)
    score_threshold: float = Field(DEFAULT_SCORE_THRESHOLD)
    period_label_key: Optional[str] = None
    publication_date: Optional[str] = None
    publication_date_from: Optional[str] = None
    publication_date_to: Optional[str] = None
    model: str = Field(DEFAULT_GENERATION_MODEL)


class QueryResponse(BaseModel):
    answer: str
    sources: list


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
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
        raise HTTPException(status_code=500, detail=str(e))