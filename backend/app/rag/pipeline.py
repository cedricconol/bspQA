"""RAG pipeline orchestration utilities."""

from __future__ import annotations

from typing import Any, Optional

from .generator import (
    DEFAULT_GENERATION_MODEL,
    SYSTEM_PROMPT,
    _build_context,
    _build_sources,
)
from .retriever import DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K, get_openai_client, retrieve_chunks


def run_rag_pipeline(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    period_label_key: Optional[str] = None,
    publication_date: Optional[str] = None,
    publication_date_from: Optional[str] = None,
    publication_date_to: Optional[str] = None,
    model: str = DEFAULT_GENERATION_MODEL,
    openai_client: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Retrieve relevant chunks and generate a grounded answer in one call.

    Returns a structured payload intended for API responses.
    """
    hits = retrieve_chunks(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        period_label_key=period_label_key,
        publication_date=publication_date,
        publication_date_from=publication_date_from,
        publication_date_to=publication_date_to,
    )
    sources = _build_sources(hits)
    if not hits:
        return {
            "answer": "I could not find a reliable answer in the available BSP reports.",
            "sources": [],
        }

    client = openai_client or get_openai_client()
    context = _build_context(hits)
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=(f"Question:\n{query}\n\nSources:\n{context}"),
    )
    answer_text = (response.output_text or "").strip()
    return {"answer": answer_text, "sources": sources}
