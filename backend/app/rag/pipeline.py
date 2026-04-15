"""RAG pipeline orchestration utilities."""

from __future__ import annotations

from typing import Any

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
    period_label_key: str | None = None,
    publication_date: str | None = None,
    publication_date_from: str | None = None,
    publication_date_to: str | None = None,
    model: str = DEFAULT_GENERATION_MODEL,
    openai_client: Any | None = None,
) -> dict[str, Any]:
    """Retrieve relevant chunks and generate a grounded answer in one call.

    Args:
        query: The natural language question to answer.
        top_k: Maximum number of chunks to retrieve.
        score_threshold: Minimum similarity score to include a chunk.
        period_label_key: Optional normalized period slug to filter by.
        publication_date: Exact ISO 8601 date to filter by.
        publication_date_from: Start of an ISO 8601 date range (inclusive).
        publication_date_to: End of an ISO 8601 date range (inclusive).
        model: OpenAI model name to use for generation.
        openai_client: Optional pre-built OpenAI client.

    Returns:
        A dict with "answer" (str) and "sources" (list) for use in API responses.
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
