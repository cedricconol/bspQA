"""RAG pipeline orchestration utilities."""

from __future__ import annotations

import logging
import time
from typing import Any

from opentelemetry import trace

from .generator import (
    DEFAULT_GENERATION_MODEL,
    SYSTEM_PROMPT,
    _build_context,
    _build_sources,
)
from .retriever import DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K, get_openai_client, retrieve_chunks

_logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)


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
    with _tracer.start_as_current_span("rag.pipeline") as span:
        span.set_attribute("query", query)
        span.set_attribute("model", model)
        span.set_attribute("top_k", top_k)

        hits = retrieve_chunks(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            period_label_key=period_label_key,
            publication_date=publication_date,
            publication_date_from=publication_date_from,
            publication_date_to=publication_date_to,
        )

        span.set_attribute("hit_count", len(hits))
        _logger.info("pipeline.retrieved", extra={"hit_count": len(hits)})

        sources = _build_sources(hits)
        if not hits:
            span.set_attribute("fallback", True)
            _logger.warning("pipeline.no_hits", extra={"query": query})
            return {
                "answer": "I could not find a reliable answer in the available BSP reports.",
                "sources": [],
            }

        client = openai_client or get_openai_client()
        context = _build_context(hits)

        with _tracer.start_as_current_span("rag.llm_generate") as llm_span:
            llm_span.set_attribute("model", model)
            llm_start = time.monotonic()
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=(f"Question:\n{query}\n\nSources:\n{context}"),
            )
            llm_duration_ms = round((time.monotonic() - llm_start) * 1000, 1)
            answer_text = (response.output_text or "").strip()
            llm_span.set_attribute("answer_length", len(answer_text))
            _logger.info(
                "pipeline.llm_done",
                extra={"model": model, "duration_ms": llm_duration_ms},
            )

        return {"answer": answer_text, "sources": sources}
