"""Qdrant retrieval utilities for RAG."""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any

from opentelemetry import trace

from ..config import get_settings

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 15
DEFAULT_SCORE_THRESHOLD = 0.0
RECENCY_CANDIDATE_MULTIPLIER = 3
RECENCY_WEIGHT = 0.85

_logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)


def get_openai_client() -> Any:
    """Return an OpenAI client configured from application settings.

    Returns:
        An authenticated OpenAI client instance.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Missing dependency: openai") from e
    return OpenAI(api_key=get_settings().openai_api_key)


def get_qdrant_client() -> Any:
    """Return a Qdrant client configured from application settings.

    Returns:
        An authenticated QdrantClient instance.
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError as e:
        raise RuntimeError("Missing dependency: qdrant-client") from e
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_cluster_endpoint,
        api_key=settings.qdrant_api_key,
        timeout=30.0,
    )


def get_qdrant_collection_name() -> str:
    """Return the Qdrant collection name from application settings."""
    return get_settings().qdrant_collection_name


def embed_query(client: Any, query: str) -> list[float]:
    """Embed a query string using the configured embedding model.

    Args:
        client: An OpenAI client instance.
        query: The natural language query to embed.

    Returns:
        A list of floats representing the query embedding.
    """
    with _tracer.start_as_current_span("rag.embed_query") as span:
        span.set_attribute("model", EMBEDDING_MODEL)
        start = time.monotonic()
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
        span.set_attribute("duration_ms", round((time.monotonic() - start) * 1000, 1))
        return response.data[0].embedding


def _build_query_filter(
    *,
    period_label_key: str | None = None,
    publication_date: str | None = None,
    publication_date_from: str | None = None,
    publication_date_to: str | None = None,
) -> Any:
    """Build a Qdrant metadata filter for optional period/date constraints."""
    must_conditions: list[Any] = []
    try:
        from qdrant_client import models as qmodels
    except ImportError as e:
        raise RuntimeError("Missing dependency: qdrant-client") from e

    if period_label_key:
        must_conditions.append(
            qmodels.FieldCondition(
                key="period_label_key",
                match=qmodels.MatchValue(value=period_label_key),
            )
        )

    if publication_date:
        must_conditions.append(
            qmodels.FieldCondition(
                key="publication_date",
                match=qmodels.MatchValue(value=publication_date),
            )
        )
    elif publication_date_from or publication_date_to:
        must_conditions.append(
            qmodels.FieldCondition(
                key="publication_date",
                range=qmodels.DatetimeRange(
                    gte=publication_date_from,
                    lte=publication_date_to,
                ),
            )
        )

    if not must_conditions:
        return None
    return qmodels.Filter(must=must_conditions)


def _parse_publication_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _recency_boosted_hits(hits: list[Any], top_k: int) -> list[Any]:
    """Blend relevance score with source recency so newer docs are preferred.

    Args:
        hits: Qdrant search result points.
        top_k: Maximum number of results to return.

    Returns:
        Re-ranked and trimmed list of hits.
    """
    if not hits:
        return hits

    dated_hits: list[tuple[Any, date]] = []
    for hit in hits:
        payload = getattr(hit, "payload", {}) or {}
        publication_date = _parse_publication_date(payload.get("publication_date"))
        if publication_date is not None:
            dated_hits.append((hit, publication_date))

    if not dated_hits:
        return hits[:top_k]

    min_date = min(d for _, d in dated_hits)
    max_date = max(d for _, d in dated_hits)
    date_span_days = (max_date - min_date).days
    max_score = max(float(getattr(hit, "score", 0.0)) for hit in hits) or 1.0

    def combined_score(hit: Any) -> float:
        relevance = float(getattr(hit, "score", 0.0)) / max_score
        payload = getattr(hit, "payload", {}) or {}
        pub_date = _parse_publication_date(payload.get("publication_date"))
        if pub_date is None or date_span_days <= 0:
            recency = 0.0
        else:
            recency = (pub_date - min_date).days / date_span_days
        return (1.0 - RECENCY_WEIGHT) * relevance + RECENCY_WEIGHT * recency

    return sorted(hits, key=combined_score, reverse=True)[:top_k]


def retrieve_chunks(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    period_label_key: str | None = None,
    publication_date: str | None = None,
    publication_date_from: str | None = None,
    publication_date_to: str | None = None,
    openai_client: Any | None = None,
    qdrant_client: Any | None = None,
    collection_name: str | None = None,
) -> list[Any]:
    """Retrieve relevant chunks from Qdrant for a natural language query.

    Args:
        query: The natural language question to answer.
        top_k: Maximum number of chunks to return after re-ranking.
        score_threshold: Minimum cosine similarity score to include a result.
        period_label_key: Optional normalized period slug to filter by (e.g. "q1_2022").
        publication_date: Exact ISO 8601 date to filter by.
        publication_date_from: Start of an ISO 8601 date range (inclusive).
        publication_date_to: End of an ISO 8601 date range (inclusive).
        openai_client: Optional pre-built OpenAI client (defaults to get_openai_client()).
        qdrant_client: Optional pre-built QdrantClient (defaults to get_qdrant_client()).
        collection_name: Optional collection name override.

    Returns:
        A list of Qdrant ScoredPoint objects, re-ranked by recency-blended score.
    """
    openai_client = openai_client or get_openai_client()
    qdrant_client = qdrant_client or get_qdrant_client()
    collection_name = collection_name or get_qdrant_collection_name()

    with _tracer.start_as_current_span("rag.retrieve_chunks") as span:
        span.set_attribute("query", query)
        span.set_attribute("top_k", top_k)

        query_vector = embed_query(openai_client, query)  # child span: rag.embed_query
        query_filter = _build_query_filter(
            period_label_key=period_label_key,
            publication_date=publication_date,
            publication_date_from=publication_date_from,
            publication_date_to=publication_date_to,
        )
        candidate_limit = max(top_k, top_k * RECENCY_CANDIDATE_MULTIPLIER)
        filter_summary = {
            "period_label_key": period_label_key,
            "publication_date": publication_date,
            "publication_date_from": publication_date_from,
            "publication_date_to": publication_date_to,
        }

        with _tracer.start_as_current_span("qdrant.query_points") as qs:
            qs.set_attribute("collection", collection_name)
            qs.set_attribute("candidate_limit", candidate_limit)
            qdrant_start = time.monotonic()
            points = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=candidate_limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            ).points
            qdrant_duration_ms = round((time.monotonic() - qdrant_start) * 1000, 1)
            qs.set_attribute("raw_hit_count", len(points))
            qs.set_attribute("duration_ms", qdrant_duration_ms)

        _logger.info(
            "qdrant.query_done",
            extra={
                "raw_hit_count": len(points),
                "candidate_limit": candidate_limit,
                "filter": filter_summary,
                "duration_ms": qdrant_duration_ms,
            },
        )

        ranked = _recency_boosted_hits(points, top_k)
        span.set_attribute("ranked_count", len(ranked))
        _logger.debug(
            "recency_rerank.done",
            extra={"input_count": len(points), "output_count": len(ranked)},
        )
        return ranked
