"""Qdrant retrieval utilities for RAG."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Optional

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 15
DEFAULT_SCORE_THRESHOLD = 0.0
RECENCY_CANDIDATE_MULTIPLIER = 3
RECENCY_WEIGHT = 0.85


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    return _backend_root().parent


def load_env_from_dotenv() -> None:
    """
    Optional local-dev helper.

    In Railway (and most production deploys), environment variables are provided
    by the platform and `.env` loading is unnecessary. This function should be
    called explicitly by local scripts/tests when desired.
    """
    try:
        from dotenv import load_dotenv  # imported lazily for import-safety
    except ImportError as e:
        raise RuntimeError("Missing dependency: python-dotenv") from e
    load_dotenv(_repo_root() / ".env")
    load_dotenv(_backend_root() / ".env")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_openai_client() -> Any:
    try:
        from openai import OpenAI  # imported lazily for import-safety
    except ImportError as e:
        raise RuntimeError("Missing dependency: openai") from e
    return OpenAI(api_key=_require_env("OPENAI_API_KEY"))


def get_qdrant_client() -> Any:
    try:
        from qdrant_client import QdrantClient  # imported lazily for import-safety
    except ImportError as e:
        raise RuntimeError("Missing dependency: qdrant-client") from e
    return QdrantClient(
        url=_require_env("QDRANT_CLUSTER_ENDPOINT"),
        api_key=_require_env("QDRANT_API_KEY"),
        timeout=30.0,
    )


def get_qdrant_collection_name() -> str:
    return _require_env("QDRANT_COLLECTION_NAME")


def embed_query(client: Any, query: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    return response.data[0].embedding


def _build_query_filter(
    *,
    period_label_key: Optional[str] = None,
    publication_date: Optional[str] = None,
    publication_date_from: Optional[str] = None,
    publication_date_to: Optional[str] = None,
) -> Any:
    """
    Build a Qdrant metadata filter for optional period/date constraints.
    """
    must_conditions: list[Any] = []
    try:
        from qdrant_client import models as qmodels  # imported lazily
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


def _parse_publication_date(value: Any) -> Optional[date]:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _recency_boosted_hits(hits: list[Any], top_k: int) -> list[Any]:
    """
    Blend relevance score with source recency so newer docs are preferred
    when chunks are similarly relevant.
    """
    if len(hits) <= top_k:
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
    period_label_key: Optional[str] = None,
    publication_date: Optional[str] = None,
    publication_date_from: Optional[str] = None,
    publication_date_to: Optional[str] = None,
    openai_client: Optional[Any] = None,
    qdrant_client: Optional[Any] = None,
    collection_name: Optional[str] = None,
):
    """
    Retrieve relevant chunks from Qdrant for a natural language query.

    This function is import-safe and suitable for server environments. It does
    not read `.env` files automatically; provide environment variables via the
    process environment (e.g. Railway) or call `load_env_from_dotenv()` in local
    dev before invoking.
    """
    openai_client = openai_client or get_openai_client()
    qdrant_client = qdrant_client or get_qdrant_client()
    collection_name = collection_name or get_qdrant_collection_name()

    query_vector = embed_query(openai_client, query)
    query_filter = _build_query_filter(
        period_label_key=period_label_key,
        publication_date=publication_date,
        publication_date_from=publication_date_from,
        publication_date_to=publication_date_to,
    )
    candidate_limit = max(top_k, top_k * RECENCY_CANDIDATE_MULTIPLIER)
    points = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=candidate_limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
    ).points
    return _recency_boosted_hits(points, top_k)
