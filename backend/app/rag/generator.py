"""Answer generation utilities for RAG."""

from __future__ import annotations

from typing import Any, Optional

from .retriever import (
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    get_openai_client,
    retrieve_chunks,
)

DEFAULT_GENERATION_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a research assistant specializing in Philippine monetary policy and economic analysis.
You must answer questions based exclusively on the provided excerpts from official BSP reports.

You will be given:
- A Question
- A set of Sources, each with an excerpt and minimal metadata

Rules:
- Use only the provided Sources. Do not use outside knowledge.
- If the Sources do not contain enough information to answer, say exactly:
  "I could not find a reliable answer in the available BSP reports."
- Every factual claim must be supported by at least one citation in the format [Source N].
- Do not speculate or infer beyond what the excerpts explicitly state.
- Use clear, plain English; keep it concise.
- If the question asks about a time period not covered by the Sources, say so explicitly.

Output format:
- Answer: (your answer with inline citations like [Source 1])
- Sources: (a list of each cited source, one per line)
"""


def _build_context(hits: list[Any]) -> str:
    sections: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        text = str(payload.get("text", "")).strip()
        source = payload.get("source_file", "unknown")
        chunk_index = payload.get("chunk_index", "unknown")
        score = getattr(hit, "score", 0.0)
        sections.append(
            "\n".join(
                [
                    f"[Source {idx}]",
                    f"source_file: {source}",
                    f"chunk_index: {chunk_index}",
                    f"score: {score:.4f}",
                    f"text: {text}",
                ]
            )
        )
    return "\n\n".join(sections)


def _build_sources(hits: list[Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for idx, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        sources.append(
            {
                "source_id": idx,
                "source_file": payload.get("source_file", "unknown"),
                "chunk_index": payload.get("chunk_index", "unknown"),
                "score": float(getattr(hit, "score", 0.0)),
            }
        )
    return sources


def generate_answer(
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
    Generate a grounded answer from retrieved sources.

    Returns a structured payload intended for use by an HTTP API layer.
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
