"""Answer generation utilities for RAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_GENERATION_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a research assistant specializing in Philippine monetary policy and economic analysis.
You must answer questions based exclusively on the provided excerpts from official BSP reports.

You will be given:
- A Question
- A set of numbered Sources (e.g. [1], [2], …), each with an excerpt and minimal metadata

Rules:
- Use only the provided Sources. Do not use outside knowledge.
- If the Sources do not contain enough information to answer, say exactly:
  "I could not find a reliable answer in the available BSP reports."
- Every factual claim must be cited inline using the bracketed number of the source, e.g. [1] or [2].
- Do not speculate or infer beyond what the excerpts explicitly state.
- Use clear, plain English; keep it concise.
- If the question asks about a time period not covered by the Sources, say so explicitly.
- When multiple sources address the same question, prefer the one with the most recent publication_date. Only cite an older source if the newer one does not contain the relevant information.

Output format:
- Output ONLY the answer text with inline citations like [1].
- Do NOT include a "Sources:" or "References:" list — that is handled separately.
"""


def _load_manifest_lookup() -> dict[str, dict[str, str | None]]:
    """Build a source_file → {url, period_label, title} lookup from manifest.json.

    Returns:
        A dict keyed by source txt filename (e.g. ``FullReport_2022_1.txt``).
    """
    manifest_path = Path(__file__).resolve().parents[3] / "ingestion" / "manifest.json"
    if not manifest_path.is_file():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    lookup: dict[str, dict[str, str | None]] = {}
    for doc in data.get("documents", []):
        url = doc.get("url")
        if not isinstance(url, str):
            continue
        source_file = Path(url).stem + ".txt"
        lookup[source_file] = {
            "url": url,
            "period_label": doc.get("period_label"),
            "title": doc.get("title"),
        }
    return lookup


_MANIFEST_LOOKUP: dict[str, dict[str, str | None]] = _load_manifest_lookup()


def _build_context(hits: list[Any]) -> str:
    """Serialize retrieved chunks into a numbered source block for the LLM prompt.

    Args:
        hits: Qdrant ScoredPoint objects with payload fields.

    Returns:
        A formatted multi-line string of numbered source excerpts.
    """
    sections: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        text = str(payload.get("text", "")).strip()
        source = payload.get("source_file", "unknown")
        chunk_index = payload.get("chunk_index", "unknown")
        publication_date = payload.get("publication_date", "unknown")
        sections.append(
            "\n".join(
                [
                    f"[{idx}]",
                    f"source_file: {source}",
                    f"publication_date: {publication_date}",
                    f"chunk_index: {chunk_index}",
                    f"text: {text}",
                ]
            )
        )
    return "\n\n".join(sections)


def _build_sources(hits: list[Any]) -> list[dict[str, Any]]:
    """Build the structured sources list for the API response.

    Args:
        hits: Qdrant ScoredPoint objects with payload fields.

    Returns:
        A list of dicts with source_id, source_file, chunk_index, and score.
    """
    sources: list[dict[str, Any]] = []
    for idx, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        source_file = payload.get("source_file", "unknown")
        meta = _MANIFEST_LOOKUP.get(str(source_file), {})
        sources.append(
            {
                "source_id": idx,
                "source_file": source_file,
                "chunk_index": payload.get("chunk_index", "unknown"),
                "score": float(getattr(hit, "score", 0.0)),
                "period_label": meta.get("period_label") or payload.get("period_label"),
                "title": meta.get("title"),
                "url": meta.get("url"),
            }
        )
    return sources
