"""LLM-as-judge scoring helpers and deterministic metrics for bspQA evals."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

_logger = logging.getLogger(__name__)

_FAITHFULNESS_SYSTEM = """\
You are a strict faithfulness evaluator for a RAG system.

Given:
- SOURCES: the only text excerpts the system was allowed to use
- ANSWER: the system-generated answer

Rate how well every factual claim in ANSWER is grounded in SOURCES.
- 1.0: all claims are directly and explicitly supported by SOURCES
- 0.5: most claims are supported, but some are inferred or partially unsupported
- 0.0: major claims are unsupported, fabricated, or contradict SOURCES

Respond with ONLY valid JSON, no markdown: {"score": <0.0-1.0>, "reason": "<one sentence>"}\
"""

_RELEVANCE_SYSTEM = """\
You are a relevance evaluator for a Q&A system.

Given a QUESTION and an ANSWER, rate how directly and completely the ANSWER addresses the QUESTION.
- 1.0: fully and directly answers the question
- 0.5: partially answers, or addresses only part of the question
- 0.0: does not address the question, or answers a different question

Respond with ONLY valid JSON, no markdown: {"score": <0.0-1.0>, "reason": "<one sentence>"}\
"""


def _parse_judge_response(raw: str) -> float:
    """Extract the score float from a raw LLM judge response string.

    Tries strict JSON parse first, then falls back to regex for malformed outputs.

    Args:
        raw: Raw string content from the LLM response.

    Returns:
        A float in [0.0, 1.0]. Returns 0.0 and logs a warning on total parse failure.
    """
    try:
        data = json.loads(raw.strip())
        return float(data["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', raw)
    if match:
        try:
            return min(1.0, max(0.0, float(match.group(1))))
        except ValueError:
            pass

    _logger.warning("judge.parse_failed", extra={"raw_response": raw[:200]})
    return 0.0


def score_faithfulness(
    answer: str,
    sources_text: str,
    client: Any,
    *,
    model: str = "gpt-4o-mini",
) -> float:
    """Score whether every factual claim in answer is supported by sources_text.

    Args:
        answer: The RAG system's generated answer string.
        sources_text: Raw source excerpts the system retrieved (concatenated chunk text).
        client: An authenticated OpenAI client instance.
        model: OpenAI model name to use for judging.

    Returns:
        A float in [0.0, 1.0]. Returns 0.0 on parse failure after logging a warning.
    """
    prompt = f"SOURCES:\n{sources_text}\n\nANSWER:\n{answer}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _FAITHFULNESS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
    except Exception:
        _logger.exception("judge.faithfulness_api_error")
        return 0.0
    return _parse_judge_response(raw)


def score_relevance(
    question: str,
    answer: str,
    client: Any,
    *,
    model: str = "gpt-4o-mini",
) -> float:
    """Score whether answer addresses the core intent of question.

    Args:
        question: The original eval question string.
        answer: The RAG system's generated answer string.
        client: An authenticated OpenAI client instance.
        model: OpenAI model name to use for judging.

    Returns:
        A float in [0.0, 1.0]. Returns 0.0 on parse failure after logging a warning.
    """
    prompt = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _RELEVANCE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
    except Exception:
        _logger.exception("judge.relevance_api_error")
        return 0.0
    return _parse_judge_response(raw)


def score_source_recall(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> float:
    """Compute Jaccard similarity of retrieved source filenames vs. expected.

    Filenames are normalized by stripping the .txt extension before comparison.

    For adversarial cases where expected_sources is empty:
    - Returns 1.0 if retrieved_sources is also empty (correct abstention).
    - Returns 0.0 if retrieved_sources is non-empty (should have abstained).

    Args:
        retrieved_sources: List of source_file values from the pipeline result.
        expected_sources: List of expected source_file values from eval_set.json.

    Returns:
        Jaccard similarity in [0.0, 1.0].
    """

    def _stem(name: str) -> str:
        return name.removesuffix(".txt").strip()

    retrieved = {_stem(s) for s in retrieved_sources if s}
    expected = {_stem(s) for s in expected_sources if s}

    if not expected:
        return 1.0 if not retrieved else 0.0

    intersection = retrieved & expected
    union = retrieved | expected
    return len(intersection) / len(union) if union else 1.0
