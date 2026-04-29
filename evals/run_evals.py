"""CLI eval runner for bspQA RAG pipeline evaluation.

Usage:
    uv run python evals/run_evals.py
    uv run python evals/run_evals.py --output results/my_run.json
    uv run python evals/run_evals.py --log-level DEBUG
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `backend.*` and `evals.*` resolve correctly
# regardless of which directory the script is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

_logger = logging.getLogger(__name__)

NO_ANSWER_SENTINEL = "I could not find a reliable answer"


# ---------------------------------------------------------------------------
# Pipeline abstractions
# ---------------------------------------------------------------------------


class BasePipeline(ABC):
    """Abstract interface for a RAG pipeline under evaluation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable pipeline name used in reports."""

    @abstractmethod
    def query(self, question: str) -> dict[str, Any]:
        """Run the pipeline on a question and return a standardized result dict.

        Args:
            question: The natural language question to answer.

        Returns:
            A dict with keys:
                ``answer`` (str): The generated answer text.
                ``sources`` (list[dict]): Each element must have a ``source_file`` key.
                ``chunks_text`` (str): Raw concatenated chunk text used for generation,
                    needed for faithfulness scoring.
        """


class VectorRAGPipeline(BasePipeline):
    """Wraps the current Qdrant-based RAG pipeline.

    Calls retrieve_chunks() directly to capture raw chunk text for faithfulness
    scoring, then delegates to run_rag_pipeline() for answer generation.
    """

    def __init__(self) -> None:
        _load_dotenv()

    @property
    def name(self) -> str:
        return "VectorRAG"

    def query(self, question: str) -> dict[str, Any]:
        """Query the vector RAG pipeline and return answer + raw chunk text.

        Args:
            question: The natural language question to answer.

        Returns:
            Standardized result dict with answer, sources, and chunks_text.
        """
        from backend.app.rag.pipeline import run_rag_pipeline
        from backend.app.rag.retriever import retrieve_chunks

        hits = retrieve_chunks(query=question)
        chunks_text = "\n\n".join(
            f"[{i+1}] {hit.payload.get('source_file','')}: {hit.payload.get('text','')}"
            for i, hit in enumerate(hits)
            if hit.payload
        )
        result = run_rag_pipeline(question)
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "chunks_text": chunks_text,
        }


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """A single eval question with ground-truth metadata."""

    id: str
    question: str
    expected_sources: list[str]
    reference_answer: str | None
    difficulty: str
    category: str


@dataclass
class EvalResult:
    """Scored result for one pipeline run on one eval case."""

    eval_id: str
    pipeline: str
    question: str
    answer: str
    retrieved_source_files: list[str]
    expected_sources: list[str]
    faithfulness: float | None
    relevance: float
    source_recall: float
    correctly_abstained: bool | None
    latency_ms: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Core eval logic
# ---------------------------------------------------------------------------


def _is_no_answer(answer: str) -> bool:
    return NO_ANSWER_SENTINEL.lower() in answer.lower()


def run_single_eval(
    case: EvalCase,
    pipeline: BasePipeline,
    openai_client: Any,
    *,
    judge_model: str = "gpt-4o-mini",
) -> EvalResult:
    """Execute one pipeline query on one eval case and return scored results.

    Args:
        case: The eval case to run.
        pipeline: The pipeline implementation to query.
        openai_client: OpenAI client instance for LLM-as-judge calls.
        judge_model: Model name for LLM-as-judge scoring.

    Returns:
        A fully populated EvalResult.
    """
    from scoring import score_faithfulness, score_relevance, score_source_recall

    answer = ""
    sources: list[dict[str, Any]] = []
    chunks_text = ""
    error: str | None = None

    start = time.monotonic()
    try:
        result = pipeline.query(case.question)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        chunks_text = result.get("chunks_text", "")
    except Exception as exc:
        _logger.error("eval.pipeline_error", exc_info=True, extra={"eval_id": case.id})
        error = str(exc)
    latency_ms = round((time.monotonic() - start) * 1000, 1)

    retrieved_source_files = [s.get("source_file", "") for s in sources if s.get("source_file")]

    is_adversarial = len(case.expected_sources) == 0
    correctly_abstained: bool | None = _is_no_answer(answer) if is_adversarial else None

    source_recall = score_source_recall(retrieved_source_files, case.expected_sources)
    relevance = score_relevance(case.question, answer, openai_client, model=judge_model)

    faithfulness: float | None = None
    if not is_adversarial:
        if chunks_text:
            faithfulness = score_faithfulness(answer, chunks_text, openai_client, model=judge_model)
        else:
            faithfulness = 0.0

    return EvalResult(
        eval_id=case.id,
        pipeline=pipeline.name,
        question=case.question,
        answer=answer,
        retrieved_source_files=retrieved_source_files,
        expected_sources=case.expected_sources,
        faithfulness=faithfulness,
        relevance=relevance,
        source_recall=source_recall,
        correctly_abstained=correctly_abstained,
        latency_ms=latency_ms,
        error=error,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _build_summary(results: list[EvalResult]) -> dict[str, dict[str, float]]:
    """Aggregate per-result scores into per-pipeline summary statistics.

    Args:
        results: All EvalResult objects from a run.

    Returns:
        Dict mapping pipeline name to metric name to aggregated float value.
    """
    from collections import defaultdict

    buckets: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        buckets[r.pipeline].append(r)

    summary: dict[str, dict[str, float]] = {}
    for pipeline_name, pipeline_results in buckets.items():
        faith_scores = [r.faithfulness for r in pipeline_results if r.faithfulness is not None]
        abstention_cases = [r for r in pipeline_results if r.correctly_abstained is not None]

        summary[pipeline_name] = {
            "faithfulness": round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else 0.0,
            "relevance": round(
                sum(r.relevance for r in pipeline_results) / len(pipeline_results), 3
            ),
            "source_recall": round(
                sum(r.source_recall for r in pipeline_results) / len(pipeline_results), 3
            ),
            "abstention_accuracy": round(
                sum(1 for r in abstention_cases if r.correctly_abstained) / len(abstention_cases),
                3,
            ) if abstention_cases else 0.0,
            "no_answer_rate": round(
                sum(1 for r in pipeline_results if _is_no_answer(r.answer)) / len(pipeline_results),
                3,
            ),
            "avg_latency_ms": round(
                sum(r.latency_ms for r in pipeline_results) / len(pipeline_results), 1
            ),
        }
    return summary


def _print_comparison_table(summary: dict[str, dict[str, float]]) -> None:
    """Print a formatted side-by-side comparison table to stdout.

    Args:
        summary: Output of _build_summary().
    """
    pipelines = list(summary.keys())
    metrics = [
        "faithfulness",
        "relevance",
        "source_recall",
        "abstention_accuracy",
        "no_answer_rate",
        "avg_latency_ms",
    ]

    col_w = max(20, max(len(p) for p in pipelines) + 2)
    label_w = 22

    header = f"{'Metric':<{label_w}}" + "".join(f"{p:<{col_w}}" for p in pipelines)
    print()
    print(header)
    print("-" * (label_w + col_w * len(pipelines)))
    for metric in metrics:
        row = f"{metric:<{label_w}}"
        for p in pipelines:
            val = summary.get(p, {}).get(metric, 0.0)
            row += f"{val:<{col_w}}"
        print(row)
    print()


def _save_results(
    results: list[EvalResult],
    summary: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Serialize results and summary to a JSON file.

    Args:
        results: Per-question eval results.
        summary: Per-pipeline aggregated metrics.
        output_path: Destination path; parent directories are created if needed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "per_question": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _logger.info("results.saved", extra={"path": str(output_path)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    """Load .env files from repo root and backend/ into the environment."""
    try:
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parents[1]
        load_dotenv(root / ".env")
        load_dotenv(root / "backend" / ".env")
    except ImportError:
        pass


def _load_eval_cases(path: Path) -> list[EvalCase]:
    """Load and parse eval cases from a JSON file.

    Args:
        path: Path to eval_set.json.

    Returns:
        List of EvalCase instances.

    Raises:
        SystemExit: If the file cannot be read or parsed.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [EvalCase(**c) for c in data["cases"]]
    except (FileNotFoundError, KeyError, TypeError) as exc:
        _logger.error("eval_set.load_failed: %s", exc)
        sys.exit(1)


def _build_openai_client() -> Any:
    """Construct an OpenAI client using the OPENAI_API_KEY environment variable.

    Returns:
        An authenticated OpenAI client instance.

    Raises:
        SystemExit: If OPENAI_API_KEY is not set.
    """
    import os

    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        _logger.error("OPENAI_API_KEY is not set. Cannot run LLM-as-judge scoring.")
        sys.exit(1)
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args, run evals, print comparison table, and save results."""
    parser = argparse.ArgumentParser(
        description="Run bspQA eval suite and compare pipeline performance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output JSON path. Defaults to results/run_<TIMESTAMP>.json.",
    )
    parser.add_argument(
        "--eval-set",
        default=str(Path(__file__).parent / "eval_set.json"),
        metavar="PATH",
        help="Path to eval_set.json.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="OpenAI model for LLM-as-judge scoring.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    _load_dotenv()

    pipelines: list[BasePipeline] = [VectorRAGPipeline()]

    cases = _load_eval_cases(Path(args.eval_set))
    _logger.info("Loaded %d eval cases", len(cases))

    openai_client = _build_openai_client()

    all_results: list[EvalResult] = []
    for pipeline in pipelines:
        _logger.info("Running pipeline: %s", pipeline.name)
        for case in cases:
            _logger.info("  [%s] %s", case.id, case.question[:70])
            result = run_single_eval(
                case,
                pipeline,
                openai_client,
                judge_model=args.judge_model,
            )
            all_results.append(result)
            _logger.debug(
                "  scores: faithfulness=%s relevance=%.2f recall=%.2f latency=%.0fms",
                f"{result.faithfulness:.2f}" if result.faithfulness is not None else "n/a",
                result.relevance,
                result.source_recall,
                result.latency_ms,
            )

    summary = _build_summary(all_results)
    _print_comparison_table(summary)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else Path("results") / f"run_{ts}.json"
    _save_results(all_results, summary, output_path)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
