"""Essential unit tests for RAG pipeline orchestration."""

from __future__ import annotations

from types import SimpleNamespace

from backend.app.rag import pipeline


def test_run_rag_pipeline_returns_fallback_when_no_hits(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "retrieve_chunks", lambda **_: [])

    def _unexpected_client() -> None:
        raise AssertionError("OpenAI client should not be created when no hits exist")

    monkeypatch.setattr(pipeline, "get_openai_client", _unexpected_client)

    result = pipeline.run_rag_pipeline("What did BSP say about inflation?")

    assert result == {
        "answer": "I could not find a reliable answer in the available BSP reports.",
        "sources": [],
    }


def test_run_rag_pipeline_generates_answer_from_retrieved_hits(monkeypatch) -> None:
    hit = SimpleNamespace(
        payload={
            "text": "Inflation eased in Q4.",
            "source_file": "FullReport_December2025.txt",
            "chunk_index": 12,
        },
        score=0.9876,
    )
    monkeypatch.setattr(pipeline, "retrieve_chunks", lambda **_: [hit])

    calls: list[dict[str, str]] = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="Answer: Inflation eased. [Source 1]")

    fake_client = SimpleNamespace(responses=FakeResponses())
    result = pipeline.run_rag_pipeline(
        "Did inflation ease?",
        openai_client=fake_client,
    )

    assert result["answer"] == "Answer: Inflation eased. [Source 1]"
    assert result["sources"] == [
        {
            "source_id": 1,
            "source_file": "FullReport_December2025.txt",
            "chunk_index": 12,
            "score": 0.9876,
        }
    ]
    assert len(calls) == 1
    assert calls[0]["model"] == pipeline.DEFAULT_GENERATION_MODEL
    assert calls[0]["instructions"] == pipeline.SYSTEM_PROMPT
    assert "Question:\nDid inflation ease?" in calls[0]["input"]
    assert "[Source 1]" in calls[0]["input"]
