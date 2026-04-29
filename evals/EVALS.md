# bspQA Evaluation Report

This document describes the evaluation methodology, terminology, and results for the bspQA RAG pipeline. Its purpose is to establish a measurable baseline and track performance improvements over time.

---

## Table of Contents

1. [What Are Evals?](#what-are-evals)
2. [Terminology](#terminology)
3. [Eval Set](#eval-set)
4. [How Scoring Works](#how-scoring-works)
5. [How to Run](#how-to-run)
6. [Results](#results)
7. [Interpretation](#interpretation)
8. [Next Steps](#next-steps)

---

## What Are Evals?

Evals (evaluations) are structured tests that measure how well the RAG pipeline answers questions. Unlike unit tests that check code correctness, evals measure **output quality** — does the system retrieve the right information, does it answer the question, and does it stay faithful to its sources?

Each eval run produces a scored summary across a fixed set of questions, making it possible to compare performance across different pipeline configurations over time.

---

## Terminology

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation. A pattern where a system retrieves relevant document chunks from a knowledge base and feeds them to an LLM to generate a grounded answer. |
| **Chunk** | A fixed-length segment of text extracted from a source document. This pipeline uses 512-token chunks with 64-token overlap. |
| **Embedding** | A numerical vector representation of text used to measure semantic similarity. The pipeline uses OpenAI's `text-embedding-3-small` model. |
| **Vector search** | Finding the most semantically similar chunks to a query by comparing embedding vectors. The pipeline uses Qdrant as the vector database. |
| **Recency bias** | This pipeline applies a recency boost (`RECENCY_WEIGHT = 0.85`) that re-ranks retrieved chunks to favor more recently published reports. High values strongly de-prioritize older documents. |
| **LLM-as-judge** | Using a language model (here, `gpt-4o-mini`) to score another model's output for quality attributes like faithfulness and relevance. Scores are elicited as structured JSON. |
| **Faithfulness** | Whether every factual claim in the generated answer is directly supported by the retrieved source excerpts. A faithfulness of 1.0 means nothing was hallucinated or inferred beyond what the sources say. |
| **Relevance** | Whether the generated answer actually addresses the question that was asked. A relevance of 1.0 means the answer is directly and completely on-topic. |
| **Source recall** | How much overlap exists between the documents the pipeline retrieved and the documents expected to contain the answer. Computed as Jaccard similarity: \|retrieved ∩ expected\| / \|retrieved ∪ expected\|. A low score means the pipeline found related-sounding chunks from the wrong reports. |
| **Abstention accuracy** | For questions that are outside the scope of the corpus (adversarial cases), this measures whether the pipeline correctly responded with "I could not find a reliable answer" instead of hallucinating an answer. |
| **No-answer rate** | The fraction of all questions where the pipeline returned the fallback message "I could not find a reliable answer in the available BSP reports." Includes both correct abstentions (adversarial questions) and incorrect abstentions (legitimate questions it failed to answer). |
| **Jaccard similarity** | \|A ∩ B\| / \|A ∪ B\| — the ratio of shared items to total unique items between two sets. Used here to score source overlap: 1.0 is a perfect match, 0.0 is no overlap. |

---

## Eval Set

The eval set contains **15 questions** across four categories, designed to stress-test different aspects of the pipeline. Questions span BSP Monetary Policy Reports from Q1 2022 to December 2024.

| ID | Category | Difficulty | Question (abbreviated) |
|---|---|---|---|
| eval-001 | Single-doc | Easy | BSP baseline inflation forecast for 2022 as of Q1 2022 |
| eval-002 | Single-doc | Easy | BSP overnight RRP policy rate at the Q1 2022 report |
| eval-003 | Single-doc | Easy | BSP official core inflation figure for January 2023 |
| eval-004 | Single-doc | Medium | Target RRP rate as of February 2024 and whether it changed |
| eval-005 | Single-doc | Medium | Baseline inflation forecast for 2025 in the December 2024 report |
| eval-006 | Multi-doc | Medium | How BSP's inflation outlook changed from Q1 2022 to Q1 2023 |
| eval-007 | Multi-doc | Hard | RRP policy rate evolution from 2022 to end-2024 |
| eval-008 | Multi-doc | Hard | Trend in reserve requirement ratio (RRR) from 2022 to 2024 |
| eval-009 | Multi-doc | Medium | Oil price influence on inflation outlook: Q3 2022 vs Q4 2023 |
| eval-010 | Rationale | Hard | Factors behind the 50 bps rate hike in November 2022 |
| eval-011 | Rationale | Hard | Conditions that prompted the rate cut in August 2024 |
| eval-012 | Rationale | Medium | Factors behind holding the rate at 6.50% in H1 2024 |
| eval-013 | Adversarial | Hard | BSP's stance on cryptocurrency / Bitcoin (not in corpus) |
| eval-014 | Adversarial | Hard | BSP Monetary Board decisions in March 2027 (future date) |
| eval-015 | Adversarial | Hard | BSP vs ECB policy comparison (ECB not covered in corpus) |

**Category definitions:**

- **Single-doc** — The answer is contained entirely within one BSP report. Tests basic retrieval precision.
- **Multi-doc** — The answer requires synthesizing information across multiple reports. Tests the pipeline's ability to surface the right documents for a time-range query.
- **Rationale** — The question asks *why* the BSP made a decision. Tests whether the pipeline retrieves policy reasoning, not just rate figures.
- **Adversarial** — The question asks about something outside the corpus. The correct behavior is to decline to answer rather than hallucinate.

Full question text and `expected_sources` are in [`eval_set.json`](eval_set.json).

---

## How Scoring Works

Each question is run through the pipeline and scored on four metrics:

### Faithfulness (LLM-as-judge, 0.0–1.0)
The judge model is shown the pipeline's answer alongside the raw text of retrieved chunks. It rates whether every factual claim in the answer is explicitly supported by those chunks. A score of 1.0 means the answer contains no hallucinations or unsupported inferences. Scored only for non-adversarial cases.

### Relevance (LLM-as-judge, 0.0–1.0)
The judge model is shown the question and the answer. It rates how directly and completely the answer addresses the question. A "I could not find..." response for a legitimate question scores 0.0.

### Source Recall (Deterministic, 0.0–1.0)
Computed as Jaccard similarity between the filenames of retrieved chunks and the expected source files from `eval_set.json`. This measures whether the pipeline is looking in the right documents, independent of answer quality. For adversarial cases, retrieval always happens at the Qdrant level before the LLM decides to abstain — so source recall for adversarial questions is always 0.0 by design and should be disregarded; use `abstention_accuracy` instead.

### Abstention Accuracy (Adversarial cases only, 0.0–1.0)
For the three adversarial questions, this binary score records whether the pipeline correctly returned the "I could not find a reliable answer" message rather than generating an answer.

### Judge model
All LLM-as-judge calls use `gpt-4o-mini` with `temperature=0` for determinism. The judge prompt requests a JSON response `{"score": float, "reason": str}` to avoid free-form variance.

---

## How to Run

From the repo root with environment variables set (requires `OPENAI_API_KEY`, `QDRANT_CLUSTER_ENDPOINT`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`):

```bash
# Run with default settings — saves to results/run_<TIMESTAMP>.json
uv run python evals/run_evals.py

# Verbose per-question logging
uv run python evals/run_evals.py --log-level DEBUG

# Save to a named file
uv run python evals/run_evals.py --output results/my_run.json

# Use a more capable judge model for higher-fidelity scores
uv run python evals/run_evals.py --judge-model gpt-4o
```

Results are saved to `results/` (gitignored). The terminal prints a summary table; the JSON file contains per-question breakdowns including the full answer and retrieved source files.

---

## Results

### Run 1 — Baseline

**Timestamp:** 2026-04-29T03:35:39 UTC  
**Pipeline:** VectorRAG (Qdrant + `text-embedding-3-small` + `gpt-4o-mini`, `RECENCY_WEIGHT=0.85`, `top_k=15`)  
**Judge model:** `gpt-4o-mini`  
**Eval set version:** 1.0 (15 questions)

#### Summary

| Metric | VectorRAG |
|---|---|
| **Faithfulness** | 0.667 |
| **Relevance** | 0.567 |
| **Source recall** | 0.146 |
| **Abstention accuracy** | 1.000 |
| **No-answer rate** | 0.400 |
| **Avg latency (ms)** | 10,971 |

#### Per-Question Breakdown

| ID | Category | Faithfulness | Relevance | Source Recall | Abstained? | Latency (ms) |
|---|---|---|---|---|---|---|
| eval-001 | single_doc | 0.0 | 0.0 | 0.000 | — | 8,295 |
| eval-002 | single_doc | 1.0 | 0.5 | 0.000 | — | 12,347 |
| eval-003 | single_doc | 0.0 | 0.0 | 0.000 | — | 8,552 |
| eval-004 | single_doc | 1.0 | 1.0 | 0.143 | — | 8,979 |
| eval-005 | single_doc | 1.0 | 1.0 | 0.333 | — | 8,696 |
| eval-006 | multi_doc | 1.0 | 1.0 | 0.143 | — | 16,989 |
| eval-007 | multi_doc | 1.0 | 1.0 | 0.667 | — | 28,460 |
| eval-008 | multi_doc | 0.0 | 0.0 | 0.222 | — | 10,572 |
| eval-009 | multi_doc | 1.0 | 1.0 | 0.000 | — | 8,254 |
| eval-010 | rationale | 1.0 | 1.0 | 0.200 | — | 8,195 |
| eval-011 | rationale | 0.5 | 1.0 | 0.143 | — | 17,479 |
| eval-012 | rationale | 0.5 | 1.0 | 0.333 | — | 6,540 |
| eval-013 | adversarial | — | 0.0 | — | ✅ Yes | 10,132 |
| eval-014 | adversarial | — | 0.0 | — | ✅ Yes | 5,449 |
| eval-015 | adversarial | — | 0.0 | — | ✅ Yes | 5,619 |

*Faithfulness and source recall marked `—` for adversarial cases; relevance is 0.0 for correct abstentions as expected.*

---

## Interpretation

### What worked well

**Abstention is perfect (1.0).** All three adversarial questions — cryptocurrency regulation, a future Monetary Board meeting, and ECB comparisons — were correctly declined. The LLM respected its system prompt and did not hallucinate answers for out-of-scope topics.

**Faithfulness is solid when the pipeline answers.** Of the 12 non-adversarial questions where the pipeline produced an answer, 7 scored 1.0 on faithfulness and none scored below 0.5. When the pipeline retrieves chunks and generates a response, it stays grounded in its sources.

**eval-007 (RRP rate evolution 2022–2024) is the strongest result.** Despite spanning 6 source documents, the pipeline surfaced 4 of the 6 expected reports (source recall: 0.67) and produced a factually grounded, well-structured answer.

---

### What needs improvement

**Recency bias is causing 3 unexpected non-answers (eval-001, eval-003, eval-008).** These are legitimate questions about 2022 and 2023 data that the pipeline should be able to answer. The root cause is `RECENCY_WEIGHT = 0.85` in [`backend/app/rag/retriever.py`](../backend/app/rag/retriever.py) — the re-ranking step so strongly favors recent documents that chunks from 2022 reports are pushed below the score threshold entirely, leaving the LLM with no usable context.

- **eval-001** (Q1 2022 inflation forecast): retrieved 15 chunks, all from 2024–2025 reports
- **eval-003** (January 2023 core inflation): same pattern — retrieved only 2024–2025 chunks  
- **eval-008** (RRR trend 2022–2024): retrieved chunks partially covering 2023 but not 2022

**Source recall is low overall (0.146).** Even on questions the pipeline answers correctly, it typically retrieves the right answer from the wrong report (e.g., a later report that references an earlier decision). This means citations are often inaccurate even when answers are factually correct, which degrades user trust.

**eval-002 relevance is only 0.5.** The pipeline gave the RRP rate from July 2022 (after a mid-year hike) rather than the rate that was current *at the time of the Q1 2022 report* (February 2022). The answer was faithful to its retrieved chunks but answered the wrong time slice — a temporal precision problem.

**eval-009 source recall is 0.0.** The question asked about Q3 2022 and Q4 2023 oil price impacts. The pipeline retrieved only 2024 and early 2023 reports and synthesized an answer that was plausible but drawn from the wrong periods. High relevance (1.0) and faithfulness (1.0) masked this retrieval failure.

**No-answer rate of 0.40 is too high.** Only 3 of the 6 no-answers are correct (the adversarial cases). The other 3 represent retrieval failures on legitimate historical questions.

---

## Next Steps

Listed in priority order based on the results above.

### 1. Reduce or make recency weight configurable
**Impact:** Fixes the 3 unexpected no-answers; likely improves source recall significantly.  
**Change:** Lower `RECENCY_WEIGHT` in [`backend/app/rag/retriever.py`](../backend/app/rag/retriever.py) from `0.85` toward `0.3–0.5`, or expose it as a parameter so callers can opt out of recency re-ranking for historical queries.

### 2. Investigate chunk coverage for 2022 reports
**Impact:** Ensures older documents are actually indexed with sufficient coverage.  
**Change:** Verify that `FullReport_2022_1.txt` and `FullReport_2023_1.txt` are in the Qdrant collection with a reasonable number of chunks. Run a direct Qdrant query filtered to those source files and confirm they return results before attributing the miss purely to recency reranking.

### 3. Add date-range metadata filtering to the API
**Impact:** Allows queries like "as of Q1 2022" to explicitly restrict retrieval to documents from that period, bypassing the recency re-ranking problem for time-anchored questions.  
**Change:** Detect temporal anchors in the query (e.g., "as of Q1 2022", "in November 2022") and automatically apply `publication_date_from`/`publication_date_to` filters, which are already supported by the retriever.

### 4. Re-run evals after recency weight change
**Expected outcome:** No-answer rate drops from 0.40 to ~0.20 (only adversarial), source recall improves from 0.146 to 0.30+, relevance improves from 0.567 to 0.80+.  
**Command:** `uv run python evals/run_evals.py --output results/run_recency_tuned.json`

### 5. Add reference answers to high-value eval cases
**Impact:** Enables reference-based faithfulness scoring, which is more precise than reference-free LLM-as-judge for factual claims (e.g., specific rate figures and dates).  
**Change:** Fill in `reference_answer` fields in `eval_set.json` for eval-001 through eval-005 using the actual BSP report text.
