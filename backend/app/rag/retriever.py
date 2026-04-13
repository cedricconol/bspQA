"""Query Qdrant and print retrieved chunks for a text query."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.0


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    return _backend_root().parent


def _load_dotenv_files() -> None:
    load_dotenv(_repo_root() / ".env")
    load_dotenv(_backend_root() / ".env")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Missing required environment variable: {name}", file=sys.stderr)
        raise SystemExit(1)
    return value


def _openai_client() -> OpenAI:
    return OpenAI(api_key=_require_env("OPENAI_API_KEY"))


def _qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=_require_env("QDRANT_CLUSTER_ENDPOINT"),
        api_key=_require_env("QDRANT_API_KEY"),
        timeout=30.0,
    )


def _qdrant_collection_name() -> str:
    return _require_env("QDRANT_COLLECTION_NAME")


def _embed_query(client: OpenAI, query: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    return response.data[0].embedding


def retrieve_and_print(query: str, top_k: int, score_threshold: float) -> int:
    _load_dotenv_files()
    openai_client = _openai_client()
    qdrant_client = _qdrant_client()
    query_vector = _embed_query(openai_client, query)

    hits = qdrant_client.query_points(
        collection_name=_qdrant_collection_name(),
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
    ).points

    if not hits:
        print("No chunks returned from Qdrant.")
        return 0

    print(f"Returned {len(hits)} chunk(s):")
    for idx, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        text = str(payload.get("text", "")).strip()
        source = payload.get("source_file", "unknown")
        chunk_index = payload.get("chunk_index", "unknown")
        print(f"\n[{idx}] score={hit.score:.4f} source={source} chunk_index={chunk_index}")
        print(text)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Qdrant and print chunks.")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return retrieve_and_print(
        query=args.query,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
