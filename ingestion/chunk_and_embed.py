"""Chunk parsed .txt files, embed with OpenAI, and upsert vectors into Qdrant."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

# RAG / chunking defaults (token counts use the embedding model tokenizer)
CHUNK_SIZE_TOKENS = 512
OVERLAP_TOKENS = 64
TOP_K_RETRIEVAL = 5
EMBEDDING_MODEL = "text-embedding-3-small"
# Default output size for text-embedding-3-small (must match Qdrant collection)
EMBEDDING_DIM = 1536
# cl100k_base matches common OpenAI chat/embedding tokenization
ENCODING_NAME = "cl100k_base"

EMBED_BATCH_SIZE = 128
# Stable IDs for the same logical chunk across re-ingestion runs
POINT_ID_NAMESPACE = uuid.NAMESPACE_URL
_POINT_ID_PREFIX = "https://bspqa.local/ingestion/chunk"


def _ingestion_root() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    return _ingestion_root().parent


def _parsed_dir() -> Path:
    return _ingestion_root() / "data" / "parsed"


def _embeddings_dir() -> Path:
    return _ingestion_root() / "data" / "embeddings"


def _manifest_path() -> Path:
    return _ingestion_root() / "manifest.json"


def _load_dotenv_files() -> None:
    load_dotenv(_repo_root() / ".env")
    load_dotenv(_ingestion_root() / ".env")
    load_dotenv(_repo_root() / "backend" / ".env")


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        logger.error("Missing required environment variable: %s", name)
        raise SystemExit(1)
    return v


def _openai_client() -> OpenAI:
    return OpenAI(api_key=_require_env("OPENAI_API_KEY"))


def _qdrant_client() -> QdrantClient:
    url = _require_env("QDRANT_CLUSTER_ENDPOINT")
    api_key = _require_env("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key, timeout=120.0)


def _qdrant_collection_name() -> str:
    return _require_env("QDRANT_COLLECTION_NAME")


def _normalize_period_label(period_label: str | None) -> str | None:
    if not period_label:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", period_label.strip().lower()).strip("_")
    return normalized or None


def _normalize_publication_date(publication_date: str | None) -> str | None:
    if not publication_date:
        return None
    candidate = publication_date.strip()
    if not candidate or candidate.lower() == "null":
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
        return candidate
    return None


def _manifest_doc_metadata_by_source_file() -> dict[str, dict[str, str | None]]:
    """Build a lookup keyed by parsed txt filename (e.g. ``FullReport_2022_1.txt``).

    Returns:
        A dict mapping source filenames to their period_label, period_label_key,
        and publication_date metadata from manifest.json.
    """
    path = _manifest_path()
    if not path.is_file():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    documents = data.get("documents", [])
    by_source_file: dict[str, dict[str, str | None]] = {}

    for doc in documents:
        if not isinstance(doc, dict):
            continue
        url = doc.get("url")
        if not isinstance(url, str):
            continue
        pdf_name = Path(url).name
        source_file = f"{Path(pdf_name).stem}.txt"
        by_source_file[source_file] = {
            "period_label": doc.get("period_label"),
            "period_label_key": _normalize_period_label(doc.get("period_label")),
            "publication_date": _normalize_publication_date(doc.get("publication_date")),
        }

    return by_source_file


def _is_production_environment() -> bool:
    """Return True when running in a production-like deploy (e.g. Railway production)."""
    railway = (os.environ.get("RAILWAY_ENVIRONMENT") or "").strip().lower()
    if railway == "production":
        return True
    app_env = (
        os.environ.get("ENVIRONMENT")
        or os.environ.get("APP_ENV")
        or ""
    ).strip().lower()
    return app_env in ("production", "prod")


def _should_recreate_qdrant_collection() -> bool:
    """Return whether the Qdrant collection should be dropped and recreated before ingest.

    Always returns False in production. In non-production environments, returns True
    when ENVIRONMENT/APP_ENV looks like local dev, or when QDRANT_RECREATE_COLLECTION
    is explicitly set to a truthy value.

    Returns:
        True if the collection should be recreated, False otherwise.
    """
    if _is_production_environment():
        return False

    explicit = (os.environ.get("QDRANT_RECREATE_COLLECTION") or "").strip().lower()
    if explicit in ("1", "true", "yes", "on"):
        return True
    if explicit in ("0", "false", "no", "off"):
        return False

    app_env = (
        os.environ.get("ENVIRONMENT")
        or os.environ.get("APP_ENV")
        or ""
    ).strip().lower()
    return app_env in ("development", "dev", "local", "test")


def _ensure_source_file_keyword_index(client: QdrantClient, collection: str) -> None:
    """Create payload indexes used by retrieval and deletes."""
    info = client.get_collection(collection_name=collection)
    schema = info.payload_schema or {}

    if "source_file" not in schema:
        client.create_payload_index(
            collection_name=collection,
            field_name="source_file",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
    if "period_label_key" not in schema:
        client.create_payload_index(
            collection_name=collection,
            field_name="period_label_key",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
    if "publication_date" not in schema:
        client.create_payload_index(
            collection_name=collection,
            field_name="publication_date",
            field_schema=models.PayloadSchemaType.DATETIME,
            wait=True,
        )


def _ensure_collection(
    client: QdrantClient,
    name: str,
    *,
    recreate: bool = False,
) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(collection_name=name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )
    _ensure_source_file_keyword_index(client, name)


def _delete_chunks_for_source(
    client: QdrantClient,
    collection: str,
    source_file: str,
) -> None:
    client.delete(
        collection_name=collection,
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_file",
                    match=models.MatchValue(value=source_file),
                ),
            ],
        ),
        wait=True,
    )


def _encode_for_chunking(text: str, enc: tiktoken.Encoding) -> list[int]:
    return enc.encode(text, disallowed_special=())


def chunk_text_by_tokens(
    text: str,
    enc: tiktoken.Encoding,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[str]:
    """Split text into overlapping token windows.

    Args:
        text: The raw text to chunk.
        enc: A tiktoken Encoding instance.
        chunk_size: Maximum number of tokens per chunk.
        overlap: Number of tokens shared between adjacent chunks.

    Returns:
        A list of UTF-8 decoded text chunks.

    Raises:
        ValueError: If overlap is >= chunk_size.
    """
    text = text.strip()
    if not text:
        return []

    tokens = _encode_for_chunking(text, enc)
    if not tokens:
        return []

    if len(tokens) <= chunk_size:
        return [enc.decode(tokens)]

    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        chunks.append(enc.decode(window))
        if start + chunk_size >= len(tokens):
            break
        start += stride

    return chunks


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    data = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in data]


def _point_id(source_file: str, chunk_index: int) -> str:
    return str(
        uuid.uuid5(
            POINT_ID_NAMESPACE,
            f"{_POINT_ID_PREFIX}/{source_file}/{chunk_index}",
        )
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _load_dotenv_files()

    parsed = _parsed_dir()
    if not parsed.is_dir():
        logger.error("Parsed directory not found: %s", parsed)
        return 1

    txt_files = sorted(parsed.glob("*.txt"))
    if not txt_files:
        logger.error("No .txt files in %s", parsed)
        return 1

    enc = tiktoken.get_encoding(ENCODING_NAME)
    openai_client = _openai_client()
    qdrant = _qdrant_client()
    collection = _qdrant_collection_name()
    recreate = _should_recreate_qdrant_collection()
    if recreate:
        logger.warning(
            "Qdrant collection will be recreated (dev / explicit flag); wipe+create %r",
            collection,
        )
    _ensure_collection(qdrant, collection, recreate=recreate)

    out_dir = _embeddings_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    params_path = out_dir / "rag_params.json"
    params_path.write_text(
        json.dumps(
            {
                "chunk_size_tokens": CHUNK_SIZE_TOKENS,
                "overlap_tokens": OVERLAP_TOKENS,
                "top_k_retrieval": TOP_K_RETRIEVAL,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": EMBEDDING_DIM,
                "tokenizer": ENCODING_NAME,
                "qdrant_collection": collection,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    total_upserted = 0
    manifest_by_source_file = _manifest_doc_metadata_by_source_file()

    for path in txt_files:
        _delete_chunks_for_source(qdrant, collection, path.name)
        doc_meta = manifest_by_source_file.get(path.name, {})

        raw = path.read_text(encoding="utf-8")
        chunks = chunk_text_by_tokens(
            raw,
            enc,
            chunk_size=CHUNK_SIZE_TOKENS,
            overlap=OVERLAP_TOKENS,
        )
        logger.info("%s: %d chunk(s)", path.name, len(chunks))

        pending_texts: list[str] = []
        pending_indices: list[int] = []

        def flush_batch() -> None:
            nonlocal total_upserted
            if not pending_texts:
                return
            embeddings = _embed_batch(openai_client, pending_texts)
            points: list[models.PointStruct] = []
            for idx, emb, chunk_text in zip(
                pending_indices, embeddings, pending_texts, strict=True
            ):
                if len(emb) != EMBEDDING_DIM:
                    logger.error(
                        "Embedding length %d != expected %d; check model / Qdrant collection config.",
                        len(emb),
                        EMBEDDING_DIM,
                    )
                    raise SystemExit(1)
                points.append(
                    models.PointStruct(
                        id=_point_id(path.name, idx),
                        vector=emb,
                        payload={
                            "source_file": path.name,
                            "chunk_index": idx,
                            "text": chunk_text,
                            "embedding_model": EMBEDDING_MODEL,
                            "period_label": doc_meta.get("period_label"),
                            "period_label_key": doc_meta.get("period_label_key"),
                            "publication_date": doc_meta.get("publication_date"),
                        },
                    )
                )
            qdrant.upsert(collection_name=collection, points=points, wait=True)
            total_upserted += len(points)
            pending_texts.clear()
            pending_indices.clear()

        for i, chunk in enumerate(chunks):
            pending_texts.append(chunk)
            pending_indices.append(i)
            if len(pending_texts) >= EMBED_BATCH_SIZE:
                flush_batch()

        flush_batch()

    logger.info("Upserted %d point(s) into Qdrant collection %r", total_upserted, collection)
    logger.info("Wrote params to %s", params_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
