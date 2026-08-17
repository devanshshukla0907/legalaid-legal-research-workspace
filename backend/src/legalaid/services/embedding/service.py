"""Embedding service.

Separate from the LLM service on purpose: it uses a different model, has a very different
batching profile, and sits on both the write path (ingestion) and the read path (query
embedding). Keeping it distinct means the Embeddings page can report real per-request figures.

Batching matters more than it looks. Ollama embedding calls carry roughly 2.2 s of fixed
per-request overhead that barely varies with input length, so cost is driven by request count,
not text size. Batching at 32 measured ~25x faster per chunk than one-at-a-time, and produces
identical vectors (cosine 1.000000). Hence `embedding_batch_size` and the batched
`/api/embed` endpoint are load-bearing, not an optimisation detail.
"""

from __future__ import annotations

import math
import time

from sqlalchemy import select

from legalaid.capabilities import require
from legalaid.config import get_settings
from legalaid.core.logging import get_logger
from legalaid.core.tokens import estimate_tokens
from legalaid.core.transport import register_endpoint
from legalaid.db.base import session_scope, utcnow
from legalaid.db.models import Chunk, EmbeddingMetadata
from legalaid.domain.contracts import (
    EmbedChunksRequest,
    EmbedChunksResponse,
    EmbeddingRecord,
    EmbedTextsRequest,
    EmbedTextsResponse,
    ServiceHealth,
)
from legalaid.domain.enums import Collection
from legalaid.services.llm.ollama_client import OllamaClient
from legalaid.services.vector.store import get_store

logger = get_logger(__name__)

SERVICE_NAME = "embedding-service"

PREVIEW_VALUES = 12


def _preview(vector: list[float]) -> list[float]:
    """First few components, shown in the UI so a vector is inspectable without dumping 768 floats."""
    return [round(float(v), 6) for v in vector[:PREVIEW_VALUES]]


def _norm(vector: list[float]) -> float:
    return float(math.sqrt(sum(float(v) * float(v) for v in vector)))


async def embed_texts_raw(
    texts: list[str], *, model: str | None = None
) -> tuple[list[list[float]], float, str]:
    """Embed texts in configured batches. Returns (vectors, elapsed_ms, model_name)."""
    settings = get_settings()
    client = OllamaClient()
    model_name = model or settings.embedding_model
    batch_size = max(1, settings.embedding_batch_size)

    vectors: list[list[float]] = []
    started = time.perf_counter()
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        vectors.extend(await client.embed(batch, model=model_name))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return vectors, elapsed_ms, model_name


@register_endpoint(SERVICE_NAME, "POST", "/embeddings/document", EmbedTextsResponse, EmbedTextsRequest)
async def embed_document_texts(request: EmbedTextsRequest) -> EmbedTextsResponse:
    """Embed document/knowledge text for indexing."""
    return await _embed_texts(request, default_purpose="document")


@register_endpoint(SERVICE_NAME, "POST", "/embeddings/query", EmbedTextsResponse, EmbedTextsRequest)
async def embed_query_texts(request: EmbedTextsRequest) -> EmbedTextsResponse:
    """Embed a user question for similarity search."""
    return await _embed_texts(request, default_purpose="query")


async def _embed_texts(request: EmbedTextsRequest, *, default_purpose: str) -> EmbedTextsResponse:
    require("embeddings")
    settings = get_settings()
    purpose = request.purpose or default_purpose

    if not request.texts:
        return EmbedTextsResponse(
            provider=settings.embedding_provider,
            model=request.model or settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            purpose=purpose,
            total_latency_ms=0.0,
            embeddings=[],
            vectors=[],
        )

    vectors, elapsed_ms, model_name = await embed_texts_raw(request.texts, model=request.model)
    # Per-item latency is the batch cost shared out; the batch total is reported separately so
    # the UI never implies each vector cost the full request time.
    per_item_ms = elapsed_ms / len(vectors) if vectors else 0.0

    records = [
        EmbeddingRecord(
            index=index,
            dimensions=len(vector),
            norm=_norm(vector),
            latency_ms=per_item_ms,
            input_chars=len(request.texts[index]),
            input_tokens=estimate_tokens(request.texts[index]),
            preview=_preview(vector),
            vector=vector if purpose == "query" else None,
        )
        for index, vector in enumerate(vectors)
    ]

    logger.info(
        "embed_texts",
        purpose=purpose,
        count=len(vectors),
        model=model_name,
        total_ms=round(elapsed_ms, 1),
        per_item_ms=round(per_item_ms, 1),
    )

    return EmbedTextsResponse(
        provider=settings.embedding_provider,
        model=model_name,
        dimensions=len(vectors[0]) if vectors else settings.embedding_dimensions,
        purpose=purpose,
        total_latency_ms=elapsed_ms,
        embeddings=records,
        vectors=vectors,
    )


@register_endpoint(SERVICE_NAME, "POST", "/embeddings/chunks", EmbedChunksResponse, EmbedChunksRequest)
async def embed_chunks(request: EmbedChunksRequest) -> EmbedChunksResponse:
    """Embed stored chunks, upsert them into the collection index and record metadata."""
    require("embeddings")
    require("vector_index")
    settings = get_settings()

    with session_scope() as session:
        rows = session.scalars(select(Chunk).where(Chunk.id.in_(request.chunk_ids))).all()
        pending = [(row.id, row.text, dict(row.chunk_metadata or {}), row.collection) for row in rows]
        already = {
            row.chunk_id
            for row in session.scalars(
                select(EmbeddingMetadata).where(EmbeddingMetadata.chunk_id.in_(request.chunk_ids))
            ).all()
        }

    todo = [item for item in pending if item[0] not in already]
    skipped = len(request.chunk_ids) - len(todo)

    if not todo:
        return EmbedChunksResponse(
            provider=settings.embedding_provider,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            embedded=0,
            skipped=skipped,
            total_latency_ms=0.0,
            vector_ids=[],
        )

    vectors, elapsed_ms, model_name = await embed_texts_raw([text for _, text, _, _ in todo])
    per_item_ms = elapsed_ms / len(vectors) if vectors else 0.0

    store = get_store()
    index = store.index(request.collection)
    items = [
        (chunk_id, vector, metadata)
        for (chunk_id, _text, metadata, _collection), vector in zip(todo, vectors, strict=True)
    ]
    vector_ids = index.upsert(items, embedding_model=model_name)

    with session_scope() as session:
        for (chunk_id, text, _metadata, collection), vector, vector_id in zip(
            todo, vectors, vector_ids, strict=True
        ):
            session.add(
                EmbeddingMetadata(
                    chunk_id=chunk_id,
                    vector_id=vector_id,
                    collection=collection,
                    embedding_provider=settings.embedding_provider,
                    embedding_model=model_name,
                    dimensions=len(vector),
                    norm=_norm(vector),
                    latency_ms=per_item_ms,
                    input_tokens=estimate_tokens(text),
                    vector_preview=_preview(vector),
                    created_at=utcnow(),
                )
            )

    logger.info(
        "embed_chunks",
        collection=request.collection.value,
        embedded=len(vector_ids),
        skipped=skipped,
        model=model_name,
        total_ms=round(elapsed_ms, 1),
    )

    return EmbedChunksResponse(
        provider=settings.embedding_provider,
        model=model_name,
        dimensions=len(vectors[0]) if vectors else settings.embedding_dimensions,
        embedded=len(vector_ids),
        skipped=skipped,
        total_latency_ms=elapsed_ms,
        vector_ids=vector_ids,
    )


@register_endpoint(SERVICE_NAME, "GET", "/health", ServiceHealth)
async def health() -> ServiceHealth:
    settings = get_settings()
    status = await OllamaClient().status()
    return ServiceHealth(
        service=SERVICE_NAME,
        status="ok" if status.embedding_model_available else "degraded",
        transport=settings.service_transport,
        detail={
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
            "batch_size": settings.embedding_batch_size,
            "model_available": status.embedding_model_available,
            "ollama_reachable": status.reachable,
            "indexed_vectors": {
                collection.value: idx.count
                for collection, idx in get_store().all_indexes().items()
                if collection in set(Collection)
            },
        },
    )
