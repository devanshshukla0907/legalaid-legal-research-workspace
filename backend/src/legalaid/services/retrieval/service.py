"""Retrieval service.

Owns retrieval *strategy*; the vector service owns similarity math. That split is what makes
each retrieval mode independently testable.

    question -> query embedding -> source filter -> vector search -> top-k
             -> metadata filtering -> ranking -> context construction

Every step is persisted against the trace id (`RetrievalRun` + `RetrievalResult`) so the
Retrieval Inspector shows real candidate counts, real similarity scores and a real explanation
for each hit rather than a reconstruction.
"""

from __future__ import annotations

import time

from sqlalchemy import select

from legalaid.capabilities import require
from legalaid.config import get_settings
from legalaid.core.logging import get_logger
from legalaid.core.tokens import estimate_tokens
from legalaid.core.transport import register_endpoint
from legalaid.db.base import new_id, session_scope, utcnow
from legalaid.db.models import Chunk, PipelineTrace, RetrievalResult, RetrievalRun
from legalaid.domain.contracts import (
    CollectionSearchTrace,
    QueryEmbeddingTrace,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
    ServiceHealth,
)
from legalaid.domain.enums import LAW_COLLECTIONS, Collection, RetrievalMode
from legalaid.services.embedding.service import _preview, embed_texts_raw
from legalaid.services.retrieval.explain import explain_match
from legalaid.services.retrieval.fusion import describe_fusion, reciprocal_rank_fusion
from legalaid.services.retrieval.graph import chunk_ids_for, get_graph
from legalaid.services.retrieval.keyword import get_index
from legalaid.services.vector.store import get_store

logger = get_logger(__name__)

SERVICE_NAME = "retrieval-service"

# Which collections each mode searches. Retrieval always knows what source type it is reading.
MODE_COLLECTIONS: dict[RetrievalMode, tuple[Collection, ...]] = {
    RetrievalMode.LAW_SEARCH: LAW_COLLECTIONS,
    RetrievalMode.CASE_SEARCH: (Collection.CASE_LAW,),
    RetrievalMode.USER_DOCUMENT_SEARCH: (Collection.USER_DOCUMENTS,),
    RetrievalMode.COMBINED_LEGAL_RESEARCH: (
        *LAW_COLLECTIONS,
        Collection.CASE_LAW,
        Collection.USER_DOCUMENTS,
    ),
}


def _top_k_for(collection: Collection, override: int | None) -> int:
    if override is not None:
        return override
    settings = get_settings()
    if collection is Collection.CASE_LAW:
        return settings.retrieval_top_k_cases
    if collection is Collection.USER_DOCUMENTS:
        return settings.retrieval_top_k_user_docs
    return settings.retrieval_top_k_laws


def _to_retrieved(
    row: Chunk,
    collection: Collection,
    *,
    similarity: float,
    reasons: list[str],
    matched_terms: list[str],
    retrievers: list[str],
    fused_score: float,
) -> RetrievedChunk:
    """Build the API view of a retrieved chunk, carrying its provenance."""
    metadata = dict(row.chunk_metadata or {})
    # Which retrievers found this is itself observability data the UI displays.
    metadata["retrievers"] = retrievers
    metadata["fused_score"] = round(fused_score, 6)
    return RetrievedChunk(
        chunk_id=row.id,
        collection=collection,
        rank=0,
        similarity=similarity,
        source_title=row.source_title,
        section=row.section,
        chapter=row.chapter,
        page=row.page,
        heading=row.heading,
        text=row.text,
        source_url=row.source_url,
        source_type=row.source_type,
        token_count=row.token_count,
        match_reasons=reasons,
        matched_terms=matched_terms,
        metadata=metadata,
    )


def _filters_for(
    collection: Collection, request: RetrievalRequest
) -> dict[str, object]:
    """Build metadata filters, scoping user documents to the matter under work.

    Without the matter scope, one lawyer's matter could surface another's documents. Public
    knowledge collections are deliberately unscoped.
    """
    filters = dict(request.filters)
    if collection is Collection.USER_DOCUMENTS and request.matter_id:
        filters.setdefault("matter_id", request.matter_id)
    return filters


@register_endpoint(SERVICE_NAME, "POST", "/retrieval/search", RetrievalResponse, RetrievalRequest)
async def search(request: RetrievalRequest) -> RetrievalResponse:
    """Embed the question, search the mode's collections, rank and assemble context."""
    require("rag_retrieval")
    settings = get_settings()
    started = time.perf_counter()

    # Per-request overrides, resolved once. Reading them here rather than mutating the cached
    # global Settings keeps concurrent requests independent - the previous approach swapped
    # values on a process-wide singleton and restored them in a finally block, so two
    # simultaneous searches could restore each other's toggles.
    use_keyword = (
        settings.retrieval_enable_keyword
        if request.enable_keyword is None
        else request.enable_keyword
    )
    use_graph = (
        settings.retrieval_enable_graph if request.enable_graph is None else request.enable_graph
    )
    use_dense = True if request.enable_dense is None else request.enable_dense
    min_similarity = (
        settings.retrieval_min_similarity
        if request.min_similarity is None
        else request.min_similarity
    )
    keyword_weight = (
        settings.retrieval_keyword_weight
        if request.keyword_weight is None
        else request.keyword_weight
    )
    dense_weight = (
        settings.retrieval_dense_weight if request.dense_weight is None else request.dense_weight
    )

    collections = tuple(request.collections) if request.collections else MODE_COLLECTIONS[request.mode]

    # ------------------------------------------------------- query embedding
    require("query_embedding")
    vectors, embed_ms, model_name = await embed_texts_raw([request.question])
    if not vectors:
        raise RuntimeError("Query embedding returned no vector")
    query_vector = vectors[0]

    embedding_trace = QueryEmbeddingTrace(
        provider=settings.embedding_provider,
        model=model_name,
        dimensions=len(query_vector),
        latency_ms=embed_ms,
        input_chars=len(request.question),
        input_tokens=estimate_tokens(request.question),
        preview=_preview(query_vector),
    )

    store = get_store()
    searches: list[CollectionSearchTrace] = []
    all_hits: list[RetrievedChunk] = []

    # Graph seeds are resolved once: the question names the same provisions regardless of
    # which collection is being searched.
    #
    # Two seeding routes, because the first alone measured a contribution of exactly zero:
    #   1. Explicit - the question names a section or an act. Precise, but conversational
    #      questions ("the buyer refused payment") name neither, so it almost never fires.
    #   2. Dense-anchored - take the provisions dense search already surfaced and traverse out
    #      from those. This is the graph's actual useful role: expansion from a known-good
    #      anchor, reaching provisions that cross-reference the ones semantic search found.
    graph_chunk_ids: set[str] = set()
    graph_paths: dict[str, str] = {}
    if use_graph:
        graph = get_graph()
        seeds = graph.find_seeds(request.question)

        if not seeds and use_dense:
            # No explicit anchor. Probe the largest law collection for the best semantic hits
            # and seed from those instead of contributing nothing.
            anchor_index = store.index(Collection.LEGAL_ACTS)
            if anchor_index.count:
                anchor_hits, _c, _f = anchor_index.search(
                    query_vector, top_k=3, min_similarity=min_similarity
                )
                if anchor_hits:
                    seeds = graph.seed_from_chunks([h.chunk_id for h in anchor_hits])

        if seeds:
            expanded = graph.expand(
                seeds, max_hops=settings.retrieval_graph_hops, limit=settings.retrieval_graph_limit
            )
            for result in expanded:
                for chunk_id in result.node.chunk_ids:
                    graph_chunk_ids.add(chunk_id)
                    graph_paths.setdefault(chunk_id, result.path.describe(graph))
            # Seed nodes themselves are direct hits, not just expansion origins.
            for seed in seeds:
                node = graph.nodes.get(seed)
                if node:
                    for chunk_id in node.chunk_ids:
                        graph_chunk_ids.add(chunk_id)
                        graph_paths.setdefault(chunk_id, f"named in question: {node.label}")

    for collection in collections:
        index = store.index(collection)
        top_k = _top_k_for(collection, request.top_k)
        filters = _filters_for(collection, request)

        search_started = time.perf_counter()
        # Over-fetch from each retriever so fusion has a real pool to work with; the final
        # top_k is applied after fusion, not before.
        fetch_k = top_k * settings.retrieval_candidate_multiplier
        if use_dense:
            hits, candidate_count, filtered_count = index.search(
                query_vector,
                top_k=fetch_k,
                filters=filters or None,
                min_similarity=min_similarity,
            )
        else:
            # Dense disabled: the ablation harness needs a keyword-only or graph-only run.
            hits, candidate_count, filtered_count = [], 0, 0

        keyword_hits: list[tuple[str, float, list[str]]] = []
        keyword_candidates = 0
        if use_keyword:
            bm25 = get_index()
            matter_scope = (
                request.matter_id if collection is Collection.USER_DOCUMENTS else None
            )
            kw, keyword_candidates = bm25.search(
                request.question,
                collection=collection,
                top_k=fetch_k,
                matter_id=matter_scope,
            )
            keyword_hits = [(h.chunk_id, h.score, h.matched_terms) for h in kw]
        search_ms = (time.perf_counter() - search_started) * 1000

        rank_started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [(h.chunk_id, h.similarity) for h in hits],
            keyword_hits,
            dense_weight=dense_weight,
            keyword_weight=keyword_weight,
        )

        # Graph hits join the candidate pool for this collection, so a provision named in the
        # question is reachable even when neither dense nor BM25 surfaced it.
        dense_similarity = {h.chunk_id: h.similarity for h in hits}
        fused_ids = {f.chunk_id for f in fused}
        graph_only = [cid for cid in graph_chunk_ids if cid not in fused_ids]

        candidate_ids = [f.chunk_id for f in fused] + graph_only
        rows: dict[str, Chunk] = {}
        if candidate_ids:
            with session_scope() as session:
                for row in session.scalars(
                    select(Chunk).where(Chunk.id.in_(candidate_ids))
                ).all():
                    rows[row.id] = row

        scored: list[tuple[float, RetrievedChunk]] = []
        for fused_hit in fused:
            row = rows.get(fused_hit.chunk_id)
            if row is None:
                # Index and database disagree: report rather than fabricate a result.
                logger.warning(
                    "retrieval_orphan_vector",
                    chunk_id=fused_hit.chunk_id,
                    collection=collection.value,
                )
                continue
            if row.collection != collection.value:
                continue

            similarity = fused_hit.dense_similarity or dense_similarity.get(row.id, 0.0)
            reasons, matched_terms = explain_match(
                question=request.question,
                chunk_text=row.text,
                similarity=similarity,
                section=row.section,
                source_title=row.source_title,
            )
            reasons = describe_fusion(fused_hit) + reasons
            if row.id in graph_paths:
                reasons.append(f"Graph path: {graph_paths[row.id]}")

            scored.append(
                (
                    fused_hit.fused_score,
                    _to_retrieved(
                        row,
                        collection,
                        similarity=similarity,
                        reasons=reasons,
                        matched_terms=matched_terms or fused_hit.matched_terms,
                        retrievers=fused_hit.retrievers
                        + (["graph"] if row.id in graph_paths else []),
                        fused_score=fused_hit.fused_score,
                    ),
                )
            )

        for chunk_id in graph_only:
            row = rows.get(chunk_id)
            if row is None or row.collection != collection.value:
                continue
            scored.append(
                (
                    0.0,
                    _to_retrieved(
                        row,
                        collection,
                        similarity=dense_similarity.get(chunk_id, 0.0),
                        reasons=[
                            "Retrieved by graph traversal only - no vector or keyword match",
                            f"Graph path: {graph_paths.get(chunk_id, 'related provision')}",
                        ],
                        matched_terms=[],
                        retrievers=["graph"],
                        fused_score=0.0,
                    ),
                )
            )

        scored.sort(key=lambda item: -item[0])
        results: list[RetrievedChunk] = []
        for rank, (_score, chunk) in enumerate(scored[:top_k], start=1):
            chunk.rank = rank
            results.append(chunk)
        rank_ms = (time.perf_counter() - rank_started) * 1000
        candidate_count = max(candidate_count, keyword_candidates)

        run_id = new_id("rrn_")
        searches.append(
            CollectionSearchTrace(
                collection=collection,
                retrieval_run_id=run_id,
                candidate_count=candidate_count,
                filtered_count=filtered_count,
                top_k=top_k,
                returned=len(results),
                min_similarity=min_similarity,
                filters=filters,
                embedding_ms=embed_ms,
                search_ms=search_ms,
                rank_ms=rank_ms,
                results=results,
            )
        )
        all_hits.extend(results)

        _persist_run(
            run_id=run_id,
            request=request,
            collection=collection,
            candidate_count=candidate_count,
            filtered_count=filtered_count,
            top_k=top_k,
            embedding_model=model_name,
            embedding_ms=embed_ms,
            search_ms=search_ms,
            rank_ms=rank_ms,
            query_vector=query_vector,
            results=results,
            min_similarity=min_similarity,
        )

    all_hits.sort(key=lambda c: c.similarity, reverse=True)
    context_chars = sum(len(c.text) for c in all_hits)
    total_ms = (time.perf_counter() - started) * 1000

    logger.info(
        "retrieval_complete",
        mode=request.mode.value,
        collections=[c.value for c in collections],
        returned=len(all_hits),
        total_ms=round(total_ms, 1),
    )

    return RetrievalResponse(
        trace_id=request.trace_id,
        mode=request.mode,
        question=request.question,
        query_embedding=embedding_trace,
        searches=searches,
        chunks=all_hits,
        total_ms=total_ms,
        context_chars=context_chars,
        context_tokens=sum(c.token_count for c in all_hits),
    )


def _persist_run(
    *,
    run_id: str,
    request: RetrievalRequest,
    collection: Collection,
    candidate_count: int,
    filtered_count: int,
    top_k: int,
    embedding_model: str,
    embedding_ms: float,
    search_ms: float,
    rank_ms: float,
    query_vector: list[float],
    results: list[RetrievedChunk],
    min_similarity: float,
) -> None:
    """Persist the run and its results so the UI reads real recorded data.

    `min_similarity` is passed in rather than read from settings: it can be overridden per
    request by the ablation harness, and the recorded run must show the threshold that was
    actually applied.
    """
    settings = get_settings()
    with session_scope() as session:
        # trace_id is a foreign key to pipeline_traces. Retrieval can legitimately be called
        # standalone (tests, the Retrieval page), so only link a trace that actually exists
        # rather than failing the search on a constraint violation.
        trace_id = request.trace_id
        if trace_id is not None and session.get(PipelineTrace, trace_id) is None:
            trace_id = None

        session.add(
            RetrievalRun(
                id=run_id,
                trace_id=trace_id,
                mode=request.mode.value,
                collection=collection.value,
                query_text=request.question,
                embedding_provider=settings.embedding_provider,
                embedding_model=embedding_model,
                dimensions=len(query_vector),
                top_k=top_k,
                min_similarity=min_similarity,
                candidate_count=candidate_count,
                filtered_count=filtered_count,
                returned_count=len(results),
                # Matter scoping is applied through filters, which are recorded here.
                filters={**dict(request.filters), **({"matter_id": request.matter_id} if request.matter_id else {})},
                embedding_ms=embedding_ms,
                search_ms=search_ms,
                rank_ms=rank_ms,
                query_vector_preview=_preview(query_vector),
                created_at=utcnow(),
            )
        )
        for result in results:
            session.add(
                RetrievalResult(
                    retrieval_run_id=run_id,
                    chunk_id=result.chunk_id,
                    rank=result.rank,
                    similarity=result.similarity,
                    match_reasons=result.match_reasons,
                    matched_terms=result.matched_terms,
                )
            )


@register_endpoint(SERVICE_NAME, "GET", "/health", ServiceHealth)
async def health() -> ServiceHealth:
    settings = get_settings()
    store = get_store()
    counts = {c.value: index.count for c, index in store.all_indexes().items()}
    return ServiceHealth(
        service=SERVICE_NAME,
        status="ok" if store.total_vectors() > 0 else "degraded",
        transport=settings.service_transport,
        detail={
            "vectors_by_collection": counts,
            "total_vectors": store.total_vectors(),
            "top_k_laws": settings.retrieval_top_k_laws,
            "top_k_cases": settings.retrieval_top_k_cases,
            "top_k_user_docs": settings.retrieval_top_k_user_docs,
            "min_similarity": settings.retrieval_min_similarity,
            "note": (
                None
                if store.total_vectors()
                else "Knowledge base is empty. Run scripts/ingest_legislation.py --all"
            ),
        },
    )
