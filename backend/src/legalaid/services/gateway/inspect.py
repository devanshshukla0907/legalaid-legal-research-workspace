"""Inspection endpoints: traces, chunks, embeddings, vectors, knowledge base, matters.

These exist so the UI can answer, for any request: what was chunked, what was embedded, which
vectors were searched, which chunks came back, why, which agent used them, what prompt it saw,
and what it returned. Everything here reads recorded data - nothing is recomputed for display.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, select

from legalaid.core.security import (
    Principal,
    assert_document_access,
    assert_matter_access,
    require_principal,
)
from legalaid.core.tokens import estimate_tokens
from legalaid.db.base import session_scope
from legalaid.db.models import (
    AgentRun,
    Case,
    Chunk,
    Document,
    DocumentPage,
    EmbeddingMetadata,
    KnowledgeSource,
    LegalProvision,
    Matter,
    ModelRun,
    PipelineTrace,
    RetrievalResult,
    RetrievalRun,
    TraceStage,
)
from legalaid.domain.enums import STAGE_LABELS, AgentKey, Collection, StageKey
from legalaid.services.retrieval.graph import get_graph
from legalaid.services.vector.store import get_store

router = APIRouter()


# --------------------------------------------------------------------------- traces


@router.get("/api/traces", tags=["trace"])
async def list_traces(limit: int = Query(30, le=200)) -> dict[str, Any]:
    with session_scope() as session:
        rows = session.scalars(
            select(PipelineTrace).order_by(desc(PipelineTrace.started_at)).limit(limit)
        ).all()
        return {
            "traces": [
                {
                    "trace_id": t.id,
                    "question": t.question,
                    "mode": t.mode,
                    "status": t.status,
                    "started_at": t.started_at.isoformat() if t.started_at else None,
                    "total_ms": t.total_ms,
                    "matter_id": t.matter_id,
                    "stages": session.scalar(
                        select(func.count()).select_from(TraceStage).where(
                            TraceStage.trace_id == t.id
                        )
                    )
                    or 0,
                }
                for t in rows
            ]
        }


@router.get("/api/traces/{trace_id}", tags=["trace"])
async def get_trace(
    trace_id: str, principal: Principal = Depends(require_principal)
) -> dict[str, Any]:
    """Everything recorded for one request, in pipeline order.

    Privacy note, stated rather than hidden: this returns the verbatim prompts sent to the
    model, which include excerpts of any uploaded document and therefore any client names in
    it. That is a deliberate trade-off - the whole purpose of the application is to make the
    pipeline inspectable, and a redacted trace could not show what the model actually saw.
    The mitigation is access control (this endpoint requires a key) plus the
    `contains_client_data` flag below, so a caller knows when a payload carries matter content.
    """
    with session_scope() as session:
        trace = session.get(PipelineTrace, trace_id)
        if trace is None:
            raise HTTPException(404, f"Trace {trace_id} not found")

        stages = session.scalars(
            select(TraceStage)
            .where(TraceStage.trace_id == trace_id)
            .order_by(TraceStage.ordinal)
        ).all()
        agent_runs = session.scalars(
            select(AgentRun).where(AgentRun.trace_id == trace_id).order_by(AgentRun.started_at)
        ).all()
        model_runs = session.scalars(
            select(ModelRun).where(ModelRun.trace_id == trace_id).order_by(ModelRun.created_at)
        ).all()
        retrieval_runs = session.scalars(
            select(RetrievalRun)
            .where(RetrievalRun.trace_id == trace_id)
            .order_by(RetrievalRun.created_at)
        ).all()

        retrievals: list[dict[str, Any]] = []
        for run in retrieval_runs:
            results = session.scalars(
                select(RetrievalResult)
                .where(RetrievalResult.retrieval_run_id == run.id)
                .order_by(RetrievalResult.rank)
            ).all()
            retrievals.append(
                {
                    "id": run.id,
                    "collection": run.collection,
                    "mode": run.mode,
                    "query_text": run.query_text,
                    "embedding_model": run.embedding_model,
                    "dimensions": run.dimensions,
                    "query_vector_preview": run.query_vector_preview,
                    "candidate_count": run.candidate_count,
                    "filtered_count": run.filtered_count,
                    "returned_count": run.returned_count,
                    "top_k": run.top_k,
                    "min_similarity": run.min_similarity,
                    "embedding_ms": run.embedding_ms,
                    "search_ms": run.search_ms,
                    "rank_ms": run.rank_ms,
                    "results": [
                        {
                            "chunk_id": r.chunk_id,
                            "rank": r.rank,
                            "similarity": r.similarity,
                            "match_reasons": r.match_reasons,
                            "matched_terms": r.matched_terms,
                        }
                        for r in results
                    ],
                }
            )

        return {
            "trace_id": trace.id,
            "question": trace.question,
            "mode": trace.mode,
            "status": trace.status,
            "started_at": trace.started_at.isoformat() if trace.started_at else None,
            "completed_at": trace.completed_at.isoformat() if trace.completed_at else None,
            "total_ms": trace.total_ms,
            "answer": trace.answer,
            "summary": trace.summary,
            "error": trace.error,
            "stages": [
                {
                    "id": s.id,
                    "ordinal": s.ordinal,
                    "stage_key": s.stage_key,
                    "label": s.label,
                    "service": s.service,
                    "transport": s.transport,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "input": s.input,
                    "output": s.output,
                    "error": s.error,
                }
                for s in stages
            ],
            "agent_runs": [
                {
                    "id": a.id,
                    "agent_key": a.agent_key,
                    "label": a.agent_label,
                    "status": a.status,
                    "duration_ms": a.duration_ms,
                    "prompt_name": a.prompt_name,
                    "prompt_version": a.prompt_version,
                    "included_chunk_ids": a.included_chunk_ids,
                    "excluded_chunk_ids": a.excluded_chunk_ids,
                    "context_manifest": a.context_manifest,
                    "output": a.output,
                    "error": a.error,
                }
                for a in agent_runs
            ],
            "model_runs": [
                {
                    "id": m.id,
                    "agent_key": m.agent_key,
                    "provider": m.provider,
                    "model": m.model,
                    "temperature": m.temperature,
                    "max_tokens": m.max_tokens,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "total_duration_ms": m.total_duration_ms,
                    "tokens_per_second": m.tokens_per_second,
                    "prompt_name": m.prompt_name,
                    "prompt_version": m.prompt_version,
                    "system_prompt": m.system_prompt,
                    "user_prompt": m.user_prompt,
                    "raw_response": m.raw_response,
                    "status": m.status,
                }
                for m in model_runs
            ],
            "retrievals": retrievals,
            # Tells a caller this payload carries matter content, so it can be handled
            # accordingly rather than treated as anonymous telemetry.
            "contains_client_data": any(
                run.agent_key == AgentKey.DOCUMENT_UNDERSTANDING.value for run in agent_runs
            )
            or trace.matter_id is not None,
            "privacy_note": (
                "Prompts are stored verbatim so the pipeline is inspectable. Where a matter "
                "document was involved, its text appears in these prompts."
            ),
        }


@router.get("/api/stages", tags=["trace"])
async def stage_catalogue() -> dict[str, Any]:
    """The canonical pipeline stages, so the UI can render the full graph."""
    return {
        "stages": [
            {"key": key.value, "label": STAGE_LABELS.get(key, key.value)} for key in StageKey
        ]
    }


# --------------------------------------------------------------------------- chunks


@router.get("/api/chunks", tags=["chunks"])
async def list_chunks(
    collection: Collection | None = None,
    q: str | None = None,
    limit: int = Query(50, le=300),
    offset: int = 0,
) -> dict[str, Any]:
    with session_scope() as session:
        stmt = select(Chunk)
        if collection:
            stmt = stmt.where(Chunk.collection == collection.value)
        if q:
            stmt = stmt.where(Chunk.text.ilike(f"%{q}%"))
        total = session.scalar(
            select(func.count()).select_from(stmt.subquery())
        ) or 0
        rows = session.scalars(
            stmt.order_by(Chunk.source_title, Chunk.ordinal).limit(limit).offset(offset)
        ).all()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "chunks": [
                {
                    "chunk_id": c.id,
                    "collection": c.collection,
                    "source_type": c.source_type,
                    "source_title": c.source_title,
                    "section": c.section,
                    "heading": c.heading,
                    "page": c.page,
                    "char_count": c.char_count,
                    "token_count": c.token_count,
                    "strategy": c.chunk_strategy,
                    "preview": c.text[:220],
                }
                for c in rows
            ],
        }


@router.get("/api/chunks/{chunk_id}", tags=["chunks"])
async def get_chunk(chunk_id: str) -> dict[str, Any]:
    """One chunk in full, with its embedding, vector, and everywhere it has been used."""
    with session_scope() as session:
        chunk = session.get(Chunk, chunk_id)
        if chunk is None:
            raise HTTPException(404, f"Chunk {chunk_id} not found")

        embedding = session.scalar(
            select(EmbeddingMetadata).where(EmbeddingMetadata.chunk_id == chunk_id)
        )
        results = session.scalars(
            select(RetrievalResult).where(RetrievalResult.chunk_id == chunk_id)
        ).all()
        used_by = session.scalars(
            select(AgentRun).where(
                AgentRun.included_chunk_ids.like(f'%{chunk_id}%')
            )
        ).all()

        provision = (
            session.get(LegalProvision, chunk.provision_id) if chunk.provision_id else None
        )
        case = session.get(Case, chunk.case_id) if chunk.case_id else None
        document = session.get(Document, chunk.document_id) if chunk.document_id else None

        vector = get_store().index(Collection(chunk.collection)).get_vector(chunk_id)

        return {
            "chunk_id": chunk.id,
            "collection": chunk.collection,
            "source_type": chunk.source_type,
            "source_title": chunk.source_title,
            "section": chunk.section,
            "chapter": chunk.chapter,
            "heading": chunk.heading,
            "page": chunk.page,
            "text": chunk.text,
            "char_count": chunk.char_count,
            "token_count": chunk.token_count,
            "ordinal": chunk.ordinal,
            "strategy": chunk.chunk_strategy,
            "source_url": chunk.source_url,
            "metadata": chunk.chunk_metadata,
            "version": chunk.version,
            "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
            "embedding": (
                {
                    "id": embedding.id,
                    "vector_id": embedding.vector_id,
                    "model": embedding.embedding_model,
                    "provider": embedding.embedding_provider,
                    "dimensions": embedding.dimensions,
                    "norm": embedding.norm,
                    "latency_ms": embedding.latency_ms,
                    "input_tokens": embedding.input_tokens,
                    "preview": embedding.vector_preview,
                }
                if embedding
                else None
            ),
            "vector": vector[:64] if vector else None,
            "vector_dimensions": len(vector) if vector else None,
            "retrieved_count": len(results),
            "retrievals": [
                {
                    "retrieval_run_id": r.retrieval_run_id,
                    "rank": r.rank,
                    "similarity": r.similarity,
                    "match_reasons": r.match_reasons,
                }
                for r in results[:20]
            ],
            "used_by_agents": [
                {"agent": a.agent_label, "trace_id": a.trace_id, "run_id": a.id}
                for a in used_by[:20]
            ],
            "provenance": {
                "provision": (
                    {
                        "act_name": provision.act_name,
                        "section_number": provision.section_number,
                        "section_heading": provision.section_heading,
                        "source_url": provision.source_url,
                        "source_authority": provision.source_authority,
                        "amendment_note": provision.amendment_note,
                        "retrieved_at": provision.retrieved_at.isoformat()
                        if provision.retrieved_at
                        else None,
                    }
                    if provision
                    else None
                ),
                "case": (
                    {
                        "case_title": case.case_title,
                        "citation": case.citation,
                        "neutral_citation": case.neutral_citation,
                        "court": case.court,
                        "decision_date": case.decision_date,
                        "judges": case.judges,
                        "disposal_nature": case.disposal_nature,
                        "source_url": case.source_url,
                        "licence": case.licence,
                    }
                    if case
                    else None
                ),
                "document": (
                    {
                        "filename": document.filename,
                        "doc_class": document.doc_class,
                        "page_count": document.page_count,
                    }
                    if document
                    else None
                ),
            },
        }


@router.get("/api/embeddings", tags=["embeddings"])
async def list_embeddings(limit: int = Query(50, le=300)) -> dict[str, Any]:
    with session_scope() as session:
        rows = session.scalars(
            select(EmbeddingMetadata).order_by(desc(EmbeddingMetadata.created_at)).limit(limit)
        ).all()
        total = session.scalar(select(func.count()).select_from(EmbeddingMetadata)) or 0
        models = session.scalars(select(EmbeddingMetadata.embedding_model).distinct()).all()
        return {
            "total": total,
            "models": list(models),
            "embeddings": [
                {
                    "id": e.id,
                    "chunk_id": e.chunk_id,
                    "vector_id": e.vector_id,
                    "collection": e.collection,
                    "model": e.embedding_model,
                    "dimensions": e.dimensions,
                    "norm": e.norm,
                    "latency_ms": e.latency_ms,
                    "input_tokens": e.input_tokens,
                    "preview": e.vector_preview,
                }
                for e in rows
            ],
        }


# --------------------------------------------------------------------------- knowledge


@router.get("/api/knowledge", tags=["knowledge"])
async def knowledge_overview() -> dict[str, Any]:
    store = get_store()
    with session_scope() as session:
        acts: dict[str, dict[str, Any]] = {}
        for p in session.scalars(select(LegalProvision)).all():
            entry = acts.setdefault(
                p.act_name,
                {
                    "act_name": p.act_name,
                    "act_year": p.act_year,
                    "provisions": 0,
                    "source_url": p.source_url,
                    "authority": p.source_authority,
                    "collection": p.collection,
                },
            )
            entry["provisions"] += 1

        cases = session.scalars(select(Case).order_by(desc(Case.year))).all()
        sources = session.scalars(select(KnowledgeSource)).all()

        return {
            "sources": [
                {
                    "key": s.key,
                    "name": s.name,
                    "authority": s.authority,
                    "licence": s.licence,
                    "attribution": s.attribution,
                    "source_url": s.source_url,
                    "item_count": s.item_count,
                    "retrieved_at": s.retrieved_at.isoformat() if s.retrieved_at else None,
                    "notes": s.notes,
                }
                for s in sources
            ],
            "acts": sorted(acts.values(), key=lambda a: -a["provisions"]),
            "cases": [
                {
                    "id": c.id,
                    "case_title": c.case_title,
                    "citation": c.citation,
                    "neutral_citation": c.neutral_citation,
                    "court": c.court,
                    "year": c.year,
                    "decision_date": c.decision_date,
                    "judges": c.judges,
                    "disposal_nature": c.disposal_nature,
                    "legal_provisions": c.legal_provisions,
                    "keywords": c.keywords,
                    "source_url": c.source_url,
                    "licence": c.licence,
                }
                for c in cases
            ],
            "vectors_by_collection": {
                c.value: i.count for c, i in store.all_indexes().items()
            },
        }


@router.get("/api/knowledge/provisions", tags=["knowledge"])
async def list_provisions(
    act: str | None = None, q: str | None = None, limit: int = Query(100, le=400)
) -> dict[str, Any]:
    with session_scope() as session:
        stmt = select(LegalProvision)
        if act:
            stmt = stmt.where(LegalProvision.act_name == act)
        if q:
            stmt = stmt.where(LegalProvision.text.ilike(f"%{q}%"))
        rows = session.scalars(stmt.limit(limit)).all()
        return {
            "provisions": [
                {
                    "id": p.id,
                    "act_name": p.act_name,
                    "section_number": p.section_number,
                    "section_heading": p.section_heading,
                    "text": p.text,
                    "source_url": p.source_url,
                    "source_authority": p.source_authority,
                    "amendment_note": p.amendment_note,
                    "version": p.version,
                    "chars": len(p.text),
                }
                for p in rows
            ]
        }


@router.get("/api/knowledge/graph", tags=["knowledge"])
async def knowledge_graph(limit: int = Query(160, le=600)) -> dict[str, Any]:
    """The graph-RAG structure, for visualisation. No vectors involved."""
    graph = get_graph()
    nodes = list(graph.nodes.values())[:limit]
    keys = {n.key for n in nodes}
    edges = [
        {"source": e.source, "target": e.target, "relation": e.relation, "evidence": e.evidence}
        for edges_for in graph.out.values()
        for e in edges_for
        if e.source in keys and e.target in keys
    ]
    return {
        "nodes": [
            {
                "key": n.key,
                "kind": n.kind,
                "label": n.label,
                "chunk_count": len(n.chunk_ids),
                "metadata": n.metadata,
            }
            for n in nodes
        ],
        "edges": edges,
        "stats": {
            "total_nodes": len(graph.nodes),
            "total_edges": graph.edge_count,
        },
    }


# --------------------------------------------------------------------------- matters


@router.get("/api/matters", tags=["matters"])
async def list_matters() -> dict[str, Any]:
    with session_scope() as session:
        rows = session.scalars(select(Matter).order_by(desc(Matter.updated_at))).all()
        return {
            "matters": [
                {
                    "id": m.id,
                    "title": m.title,
                    "reference_no": m.reference_no,
                    "matter_type": m.matter_type,
                    "court": m.court,
                    "status": m.status,
                    "is_demo": m.is_demo,
                    "description": m.description,
                    "documents": session.scalar(
                        select(func.count()).select_from(Document).where(
                            Document.matter_id == m.id
                        )
                    )
                    or 0,
                    "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                }
                for m in rows
            ]
        }


@router.get("/api/documents", tags=["documents"])
async def list_documents(
    matter_id: str | None = None, principal: Principal = Depends(require_principal)
) -> dict[str, Any]:
    assert_matter_access(matter_id, principal)
    with session_scope() as session:
        stmt = select(Document)
        if matter_id:
            stmt = stmt.where(Document.matter_id == matter_id)
        rows = session.scalars(stmt.order_by(desc(Document.uploaded_at))).all()
        return {
            "documents": [
                {
                    "id": d.id,
                    "filename": d.filename,
                    "matter_id": d.matter_id,
                    "doc_class": d.doc_class,
                    "status": d.status,
                    "page_count": d.page_count,
                    "size_bytes": d.size_bytes,
                    "extraction_method": d.extraction_method,
                    "is_synthetic": d.is_synthetic,
                    "chunks": session.scalar(
                        select(func.count()).select_from(Chunk).where(Chunk.document_id == d.id)
                    )
                    or 0,
                    "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
                    "error": d.error,
                }
                for d in rows
            ]
        }


@router.get("/api/documents/{document_id}", tags=["documents"])
async def get_document(
    document_id: str, principal: Principal = Depends(require_principal)
) -> dict[str, Any]:
    """A document with its extracted pages and chunks, preserving page provenance.

    Returns full extracted page text, so it is authorised through the matter that owns the
    document rather than being readable by anyone holding the id.
    """
    assert_document_access(document_id, principal)
    with session_scope() as session:
        document = session.get(Document, document_id)
        if document is None:
            raise HTTPException(404, f"Document {document_id} not found")
        pages = session.scalars(
            select(DocumentPage)
            .where(DocumentPage.document_id == document_id)
            .order_by(DocumentPage.page_number)
        ).all()
        chunks = session.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.ordinal)
        ).all()
        return {
            "id": document.id,
            "filename": document.filename,
            "doc_class": document.doc_class,
            "status": document.status,
            "page_count": document.page_count,
            "size_bytes": document.size_bytes,
            "sha256": document.sha256,
            "extraction_method": document.extraction_method,
            "metadata": document.doc_metadata,
            "is_synthetic": document.is_synthetic,
            "error": document.error,
            "pages": [
                {
                    "page_number": p.page_number,
                    "text": p.text,
                    "char_count": p.char_count,
                    "headings": p.headings,
                }
                for p in pages
            ],
            "chunks": [
                {
                    "chunk_id": c.id,
                    "ordinal": c.ordinal,
                    "page": c.page,
                    "heading": c.heading,
                    "char_count": c.char_count,
                    "token_count": c.token_count,
                    "strategy": c.chunk_strategy,
                    "preview": c.text[:200],
                }
                for c in chunks
            ],
        }


# --------------------------------------------------------------------------- metrics


@router.get("/api/metrics", tags=["system"])
async def metrics() -> dict[str, Any]:
    """Performance figures for the dashboard, computed from recorded runs."""
    with session_scope() as session:
        model_runs = session.scalars(
            select(ModelRun).order_by(desc(ModelRun.created_at)).limit(200)
        ).all()
        retrieval_runs = session.scalars(
            select(RetrievalRun).order_by(desc(RetrievalRun.created_at)).limit(200)
        ).all()
        traces = session.scalars(
            select(PipelineTrace).order_by(desc(PipelineTrace.started_at)).limit(100)
        ).all()

        def avg(values: list[float]) -> float | None:
            clean = [v for v in values if v]
            return round(sum(clean) / len(clean), 1) if clean else None

        by_agent: dict[str, dict[str, Any]] = {}
        for run in model_runs:
            entry = by_agent.setdefault(
                run.agent_key, {"runs": 0, "latency": [], "in": 0, "out": 0}
            )
            entry["runs"] += 1
            entry["latency"].append(run.total_duration_ms)
            entry["in"] += run.input_tokens or 0
            entry["out"] += run.output_tokens or 0

        return {
            "llm": {
                "calls": len(model_runs),
                "avg_latency_ms": avg([m.total_duration_ms for m in model_runs]),
                "avg_tokens_per_second": avg(
                    [m.tokens_per_second for m in model_runs if m.tokens_per_second]
                ),
                "total_input_tokens": sum(m.input_tokens or 0 for m in model_runs),
                "total_output_tokens": sum(m.output_tokens or 0 for m in model_runs),
                "failures": sum(1 for m in model_runs if m.status != "success"),
            },
            "retrieval": {
                "runs": len(retrieval_runs),
                "avg_embedding_ms": avg([r.embedding_ms for r in retrieval_runs]),
                "avg_search_ms": avg([r.search_ms for r in retrieval_runs]),
                "avg_candidates": avg([float(r.candidate_count) for r in retrieval_runs]),
            },
            "pipeline": {
                "traces": len(traces),
                "avg_total_ms": avg([t.total_ms for t in traces if t.total_ms]),
                "successful": sum(1 for t in traces if t.status == "success"),
                "partial": sum(1 for t in traces if t.status == "partial"),
                "failed": sum(1 for t in traces if t.status == "failed"),
            },
            "by_agent": {
                key: {
                    "runs": v["runs"],
                    "avg_latency_ms": avg(v["latency"]),
                    "input_tokens": v["in"],
                    "output_tokens": v["out"],
                }
                for key, v in by_agent.items()
            },
        }
