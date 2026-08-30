"""API gateway.

Every endpoint the UI needs, in one documented OpenAPI surface. The guiding rule is that the
UI must be able to answer "what happened, and where did this come from?" for any request, so
these routes expose the recorded internals - chunks, embeddings, vectors, agent contexts,
prompts, model runs, trace stages - rather than just final answers.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select

from legalaid.agents.orchestrator import Orchestrator
from legalaid.agents.prompts import all_prompts
from legalaid.capabilities import capability_matrix
from legalaid.config import get_settings
from legalaid.core.logging import get_logger
from legalaid.core.tracing import event_bus
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
    Workspace,
)
from legalaid.domain.contracts import RetrievalRequest
from legalaid.domain.enums import (
    AGENT_LABELS,
    AgentKey,
    Collection,
    DraftLength,
    DraftTone,
    DraftType,
    RetrievalMode,
)
from legalaid.core.guardrail_suite import run_suite
from legalaid.core.guardrails import screen_input
from legalaid.core.security import (
    Principal,
    assert_matter_access,
    enforce_rate_limit,
    queue_depth,
    rate_limit_snapshot,
    require_principal,
)
from legalaid.services.common import create_service_app
from legalaid.services.llm.ollama_client import OllamaClient
from legalaid.services.retrieval.graph import get_graph, graph_stats
from legalaid.services.retrieval.keyword import get_index
from legalaid.services.retrieval.service import search as retrieval_search
from legalaid.services.vector.store import get_store

logger = get_logger(__name__)

SERVICE_NAME = "gateway"

app: FastAPI = create_service_app(
    SERVICE_NAME,
    title="LegalAId API",
    description=(
        "AI-assisted Indian legal research. Every endpoint exposes the recorded internals of "
        "the pipeline so each stage can be inspected."
    ),
    init_db=True,
)


# --------------------------------------------------------------------------- models


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1)
    pasted_text: str = ""
    matter_id: str | None = None
    model: str | None = None
    enable_drafting: bool | None = None
    enable_verification: bool = True
    instructions: str = ""
    # The orchestrator has always accepted these; they were simply not exposed, so a draft
    # was permanently FORMAL/STANDARD and its type was whatever the router guessed.
    draft_type: DraftType | None = None
    tone: DraftTone = DraftTone.FORMAL
    length: DraftLength = DraftLength.STANDARD


class QuickRetrievalRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: RetrievalMode = RetrievalMode.COMBINED_LEGAL_RESEARCH
    top_k: int | None = None
    matter_id: str | None = None
    # Retriever ablation controls. None means "use the configured default".
    enable_keyword: bool | None = None
    enable_graph: bool | None = None
    enable_dense: bool | None = None
    min_similarity: float | None = Field(default=None, ge=0.0, le=1.0)


class CompareRequest(BaseModel):
    """RAG ON vs RAG OFF on the same question."""

    question: str = Field(min_length=1)
    model: str | None = None
    max_tokens: int = 400


# --------------------------------------------------------------------------- system


@app.get("/api/system/health", tags=["system"])
async def system_health() -> dict[str, Any]:
    settings = get_settings()
    llm = await OllamaClient().status()
    store = get_store()
    with session_scope() as session:
        counts = {
            "matters": session.scalar(select(func.count()).select_from(Matter)) or 0,
            "documents": session.scalar(select(func.count()).select_from(Document)) or 0,
            "provisions": session.scalar(select(func.count()).select_from(LegalProvision)) or 0,
            "cases": session.scalar(select(func.count()).select_from(Case)) or 0,
            "chunks": session.scalar(select(func.count()).select_from(Chunk)) or 0,
            "embeddings": session.scalar(select(func.count()).select_from(EmbeddingMetadata)) or 0,
            "traces": session.scalar(select(func.count()).select_from(PipelineTrace)) or 0,
            "agent_runs": session.scalar(select(func.count()).select_from(AgentRun)) or 0,
            "model_runs": session.scalar(select(func.count()).select_from(ModelRun)) or 0,
            "retrieval_runs": session.scalar(select(func.count()).select_from(RetrievalRun)) or 0,
        }
    bm25 = get_index()
    return {
        "status": "ok" if llm.reachable else "degraded",
        "environment": settings.environment,
        "active_phase": settings.active_phase,
        "transport": settings.service_transport,
        "ollama": {
            "reachable": llm.reachable,
            "base_url": llm.base_url,
            "llm_model": llm.configured_llm_model,
            "llm_model_available": llm.llm_model_available,
            "embedding_model": llm.configured_embedding_model,
            "embedding_model_available": llm.embedding_model_available,
            "error": llm.error,
        },
        "retrievers": {
            "dense": {
                "vectors": store.total_vectors(),
                "by_collection": {c.value: i.count for c, i in store.all_indexes().items()},
                "dimensions": settings.embedding_dimensions,
            },
            "keyword": {"documents": len(bm25.docs), "vocabulary": len(bm25.df)},
            "graph": graph_stats(),
        },
        "counts": counts,
        "capabilities": capability_matrix(),
        "guardrails": {
            "enabled": settings.guardrails_enabled,
            "corpus_similarity_floor": settings.guardrail_corpus_similarity_floor,
            "auth_enabled": settings.auth_enabled,
            "rate_limit": f"{settings.rate_limit_requests}/"
            f"{settings.rate_limit_window_seconds:.0f}s",
            "llm_max_concurrency": settings.llm_max_concurrency,
            "llm_queue_depth": queue_depth(),
            # Live per-key request counts, so throttling can be observed rather than assumed.
            "rate_limit_buckets": rate_limit_snapshot(),
        },
    }


@app.get("/api/models", tags=["system"])
async def list_models() -> dict[str, Any]:
    """Installed Ollama models, so the UI can offer a real choice."""
    settings = get_settings()
    status = await OllamaClient().status()
    installed = [
        {
            "name": m.name,
            "size_bytes": m.size_bytes,
            "parameter_size": m.parameter_size,
            "quantization": m.quantization,
            "family": m.family,
            # Embedding models cannot be used for generation; the UI must not offer them.
            "usable_for_generation": "embed" not in m.name.lower(),
        }
        for m in status.installed_models
    ]
    return {
        "provider": settings.llm_provider,
        "default_llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "num_ctx": settings.llm_num_ctx,
        "temperature": settings.llm_temperature,
        "reachable": status.reachable,
        "installed": installed,
    }


@app.get("/api/agents", tags=["agents"])
async def list_agents() -> dict[str, Any]:
    """Agent roster with prompt versions and measured recent performance."""
    prompts = {p.name: p for p in all_prompts()}
    roster = [
        {
            "key": AgentKey.INTAKE_ROUTER.value,
            "label": AGENT_LABELS[AgentKey.INTAKE_ROUTER],
            "order": 0,
            "responsibility": "Classify the request and rewrite it into a retrieval query",
            "sees": "raw user input and document summaries",
            "withheld": "the knowledge base (it runs before retrieval)",
            "prompt_name": "intake_router",
        },
        {
            "key": AgentKey.DOCUMENT_UNDERSTANDING.value,
            "label": AGENT_LABELS[AgentKey.DOCUMENT_UNDERSTANDING],
            "order": 1,
            "responsibility": "Extract parties, dates, amounts and obligations from documents",
            "sees": "matter document chunks only",
            "withheld": "all statute and case law",
            "prompt_name": "document_understanding",
        },
        {
            "key": AgentKey.LEGAL_RESEARCH.value,
            "label": AGENT_LABELS[AgentKey.LEGAL_RESEARCH],
            "order": 2,
            "responsibility": "Identify applicable Acts, sections and provisions",
            "sees": "law chunks + a little matter context",
            "withheld": "all case law",
            "prompt_name": "legal_research",
        },
        {
            "key": AgentKey.SIMILAR_CASES.value,
            "label": AGENT_LABELS[AgentKey.SIMILAR_CASES],
            "order": 3,
            "responsibility": "Find and assess similar prior judgments",
            "sees": "case-law chunks + a few provisions",
            "withheld": "matter documents",
            "prompt_name": "similar_cases",
        },
        {
            "key": AgentKey.DRAFTING.value,
            "label": AGENT_LABELS[AgentKey.DRAFTING],
            "order": 4,
            "responsibility": "Produce a lawyer-reviewable draft",
            "sees": "verified facts + both research agents' findings",
            "withheld": "raw chunks entirely, to prevent paraphrased authority",
            "prompt_name": "drafting",
        },
        {
            "key": AgentKey.VERIFICATION.value,
            "label": AGENT_LABELS[AgentKey.VERIFICATION],
            "order": 5,
            "responsibility": "Audit grounding and drive the refinement loop",
            "sees": "the reviewed agent's evidence set and output",
            "withheld": "nothing beyond that agent's own context",
            "prompt_name": "verification",
        },
    ]

    with session_scope() as session:
        for entry in roster:
            rows = session.scalars(
                select(AgentRun)
                .where(AgentRun.agent_key == entry["key"])
                .order_by(desc(AgentRun.started_at))
                .limit(20)
            ).all()
            durations = [r.duration_ms for r in rows if r.duration_ms]
            entry["runs"] = len(rows)
            entry["avg_latency_ms"] = (
                round(sum(durations) / len(durations), 1) if durations else None
            )
            entry["success_rate"] = (
                round(sum(1 for r in rows if r.status == "success") / len(rows), 3)
                if rows
                else None
            )
            entry["recent"] = [
                {
                    "id": r.id,
                    "trace_id": r.trace_id,
                    "status": r.status,
                    "duration_ms": r.duration_ms,
                    "prompt_version": r.prompt_version,
                    "included": len(r.included_chunk_ids or []),
                    "excluded": len(r.excluded_chunk_ids or []),
                }
                for r in rows[:5]
            ]
            prompt = prompts.get(str(entry["prompt_name"]))
            if prompt:
                entry["prompt_version"] = prompt.version
                entry["temperature"] = prompt.temperature
                entry["max_tokens"] = prompt.max_tokens
                entry["prompt_description"] = prompt.description
    return {"agents": roster}


@app.get("/api/prompts", tags=["agents"])
async def list_prompts() -> dict[str, Any]:
    return {
        "prompts": [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "temperature": p.temperature,
                "max_tokens": p.max_tokens,
                "variables": list(p.variables),
                "created_at": p.created_at,
                "system_prompt": p.system_prompt,
                "user_prompt": p.user_prompt,
            }
            for p in all_prompts()
        ]
    }


# --------------------------------------------------------------------------- research


@app.post("/api/research/query", tags=["research"])
async def research_query(
    payload: ResearchRequest, principal: Principal = Depends(enforce_rate_limit)
) -> dict[str, Any]:
    """Run the full multi-agent pipeline. Returns the trace id plus every agent's output."""
    assert_matter_access(payload.matter_id, principal)
    orchestrator = Orchestrator(model=payload.model)
    document_summaries: list[str] = []
    matter_title = ""
    with session_scope() as session:
        if payload.matter_id:
            matter = session.get(Matter, payload.matter_id)
            if matter is None:
                raise HTTPException(404, f"Matter {payload.matter_id} not found")
            matter_title = matter.title
            documents = session.scalars(
                select(Document).where(Document.matter_id == payload.matter_id)
            ).all()
        else:
            # Documents ingested without a matter (the common "paste an FIR then ask" flow)
            # must still reach the router, otherwise it classifies the request as a bare
            # question and the document's legal substance never informs the search query.
            documents = session.scalars(
                select(Document)
                .where(Document.matter_id.is_(None))
                .order_by(desc(Document.uploaded_at))
                .limit(3)
            ).all()
        for doc in documents:
            first_page = session.scalar(
                select(DocumentPage)
                .where(DocumentPage.document_id == doc.id)
                .order_by(DocumentPage.page_number)
            )
            excerpt = (first_page.text[:700] if first_page else "").replace("\n", " ")
            document_summaries.append(
                f"{doc.filename} [{doc.doc_class}, {doc.page_count} page(s)]: {excerpt}"
            )

    result = await orchestrator.run(
        question=payload.question,
        pasted_text=payload.pasted_text,
        matter_id=payload.matter_id,
        matter_title=matter_title,
        instructions=payload.instructions,
        enable_drafting=payload.enable_drafting,
        enable_verification=payload.enable_verification,
        document_summaries=document_summaries,
        draft_type=payload.draft_type,
        tone=payload.tone,
        length=payload.length,
    )
    return _serialise_result(result)


@app.get("/api/research/stream", tags=["research"])
async def research_stream(trace_id: str = Query(...)) -> StreamingResponse:
    """Server-sent events for live pipeline progress."""

    async def generator() -> Any:
        queue = await event_bus.subscribe(trace_id)
        try:
            yield f"data: {json.dumps({'type': 'connected', 'trace_id': trace_id})}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "trace_completed":
                    break
        finally:
            await event_bus.unsubscribe(trace_id, queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/retrieval/search", tags=["retrieval"])
async def retrieval_only(
    payload: QuickRetrievalRequest, principal: Principal = Depends(enforce_rate_limit)
) -> dict[str, Any]:
    """Retrieval without any agents, for the Retrieval Inspector.

    Retriever toggles travel on the request. They used to be applied by mutating the cached
    global Settings and restoring it in a `finally`, which meant two concurrent requests could
    overwrite each other's toggles and restore the wrong values.

    Rate limited: this embeds the query, so it occupies the same GPU as generation.
    """
    assert_matter_access(payload.matter_id, principal)
    response = await retrieval_search(
        RetrievalRequest(
            question=payload.question,
            mode=payload.mode,
            top_k=payload.top_k,
            matter_id=payload.matter_id,
            enable_keyword=payload.enable_keyword,
            enable_graph=payload.enable_graph,
            enable_dense=payload.enable_dense,
            min_similarity=payload.min_similarity,
        )
    )
    return response.model_dump(mode="json")


@app.get("/api/guardrails/config", tags=["guardrails"])
async def guardrail_config(
    principal: Principal = Depends(require_principal),
) -> dict[str, Any]:
    """What the guardrail stack currently enforces, and on what measured basis."""
    settings = get_settings()
    return {
        "enabled": settings.guardrails_enabled,
        "layers": [
            {
                "layer": "empty",
                "kind": "deterministic",
                "checks": "the request has content at all",
            },
            {
                "layer": "too_long",
                "kind": "deterministic",
                "checks": f"under half the {settings.llm_num_ctx}-token context window, so a "
                "long request is refused rather than silently truncated",
            },
            {
                "layer": "injection",
                "kind": "deterministic",
                "checks": "instructions aimed at the system rather than questions about law",
            },
            {
                "layer": "denylist",
                "kind": "deterministic",
                "checks": "unambiguously non-legal requests; overridden when the text carries a "
                "legal marker, so 'draft a written statement' is not refused",
            },
            {
                "layer": "corpus_distance",
                "kind": "embedding",
                "checks": f"best cosine similarity against the corpus reaches "
                f"{settings.guardrail_corpus_similarity_floor}; this is the layer that "
                "generalises beyond a keyword list",
            },
            {
                "layer": "llm_scope",
                "kind": "model",
                "checks": "the intake router's is_legal_matter verdict, applied last rather "
                "than as the only check",
            },
        ],
        "corpus_similarity_floor": settings.guardrail_corpus_similarity_floor,
        "floor_basis": (
            "Measured on this corpus: 0.81/0.78 for statutory questions it can answer, 0.62 for "
            "a legal question with no supporting material, 0.49 for a question about this "
            "repository's source files, 0.37-0.43 for general knowledge. 0.55 is the widest gap."
        ),
        "output_guardrail": (
            "Scores groundedness as grounded/total citations and refuses to present a "
            "high-confidence answer that cites nothing while chunks were retrieved."
        ),
        "auth_enabled": settings.auth_enabled,
        "rate_limit": f"{settings.rate_limit_requests} requests / "
        f"{settings.rate_limit_window_seconds:.0f}s per key",
        "llm_max_concurrency": settings.llm_max_concurrency,
    }


@app.post("/api/guardrails/probe", tags=["guardrails"])
async def guardrail_probe(
    check_corpus: bool = Query(True, description="Include the embedding-distance layer"),
    principal: Principal = Depends(require_principal),
) -> dict[str, Any]:
    """Run the adversarial probe suite and report catch rate and false-positive rate.

    Not rate limited: it embeds one short string per case, and being able to demonstrate the
    controls repeatedly is the point.
    """
    return await run_suite(check_corpus=check_corpus)


@app.post("/api/research/compare", tags=["research"])
async def compare_rag(
    payload: CompareRequest, principal: Principal = Depends(enforce_rate_limit)
) -> dict[str, Any]:
    """Same question with and without retrieval, so the difference is visible."""
    from legalaid.domain.contracts import LLMGenerateRequest
    from legalaid.services.llm import service as llm_service

    settings = get_settings()
    model = payload.model or settings.llm_model

    # This endpoint calls the model directly, with no agents and no grounding audit, so it
    # needs the scope gate more than the pipeline does - not less. It previously had none at
    # all, which made it the one place "write me a poem" was answered from model weights.
    if settings.guardrails_enabled:
        guard = await screen_input(payload.question)
        if not guard.allowed:
            raise HTTPException(
                422,
                {
                    "message": guard.message,
                    "blocked_by": guard.blocked_by.value if guard.blocked_by else None,
                    "guardrail": guard.to_dict(),
                },
            )

    no_rag = await llm_service.generate(
        LLMGenerateRequest(
            agent_key=AgentKey.DIRECT_LLM,
            system_prompt=(
                "You are a legal assistant for Indian law. Answer concisely from your own "
                "knowledge."
            ),
            user_prompt=payload.question,
            model=model,
            max_tokens=payload.max_tokens,
        )
    )

    retrieval = await retrieval_search(
        RetrievalRequest(question=payload.question, mode=RetrievalMode.LAW_SEARCH, top_k=5)
    )
    context_block = "\n\n".join(
        f"[{i}] {c.section or ''} {c.source_title}\n{c.text[:700]}"
        for i, c in enumerate(retrieval.chunks[:5], start=1)
    )
    with_rag = await llm_service.generate(
        LLMGenerateRequest(
            agent_key=AgentKey.DIRECT_LLM,
            system_prompt=(
                "You are a legal assistant for Indian law. Answer ONLY from the provided "
                "context. If the context does not answer the question, say so."
            ),
            user_prompt=f"CONTEXT\n{context_block}\n\nQUESTION\n{payload.question}",
            model=model,
            max_tokens=payload.max_tokens,
        )
    )

    return {
        "question": payload.question,
        "model": model,
        "without_rag": {
            "answer": no_rag.text,
            "input_tokens": no_rag.input_tokens,
            "output_tokens": no_rag.output_tokens,
            "latency_ms": no_rag.total_duration_ms,
            "sources": [],
        },
        "with_rag": {
            "answer": with_rag.text,
            "input_tokens": with_rag.input_tokens,
            "output_tokens": with_rag.output_tokens,
            "latency_ms": with_rag.total_duration_ms,
            "sources": [
                {
                    "chunk_id": c.chunk_id,
                    "section": c.section,
                    "source_title": c.source_title,
                    "similarity": c.similarity,
                    "source_url": c.source_url,
                }
                for c in retrieval.chunks[:5]
            ],
        },
        "retrieved_context": context_block,
    }


from legalaid.services.gateway.inspect import router as inspect_router  # noqa: E402
from legalaid.services.gateway.uploads import router as uploads_router  # noqa: E402

app.include_router(inspect_router)
app.include_router(uploads_router)


def _serialise_result(result: Any) -> dict[str, Any]:
    return {
        "trace_id": result.trace_id,
        "question": result.question,
        "total_ms": result.total_ms,
        "routing": result.routing.to_dict() if result.routing else None,
        "guardrail": result.guardrail.to_dict() if result.guardrail else None,
        "output_guardrail": (
            result.output_guardrail.to_dict() if result.output_guardrail else None
        ),
        "plan": result.plan,
        "answer": result.answer,
        "facts": result.facts,
        "legal": result.legal,
        "cases": result.cases,
        "draft": result.draft,
        "warnings": result.warnings,
        "retrieval": result.retrieval.model_dump(mode="json") if result.retrieval else None,
        "loop_iterations": [
            {
                "agent": i.agent,
                "iteration": i.iteration,
                "action": i.action,
                "audit": i.audit,
                "guidance": i.guidance,
            }
            for i in result.loop_iterations
        ],
        "agents": {
            key: {
                "agent_run_id": r.agent_run_id,
                "agent_key": r.agent_key.value,
                "label": AGENT_LABELS.get(r.agent_key, key),
                "model": r.model,
                "prompt": r.prompt_label,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
                "attempts": r.attempts,
                "status": r.status,
                "parse_ok": r.parse_ok,
                "parse_error": r.parse_error,
                "output": r.output,
                "system_prompt": r.system_prompt,
                "user_prompt": r.user_prompt,
                "raw_text": r.raw_text,
                "context": r.context.manifest(),
            }
            for key, r in result.agent_results.items()
        },
    }
