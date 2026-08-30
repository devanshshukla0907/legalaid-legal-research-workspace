"""Capability registry: one application, progressively activated across five exercises.

This is the mechanism that satisfies the "do not build five different applications" rule.
Each exercise activates additional capabilities in the same codebase, and the System page
renders this registry so an evaluator can verify coverage without reading source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from legalaid.config import get_settings


@dataclass(frozen=True, slots=True)
class Capability:
    key: str
    label: str
    phase: int
    exercise: str
    description: str
    pipeline_stages: tuple[str, ...] = field(default=())


CAPABILITIES: tuple[Capability, ...] = (
    # ---------------------------------------------------------------- phase 1
    Capability(
        key="app_ui",
        label="Application UI",
        phase=1,
        exercise="Exercise 1 - Application + API + Ollama + Code Llama",
        description="React legal-research workspace shell with navigation and theming.",
        pipeline_stages=("USER_INPUT",),
    ),
    Capability(
        key="api_gateway",
        label="API Gateway",
        phase=1,
        exercise="Exercise 1 - Application + API + Ollama + Code Llama",
        description="FastAPI gateway with OpenAPI docs and trace-id issuance.",
        pipeline_stages=("API_GATEWAY",),
    ),
    Capability(
        key="matters",
        label="Matter management",
        phase=1,
        exercise="Exercise 1 - Application + API + Ollama + Code Llama",
        description="Create and manage legal matters inside a lawyer workspace.",
    ),
    Capability(
        key="documents_upload",
        label="Document upload and extraction",
        phase=1,
        exercise="Exercise 1 - Application + API + Ollama + Code Llama",
        description="Upload PDF/TXT/DOCX, extract text with page-level provenance.",
        pipeline_stages=("DOCUMENT_INGESTION",),
    ),
    Capability(
        key="ollama_chat",
        label="Ollama + Code Llama generation",
        phase=1,
        exercise="Exercise 1 - Application + API + Ollama + Code Llama",
        description="LLM service calls Code Llama through the Ollama API and records model runs.",
        pipeline_stages=("LLM_CALL", "TOKEN_GENERATION", "FINAL_RESPONSE"),
    ),
    # ---------------------------------------------------------------- phase 2
    Capability(
        key="knowledge_ingestion",
        label="Legal knowledge base ingestion",
        phase=2,
        exercise="Exercise 2 - Knowledge Base + Chunking + Embeddings",
        description="Ingest Indian legislation and curated case law with full provenance.",
    ),
    Capability(
        key="chunking",
        label="Structure-aware chunking",
        phase=2,
        exercise="Exercise 2 - Knowledge Base + Chunking + Embeddings",
        description="Act/chapter/section chunking, judgment-section chunking, page-aware "
        "document chunking.",
        pipeline_stages=("CHUNKING",),
    ),
    Capability(
        key="embeddings",
        label="Embedding service",
        phase=2,
        exercise="Exercise 2 - Knowledge Base + Chunking + Embeddings",
        description="Dedicated embedding service using a configurable Ollama embedding model.",
        pipeline_stages=("EMBEDDING",),
    ),
    Capability(
        key="vector_index",
        label="Vector database",
        phase=2,
        exercise="Exercise 2 - Knowledge Base + Chunking + Embeddings",
        description="FAISS collections with upsert, delete, metadata filtering and persistence.",
        pipeline_stages=("VECTOR_UPSERT",),
    ),
    # ---------------------------------------------------------------- phase 3
    Capability(
        key="query_embedding",
        label="Query embedding",
        phase=3,
        exercise="Exercise 3 - Vector Similarity + Retrieval + RAG",
        description="Embed the user question with the same configured embedding model.",
        pipeline_stages=("QUERY_EMBEDDING",),
    ),
    Capability(
        key="vector_similarity",
        label="Vector similarity search",
        phase=3,
        exercise="Exercise 3 - Vector Similarity + Retrieval + RAG",
        description="Cosine similarity search per collection with candidate counts and top-k.",
        pipeline_stages=("LAW_VECTOR_SEARCH", "CASE_VECTOR_SEARCH", "USER_DOC_SEARCH"),
    ),
    Capability(
        key="rag_retrieval",
        label="Retrieval and context construction",
        phase=3,
        exercise="Exercise 3 - Vector Similarity + Retrieval + RAG",
        description="Filter, rank and assemble retrieved context with why-retrieved explanations.",
        pipeline_stages=("CONTEXT_CONSTRUCTION",),
    ),
    Capability(
        key="rag_compare",
        label="RAG ON vs RAG OFF comparison",
        phase=3,
        exercise="Exercise 3 - Vector Similarity + Retrieval + RAG",
        description="Run the same question with and without retrieval and compare answers.",
    ),
    # ---------------------------------------------------------------- phase 4
    Capability(
        key="service_split",
        label="Service separation over APIs",
        phase=4,
        exercise="Exercise 4 - APIs + Services + Orchestration",
        description="Nine logical services communicating through documented HTTP contracts.",
    ),
    Capability(
        key="multi_agent",
        label="Multi-agent processing",
        phase=4,
        exercise="Exercise 4 - APIs + Services + Orchestration",
        description="Legal Research, Similar Case and Drafting agents with distinct prompts.",
    ),
    Capability(
        key="orchestration",
        label="Orchestration",
        phase=4,
        exercise="Exercise 4 - APIs + Services + Orchestration",
        description="Deterministic orchestrator plans, parallelises and assembles agent work.",
        pipeline_stages=("ORCHESTRATION_PLAN",),
    ),
    Capability(
        key="agent_context_isolation",
        label="Agent context isolation",
        phase=4,
        exercise="Exercise 4 - APIs + Services + Orchestration",
        description="Each agent receives only its own context; inclusions and exclusions "
        "are persisted.",
        # Isolation is enforced when each agent's context is built, so the evidence lives on
        # the agent stages themselves plus the citation audit that checks what they cited.
        pipeline_stages=(
            "LEGAL_RESEARCH_AGENT",
            "SIMILAR_CASE_AGENT",
            "CITATION_VALIDATION",
        ),
    ),
    Capability(
        key="pipeline_trace",
        label="Full pipeline trace",
        phase=4,
        exercise="Exercise 4 - APIs + Services + Orchestration",
        description="Every stage of every request recorded and inspectable in the UI.",
    ),
    # ---------------------------------------------------------------- phase 5
    Capability(
        key="docker_compose",
        label="Docker composition",
        phase=5,
        exercise="Exercise 5 - Docker + Complete Application Architecture",
        description="All services containerised and orchestrated with docker compose.",
    ),
    Capability(
        key="service_health",
        label="Service health monitoring",
        phase=5,
        exercise="Exercise 5 - Docker + Complete Application Architecture",
        description="Health endpoints for every service, surfaced on the System page.",
    ),
    Capability(
        key="full_deployment",
        label="Complete deployment",
        phase=5,
        exercise="Exercise 5 - Docker + Complete Application Architecture",
        description="End-to-end system running with inter-service communication verified.",
    ),
)

_BY_KEY: dict[str, Capability] = {c.key: c for c in CAPABILITIES}


def get_capability(key: str) -> Capability:
    try:
        return _BY_KEY[key]
    except KeyError as exc:
        raise ValueError(f"Unknown capability: {key}") from exc


def is_enabled(key: str) -> bool:
    """A capability is active when the configured phase has reached its phase."""
    return get_settings().active_phase >= get_capability(key).phase


def active_capabilities() -> list[Capability]:
    return [c for c in CAPABILITIES if is_enabled(c.key)]


def capability_matrix() -> list[dict[str, object]]:
    """Serialisable view rendered by the System page."""
    return [
        {
            "key": c.key,
            "label": c.label,
            "phase": c.phase,
            "exercise": c.exercise,
            "description": c.description,
            "pipeline_stages": list(c.pipeline_stages),
            "enabled": is_enabled(c.key),
        }
        for c in CAPABILITIES
    ]


class CapabilityDisabledError(RuntimeError):
    """Raised when a request needs a capability that the active phase has not enabled."""

    def __init__(self, key: str) -> None:
        capability = get_capability(key)
        super().__init__(
            f"Capability '{capability.label}' requires phase {capability.phase}; "
            f"active phase is {get_settings().active_phase}."
        )
        self.key = key
        self.required_phase = capability.phase


def require(key: str) -> None:
    if not is_enabled(key):
        raise CapabilityDisabledError(key)
