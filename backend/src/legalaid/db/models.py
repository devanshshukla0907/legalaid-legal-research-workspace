"""Relational models.

Two design rules worth stating explicitly:

1. **Firm-ready tenancy.** Organizations, roles, permissions, memberships, matter_members,
   document_permissions and audit_logs exist from day one. A Phase-1 lawyer is an
   organization of kind 'personal'. Adding firm mode later is authz + UI, not a migration.
2. **No bulk vectors in the RDBMS.** FAISS owns the vectors. `EmbeddingMetadata` keeps only a
   short preview for the UI inspector.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from legalaid.db.base import Base, IdColumn, new_id, utcnow

# --------------------------------------------------------------------------- #
# Tenancy and identity
# --------------------------------------------------------------------------- #


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("org_"))
    name: Mapped[str] = mapped_column(String(255))
    slug: Mapped[str] = mapped_column(String(120), unique=True)
    kind: Mapped[str] = mapped_column(String(32), default="personal")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    workspaces: Mapped[list[Workspace]] = relationship(back_populates="organization")
    roles: Mapped[list[Role]] = relationship(back_populates="organization")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("usr_"))
    email: Mapped[str] = mapped_column(String(255), unique=True)
    full_name: Mapped[str] = mapped_column(String(255))
    password_hash: Mapped[str | None] = mapped_column(String(255), default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Role(Base):
    __tablename__ = "roles"
    __table_args__ = (UniqueConstraint("organization_id", "name"),)

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("rol_"))
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"))
    name: Mapped[str] = mapped_column(String(64))
    is_system: Mapped[bool] = mapped_column(Boolean, default=False)

    organization: Mapped[Organization] = relationship(back_populates="roles")
    permissions: Mapped[list[RolePermission]] = relationship(back_populates="role")


class Permission(Base):
    __tablename__ = "permissions"

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("prm_"))
    code: Mapped[str] = mapped_column(String(96), unique=True)
    description: Mapped[str] = mapped_column(String(255), default="")


class RolePermission(Base):
    __tablename__ = "role_permissions"
    __table_args__ = (UniqueConstraint("role_id", "permission_id"),)

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("rp_"))
    role_id: Mapped[str] = mapped_column(ForeignKey("roles.id"))
    permission_id: Mapped[str] = mapped_column(ForeignKey("permissions.id"))

    role: Mapped[Role] = relationship(back_populates="permissions")
    permission: Mapped[Permission] = relationship()


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("organization_id", "user_id"),)

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("mem_"))
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    role_id: Mapped[str] = mapped_column(ForeignKey("roles.id"))
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("wsp_"))
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"))
    owner_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(32), default="personal")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    organization: Mapped[Organization] = relationship(back_populates="workspaces")
    matters: Mapped[list[Matter]] = relationship(back_populates="workspace")


# --------------------------------------------------------------------------- #
# Matters and documents
# --------------------------------------------------------------------------- #


class Matter(Base):
    __tablename__ = "matters"
    __table_args__ = (Index("ix_matters_workspace", "workspace_id"),)

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("mat_"))
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"))
    title: Mapped[str] = mapped_column(String(255))
    reference_no: Mapped[str | None] = mapped_column(String(96), default=None)
    matter_type: Mapped[str | None] = mapped_column(String(96), default=None)
    court: Mapped[str | None] = mapped_column(String(160), default=None)
    parties: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="active")
    extracted_facts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    legal_issues: Mapped[list[Any]] = mapped_column(JSON, default=list)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    workspace: Mapped[Workspace] = relationship(back_populates="matters")
    documents: Mapped[list[Document]] = relationship(
        back_populates="matter", cascade="all, delete-orphan"
    )


class MatterMember(Base):
    """Reserved for firm mode; unused in Phase 1 but present so schema is forward-compatible."""

    __tablename__ = "matter_members"
    __table_args__ = (UniqueConstraint("matter_id", "user_id"),)

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("mm_"))
    matter_id: Mapped[str] = mapped_column(ForeignKey("matters.id"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    role_id: Mapped[str | None] = mapped_column(ForeignKey("roles.id"), default=None)
    access_level: Mapped[str] = mapped_column(String(32), default="read")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (Index("ix_documents_matter", "matter_id"),)

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("doc_"))
    matter_id: Mapped[str | None] = mapped_column(ForeignKey("matters.id"), default=None)
    workspace_id: Mapped[str | None] = mapped_column(ForeignKey("workspaces.id"), default=None)
    filename: Mapped[str] = mapped_column(String(512))
    stored_path: Mapped[str] = mapped_column(String(1024))
    content_type: Mapped[str] = mapped_column(String(128), default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64), default="")
    doc_class: Mapped[str] = mapped_column(String(64), default="other")
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    extraction_method: Mapped[str | None] = mapped_column(String(64), default=None)
    doc_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    uploaded_at: Mapped[datetime] = mapped_column(default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(default=None)

    matter: Mapped[Matter | None] = relationship(back_populates="documents")
    pages: Mapped[list[DocumentPage]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentPage(Base):
    __tablename__ = "document_pages"
    __table_args__ = (
        UniqueConstraint("document_id", "page_number"),
        Index("ix_document_pages_doc", "document_id"),
    )

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("pag_"))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    page_number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, default="")
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    headings: Mapped[list[Any]] = mapped_column(JSON, default=list)
    extraction_conf: Mapped[float | None] = mapped_column(Float, default=None)

    document: Mapped[Document] = relationship(back_populates="pages")


class DocumentPermission(Base):
    """Reserved for firm mode."""

    __tablename__ = "document_permissions"

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("dp_"))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    role_id: Mapped[str | None] = mapped_column(ForeignKey("roles.id"), default=None)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), default=None)
    access_level: Mapped[str] = mapped_column(String(32), default="read")


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #


class KnowledgeSource(Base):
    __tablename__ = "knowledge_sources"

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("src_"))
    key: Mapped[str] = mapped_column(String(96), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    authority: Mapped[str] = mapped_column(String(255), default="")
    source_url: Mapped[str | None] = mapped_column(String(1024), default=None)
    licence: Mapped[str | None] = mapped_column(String(255), default=None)
    attribution: Mapped[str | None] = mapped_column(Text, default=None)
    dataset_version: Mapped[str | None] = mapped_column(String(96), default=None)
    retrieved_at: Mapped[datetime | None] = mapped_column(default=None)
    item_count: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str] = mapped_column(Text, default="")


class LegalProvision(Base):
    """A statutory provision: Act -> Chapter -> Section -> Subsection."""

    __tablename__ = "legal_provisions"
    __table_args__ = (Index("ix_provisions_collection", "collection"),)

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("prv_"))
    source_id: Mapped[str | None] = mapped_column(ForeignKey("knowledge_sources.id"), default=None)
    collection: Mapped[str] = mapped_column(String(48), default="LEGAL_ACTS")
    act_name: Mapped[str] = mapped_column(String(320))
    act_year: Mapped[int | None] = mapped_column(Integer, default=None)
    chapter: Mapped[str | None] = mapped_column(String(255), default=None)
    section_number: Mapped[str | None] = mapped_column(String(64), default=None)
    section_heading: Mapped[str | None] = mapped_column(String(512), default=None)
    subsection: Mapped[str | None] = mapped_column(String(64), default=None)
    text: Mapped[str] = mapped_column(Text, default="")
    source_url: Mapped[str | None] = mapped_column(String(1024), default=None)
    source_authority: Mapped[str] = mapped_column(String(255), default="")
    document_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    page: Mapped[int | None] = mapped_column(Integer, default=None)
    version: Mapped[str] = mapped_column(String(64), default="1")
    amendment_note: Mapped[str | None] = mapped_column(Text, default=None)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    retrieved_at: Mapped[datetime | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("cse_"))
    source_id: Mapped[str | None] = mapped_column(ForeignKey("knowledge_sources.id"), default=None)
    case_ref: Mapped[str] = mapped_column(String(160), default="")
    case_title: Mapped[str] = mapped_column(String(640))
    court: Mapped[str | None] = mapped_column(String(255), default=None)
    citation: Mapped[str | None] = mapped_column(String(320), default=None)
    neutral_citation: Mapped[str | None] = mapped_column(String(320), default=None)
    decision_date: Mapped[str | None] = mapped_column(String(48), default=None)
    year: Mapped[int | None] = mapped_column(Integer, default=None)
    judges: Mapped[list[Any]] = mapped_column(JSON, default=list)
    legal_provisions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    keywords: Mapped[list[Any]] = mapped_column(JSON, default=list)
    facts_summary: Mapped[str] = mapped_column(Text, default="")
    issues: Mapped[str] = mapped_column(Text, default="")
    arguments: Mapped[str] = mapped_column(Text, default="")
    reasoning: Mapped[str] = mapped_column(Text, default="")
    decision: Mapped[str] = mapped_column(Text, default="")
    ratio: Mapped[str] = mapped_column(Text, default="")
    disposal_nature: Mapped[str | None] = mapped_column(String(160), default=None)
    source_url: Mapped[str | None] = mapped_column(String(1024), default=None)
    document_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    licence: Mapped[str | None] = mapped_column(String(255), default=None)
    dataset: Mapped[str | None] = mapped_column(String(160), default=None)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    retrieved_at: Mapped[datetime | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


# --------------------------------------------------------------------------- #
# Chunks and embeddings
# --------------------------------------------------------------------------- #


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_collection", "collection"),
        Index("ix_chunks_document", "document_id"),
        Index("ix_chunks_matter", "matter_id"),
    )

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("chk_"))
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), default=None)
    provision_id: Mapped[str | None] = mapped_column(ForeignKey("legal_provisions.id"), default=None)
    case_id: Mapped[str | None] = mapped_column(ForeignKey("cases.id"), default=None)
    matter_id: Mapped[str | None] = mapped_column(ForeignKey("matters.id"), default=None)
    collection: Mapped[str] = mapped_column(String(48))
    source_type: Mapped[str] = mapped_column(String(64), default="")
    source_title: Mapped[str] = mapped_column(String(640), default="")
    section: Mapped[str | None] = mapped_column(String(128), default=None)
    chapter: Mapped[str | None] = mapped_column(String(255), default=None)
    page: Mapped[int | None] = mapped_column(Integer, default=None)
    heading: Mapped[str | None] = mapped_column(String(512), default=None)
    text: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    parent_chunk_id: Mapped[str | None] = mapped_column(IdColumn, default=None)
    chunk_strategy: Mapped[str] = mapped_column(String(64), default="fallback_fixed")
    source_url: Mapped[str | None] = mapped_column(String(1024), default=None)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    version: Mapped[str] = mapped_column(String(32), default="1")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    embedding: Mapped[EmbeddingMetadata | None] = relationship(
        back_populates="chunk", cascade="all, delete-orphan", uselist=False
    )


class EmbeddingMetadata(Base):
    """Embedding bookkeeping. The vector itself lives in FAISS, not here."""

    __tablename__ = "embeddings_metadata"
    __table_args__ = (Index("ix_embeddings_chunk", "chunk_id"),)

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("emb_"))
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"), unique=True)
    vector_id: Mapped[str] = mapped_column(String(96))
    collection: Mapped[str] = mapped_column(String(48))
    embedding_provider: Mapped[str] = mapped_column(String(64))
    embedding_model: Mapped[str] = mapped_column(String(160))
    dimensions: Mapped[int] = mapped_column(Integer)
    norm: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    input_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    vector_preview: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    chunk: Mapped[Chunk] = relationship(back_populates="embedding")


# --------------------------------------------------------------------------- #
# Observability
# --------------------------------------------------------------------------- #


class PipelineTrace(Base):
    __tablename__ = "pipeline_traces"

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True)
    workspace_id: Mapped[str | None] = mapped_column(ForeignKey("workspaces.id"), default=None)
    matter_id: Mapped[str | None] = mapped_column(ForeignKey("matters.id"), default=None)
    mode: Mapped[str] = mapped_column(String(32), default="rag")
    question: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="running")
    started_at: Mapped[datetime] = mapped_column(default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    total_ms: Mapped[float | None] = mapped_column(Float, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    capability_snapshot: Mapped[list[Any]] = mapped_column(JSON, default=list)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    answer: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    parent_trace_id: Mapped[str | None] = mapped_column(IdColumn, default=None)

    stages: Mapped[list[TraceStage]] = relationship(
        back_populates="trace", cascade="all, delete-orphan"
    )


class TraceStage(Base):
    __tablename__ = "trace_stages"
    __table_args__ = (Index("ix_trace_stages_trace_ordinal", "trace_id", "ordinal"),)

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("stg_"))
    trace_id: Mapped[str] = mapped_column(ForeignKey("pipeline_traces.id"))
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    stage_key: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(160), default="")
    service: Mapped[str] = mapped_column(String(64), default="")
    transport: Mapped[str] = mapped_column(String(32), default="inprocess")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    started_at: Mapped[datetime] = mapped_column(default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    duration_ms: Mapped[float | None] = mapped_column(Float, default=None)
    input: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    output: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    trace: Mapped[PipelineTrace] = relationship(back_populates="stages")


class RetrievalRun(Base):
    __tablename__ = "retrieval_runs"
    __table_args__ = (Index("ix_retrieval_runs_trace", "trace_id"),)

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("ret_"))
    trace_id: Mapped[str | None] = mapped_column(ForeignKey("pipeline_traces.id"), default=None)
    mode: Mapped[str] = mapped_column(String(48), default="")
    collection: Mapped[str] = mapped_column(String(48), default="")
    query_text: Mapped[str] = mapped_column(Text, default="")
    query_vector_id: Mapped[str | None] = mapped_column(String(96), default=None)
    top_k: Mapped[int] = mapped_column(Integer, default=0)
    candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    filtered_count: Mapped[int] = mapped_column(Integer, default=0)
    returned_count: Mapped[int] = mapped_column(Integer, default=0)
    min_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    filters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    embedding_provider: Mapped[str] = mapped_column(String(64), default="")
    embedding_model: Mapped[str] = mapped_column(String(160), default="")
    dimensions: Mapped[int] = mapped_column(Integer, default=0)
    embedding_ms: Mapped[float] = mapped_column(Float, default=0.0)
    search_ms: Mapped[float] = mapped_column(Float, default=0.0)
    rank_ms: Mapped[float] = mapped_column(Float, default=0.0)
    query_vector_preview: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    results: Mapped[list[RetrievalResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class RetrievalResult(Base):
    __tablename__ = "retrieval_results"
    __table_args__ = (Index("ix_retrieval_results_run_rank", "retrieval_run_id", "rank"),)

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("rr_"))
    retrieval_run_id: Mapped[str] = mapped_column(ForeignKey("retrieval_runs.id"))
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"))
    rank: Mapped[int] = mapped_column(Integer, default=0)
    similarity: Mapped[float] = mapped_column(Float, default=0.0)
    selected: Mapped[bool] = mapped_column(Boolean, default=True)
    match_reasons: Mapped[list[Any]] = mapped_column(JSON, default=list)
    matched_terms: Mapped[list[Any]] = mapped_column(JSON, default=list)

    run: Mapped[RetrievalRun] = relationship(back_populates="results")


class AgentRun(Base):
    """One agent execution.

    `excluded_chunk_ids` is the persisted proof of context isolation: it records what the
    agent was deliberately NOT shown.
    """

    __tablename__ = "agent_runs"
    __table_args__ = (Index("ix_agent_runs_trace", "trace_id"),)

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("agr_"))
    trace_id: Mapped[str] = mapped_column(ForeignKey("pipeline_traces.id"))
    agent_key: Mapped[str] = mapped_column(String(64))
    agent_label: Mapped[str] = mapped_column(String(160), default="")
    status: Mapped[str] = mapped_column(String(32), default="running")
    started_at: Mapped[datetime] = mapped_column(default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    duration_ms: Mapped[float | None] = mapped_column(Float, default=None)
    prompt_name: Mapped[str | None] = mapped_column(String(160), default=None)
    prompt_version: Mapped[str | None] = mapped_column(String(48), default=None)
    included_chunk_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    excluded_chunk_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    context_manifest: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tools_used: Mapped[list[Any]] = mapped_column(JSON, default=list)
    input_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    output: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, default=None)


class ModelRun(Base):
    """One LLM call. Stores the actual prompts so 'what did the model see?' reads real data."""

    __tablename__ = "model_runs"
    __table_args__ = (Index("ix_model_runs_trace", "trace_id"),)

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("mdl_"))
    trace_id: Mapped[str | None] = mapped_column(ForeignKey("pipeline_traces.id"), default=None)
    agent_run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id"), default=None)
    agent_key: Mapped[str | None] = mapped_column(String(64), default=None)
    provider: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str] = mapped_column(String(160), default="")
    role: Mapped[str] = mapped_column(String(48), default="generation")
    temperature: Mapped[float] = mapped_column(Float, default=0.0)
    max_tokens: Mapped[int] = mapped_column(Integer, default=0)
    system_prompt: Mapped[str | None] = mapped_column(Text, default=None)
    user_prompt: Mapped[str | None] = mapped_column(Text, default=None)
    raw_response: Mapped[str | None] = mapped_column(Text, default=None)
    input_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    output_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    total_duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    load_ms: Mapped[float | None] = mapped_column(Float, default=None)
    eval_ms: Mapped[float | None] = mapped_column(Float, default=None)
    tokens_per_second: Mapped[float | None] = mapped_column(Float, default=None)
    prompt_name: Mapped[str | None] = mapped_column(String(160), default=None)
    prompt_version: Mapped[str | None] = mapped_column(String(48), default=None)
    status: Mapped[str] = mapped_column(String(32), default="success")
    error: Mapped[str | None] = mapped_column(Text, default=None)
    request_id: Mapped[str | None] = mapped_column(String(96), default=None)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Draft(Base):
    __tablename__ = "drafts"

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("drf_"))
    matter_id: Mapped[str | None] = mapped_column(ForeignKey("matters.id"), default=None)
    trace_id: Mapped[str | None] = mapped_column(ForeignKey("pipeline_traces.id"), default=None)
    draft_type: Mapped[str] = mapped_column(String(64), default="case_summary")
    tone: Mapped[str] = mapped_column(String(32), default="formal")
    length: Mapped[str] = mapped_column(String(32), default="standard")
    title: Mapped[str] = mapped_column(String(512), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    grounding: Mapped[list[Any]] = mapped_column(JSON, default=list)
    unverified_flags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class CitationRecord(Base):
    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("cit_"))
    trace_id: Mapped[str | None] = mapped_column(ForeignKey("pipeline_traces.id"), default=None)
    kind: Mapped[str] = mapped_column(String(32), default="law")
    chunk_id: Mapped[str | None] = mapped_column(ForeignKey("chunks.id"), default=None)
    label: Mapped[str] = mapped_column(String(640), default="")
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verification_note: Mapped[str | None] = mapped_column(Text, default=None)
    agent_key: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=lambda: new_id("aud_"))
    organization_id: Mapped[str | None] = mapped_column(
        ForeignKey("organizations.id"), default=None
    )
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), default=None)
    action: Mapped[str] = mapped_column(String(96))
    entity_type: Mapped[str] = mapped_column(String(64), default="")
    entity_id: Mapped[str | None] = mapped_column(IdColumn, default=None)
    trace_id: Mapped[str | None] = mapped_column(IdColumn, default=None)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
