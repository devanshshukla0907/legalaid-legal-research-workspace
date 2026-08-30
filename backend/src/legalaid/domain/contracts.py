"""Service contracts.

These Pydantic models are the wire format for every inter-service call. Both the HTTP and
in-process transports use exactly these models, so there is no divergence between local
development and the Docker deployment.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from legalaid.domain.enums import (
    AgentKey,
    ChunkStrategy,
    CitationKind,
    Collection,
    DocumentClass,
    DraftLength,
    DraftTone,
    DraftType,
    ResearchMode,
    RetrievalMode,
    SourceBadge,
    StageStatus,
)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------- health


class ServiceHealth(Contract):
    service: str
    status: str
    version: str = "0.1.0"
    transport: str
    detail: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[ServiceHealth] = Field(default_factory=list)


# ------------------------------------------------------------------ llm calls


class LLMMessage(Contract):
    role: str
    content: str


class LLMGenerateRequest(Contract):
    trace_id: str | None = None
    agent_key: AgentKey = AgentKey.DIRECT_LLM
    agent_run_id: str | None = None
    system_prompt: str | None = None
    user_prompt: str
    temperature: float | None = None
    max_tokens: int | None = None
    model: str | None = None
    json_mode: bool = False
    # A JSON Schema here makes Ollama constrain decoding, guaranteeing schema-valid output.
    # Preferred over json_mode for the agents, which need specific shapes rather than merely
    # syntactically valid JSON.
    json_schema: dict[str, Any] | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    stop: list[str] | None = None


class LLMGenerateResponse(Contract):
    model_run_id: str
    provider: str
    model: str
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_duration_ms: float
    load_ms: float | None = None
    eval_ms: float | None = None
    tokens_per_second: float | None = None
    temperature: float
    max_tokens: int
    status: str = "success"
    error: str | None = None
    created_at: datetime


class OllamaModelInfo(Contract):
    name: str
    size_bytes: int | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    family: str | None = None
    modified_at: str | None = None


class LLMStatus(Contract):
    provider: str
    base_url: str
    reachable: bool
    configured_llm_model: str
    configured_embedding_model: str
    llm_model_available: bool
    embedding_model_available: bool
    installed_models: list[OllamaModelInfo] = Field(default_factory=list)
    error: str | None = None
    checked_at: datetime


# ------------------------------------------------------------------ embedding


class EmbedTextsRequest(Contract):
    trace_id: str | None = None
    texts: list[str]
    purpose: str = "document"
    model: str | None = None


class EmbeddingRecord(Contract):
    index: int
    dimensions: int
    norm: float
    latency_ms: float
    input_chars: int
    input_tokens: int | None = None
    preview: list[float]
    vector: list[float] | None = None


class EmbedTextsResponse(Contract):
    provider: str
    model: str
    dimensions: int
    purpose: str
    total_latency_ms: float
    embeddings: list[EmbeddingRecord]
    vectors: list[list[float]]


class EmbedChunksRequest(Contract):
    trace_id: str | None = None
    chunk_ids: list[str]
    collection: Collection


class EmbedChunksResponse(Contract):
    provider: str
    model: str
    dimensions: int
    embedded: int
    skipped: int
    total_latency_ms: float
    vector_ids: list[str]


# --------------------------------------------------------------------- chunks


class ChunkMetadata(Contract):
    chunk_id: str
    document_id: str | None = None
    provision_id: str | None = None
    case_id: str | None = None
    collection: Collection
    source_type: str
    source_title: str
    page: int | None = None
    section: str | None = None
    chapter: str | None = None
    heading: str | None = None
    text: str
    char_count: int
    token_count: int
    ordinal: int
    chunk_strategy: ChunkStrategy
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    vector_id: str | None = None
    matter_id: str | None = None
    source_url: str | None = None
    created_at: datetime | None = None
    version: str = "1"
    extra: dict[str, Any] = Field(default_factory=dict)


class ChunkDetail(Contract):
    chunk: ChunkMetadata
    document_filename: str | None = None
    retrieval_history: list[ChunkRetrievalHistory] = Field(default_factory=list)
    used_by_agents: list[str] = Field(default_factory=list)
    vector_preview: list[float] = Field(default_factory=list)
    embedding_latency_ms: float | None = None


class ChunkRetrievalHistory(Contract):
    trace_id: str
    retrieval_run_id: str
    collection: Collection
    rank: int
    similarity: float
    selected: bool
    question: str
    created_at: datetime | None = None


# ----------------------------------------------------------- vector service


class VectorUpsertItem(Contract):
    chunk_id: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorUpsertRequest(Contract):
    collection: Collection
    items: list[VectorUpsertItem]


class VectorUpsertResponse(Contract):
    collection: Collection
    upserted: int
    vector_ids: list[str]
    total_vectors: int
    dimensions: int


class VectorSearchRequest(Contract):
    collection: Collection
    vector: list[float]
    top_k: int
    filters: dict[str, Any] = Field(default_factory=dict)
    min_similarity: float | None = None


class VectorMatch(Contract):
    chunk_id: str
    vector_id: str
    similarity: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorSearchResponse(Contract):
    collection: Collection
    candidate_count: int
    filtered_count: int
    top_k: int
    search_ms: float
    matches: list[VectorMatch]
    dimensions: int


class VectorDeleteRequest(Contract):
    collection: Collection
    chunk_ids: list[str]


class VectorDeleteResponse(Contract):
    collection: Collection
    deleted: int
    total_vectors: int


class CollectionStats(Contract):
    collection: Collection
    vector_count: int
    dimensions: int
    index_type: str
    embedding_model: str | None = None
    last_updated: datetime | None = None
    size_bytes: int | None = None


class VectorStats(Contract):
    collections: list[CollectionStats]
    total_vectors: int
    index_dir: str


# --------------------------------------------------------------- retrieval


class RetrievalRequest(Contract):
    trace_id: str | None = None
    question: str
    mode: RetrievalMode
    matter_id: str | None = None
    top_k: int | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    collections: list[Collection] | None = None
    # Per-request retriever ablation, so the UI and the evaluation harness can measure each
    # retriever's contribution. These live on the request rather than in settings because the
    # endpoint used to mutate the process-wide cached Settings object: two concurrent requests
    # clobbered each other's toggles and restored the wrong values.
    # None means "use the configured default".
    enable_keyword: bool | None = None
    enable_graph: bool | None = None
    enable_dense: bool | None = None
    min_similarity: float | None = None
    # Fusion weights, overridable so the weight can be tuned by measurement rather than
    # intuition. None means "use the configured default".
    keyword_weight: float | None = None
    dense_weight: float | None = None


class RetrievedChunk(Contract):
    chunk_id: str
    collection: Collection
    rank: int
    similarity: float
    source_title: str
    section: str | None = None
    chapter: str | None = None
    page: int | None = None
    heading: str | None = None
    text: str
    source_url: str | None = None
    source_type: str
    token_count: int
    match_reasons: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionSearchTrace(Contract):
    """Per-collection search detail rendered by the Retrieval Inspector."""

    collection: Collection
    retrieval_run_id: str
    candidate_count: int
    filtered_count: int
    top_k: int
    returned: int
    min_similarity: float
    filters: dict[str, Any] = Field(default_factory=dict)
    embedding_ms: float
    search_ms: float
    rank_ms: float
    results: list[RetrievedChunk] = Field(default_factory=list)


class QueryEmbeddingTrace(Contract):
    provider: str
    model: str
    dimensions: int
    latency_ms: float
    input_chars: int
    input_tokens: int | None = None
    preview: list[float]
    vector_id: str | None = None


class RetrievalResponse(Contract):
    trace_id: str | None = None
    mode: RetrievalMode
    question: str
    query_embedding: QueryEmbeddingTrace | None = None
    searches: list[CollectionSearchTrace] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    total_ms: float
    context_chars: int = 0
    context_tokens: int = 0


# ------------------------------------------------------------------- agents


class AgentContextItem(Contract):
    """One item of context handed to an agent, or explicitly withheld from it."""

    kind: str
    ref_id: str | None = None
    label: str
    detail: str | None = None
    token_count: int | None = None
    similarity: float | None = None
    collection: Collection | None = None


class AgentContextManifest(Contract):
    agent_key: AgentKey
    agent_label: str
    included: list[AgentContextItem] = Field(default_factory=list)
    excluded: list[AgentContextItem] = Field(default_factory=list)
    exclusion_reasons: dict[str, str] = Field(default_factory=dict)
    included_chunk_ids: list[str] = Field(default_factory=list)
    excluded_chunk_ids: list[str] = Field(default_factory=list)
    total_context_tokens: int = 0
    summary: str = ""


class ToolInvocation(Contract):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: str
    duration_ms: float
    status: str = "success"
    error: str | None = None


class Citation(Contract):
    kind: CitationKind
    label: str
    chunk_id: str | None = None
    act_name: str | None = None
    section: str | None = None
    case_title: str | None = None
    court: str | None = None
    citation_text: str | None = None
    year: int | None = None
    date: str | None = None
    page: int | None = None
    document: str | None = None
    source: str | None = None
    source_url: str | None = None
    verified: bool = False
    verification_note: str | None = None
    agent_key: AgentKey | None = None


class ExtractedFacts(Contract):
    parties: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    documents_considered: list[str] = Field(default_factory=list)
    legal_issues: list[str] = Field(default_factory=list)
    provisions_mentioned: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    evidence_sufficient: bool = True


class LegalResearchSection(Contract):
    act_name: str
    section: str
    heading: str | None = None
    relevance: str
    chunk_id: str | None = None
    verified: bool = False


class LegalResearchOutput(Contract):
    legal_issues: list[str] = Field(default_factory=list)
    applicable_laws: list[str] = Field(default_factory=list)
    sections: list[LegalResearchSection] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence_sufficient: bool = True
    parse_status: str = "ok"
    raw_text: str | None = None


class SimilarCaseItem(Contract):
    case_title: str
    court: str | None = None
    citation: str | None = None
    date: str | None = None
    similarity_score: float | None = None
    why_similar: list[str] = Field(default_factory=list)
    relevant_principle: str | None = None
    outcome: str | None = None
    differences: list[str] = Field(default_factory=list)
    practical_relevance: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    source_url: str | None = None
    verified: bool = False


class SimilarCaseOutput(Contract):
    cases: list[SimilarCaseItem] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence_sufficient: bool = True
    parse_status: str = "ok"
    raw_text: str | None = None


class DraftingOutput(Contract):
    draft_type: DraftType
    title: str
    content: str
    grounding: list[str] = Field(default_factory=list)
    unverified_flags: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    parse_status: str = "ok"
    raw_text: str | None = None


class AgentRunRecord(Contract):
    agent_run_id: str
    trace_id: str
    agent_key: AgentKey
    agent_label: str
    status: str
    prompt_name: str | None = None
    prompt_version: str | None = None
    duration_ms: float
    context_manifest: AgentContextManifest | None = None
    tools_used: list[ToolInvocation] = Field(default_factory=list)
    model_runs: list[ModelRunRecord] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ModelRunRecord(Contract):
    model_run_id: str
    trace_id: str | None = None
    agent_run_id: str | None = None
    agent_key: AgentKey | None = None
    provider: str
    model: str
    role: str
    temperature: float
    max_tokens: int
    system_prompt: str | None = None
    user_prompt: str | None = None
    raw_response: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_duration_ms: float
    load_ms: float | None = None
    eval_ms: float | None = None
    tokens_per_second: float | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    status: str
    error: str | None = None
    created_at: datetime | None = None


# ---------------------------------------------------- agent service requests


class LegalResearchAgentRequest(Contract):
    trace_id: str
    question: str
    matter_id: str | None = None
    facts: ExtractedFacts | None = None
    retrieval: RetrievalResponse | None = None


class SimilarCaseAgentRequest(Contract):
    trace_id: str
    question: str
    matter_id: str | None = None
    facts: ExtractedFacts | None = None
    retrieval: RetrievalResponse | None = None


class DraftingAgentRequest(Contract):
    trace_id: str
    question: str
    matter_id: str | None = None
    instructions: str | None = None
    draft_type: DraftType = DraftType.CASE_SUMMARY
    tone: DraftTone = DraftTone.FORMAL
    length: DraftLength = DraftLength.STANDARD
    facts: ExtractedFacts | None = None
    legal_research: LegalResearchOutput | None = None
    similar_cases: SimilarCaseOutput | None = None


class DocumentUnderstandingRequest(Contract):
    trace_id: str
    matter_id: str
    question: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)


class AgentResponse(Contract):
    run: AgentRunRecord
    legal_research: LegalResearchOutput | None = None
    similar_cases: SimilarCaseOutput | None = None
    drafting: DraftingOutput | None = None
    facts: ExtractedFacts | None = None


# ------------------------------------------------------------------- traces


class TraceStageRecord(Contract):
    stage_id: str
    trace_id: str
    ordinal: int
    stage_key: str
    label: str
    service: str
    transport: str
    status: StageStatus
    duration_ms: float | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class AnswerSection(Contract):
    heading: str
    badge: SourceBadge
    body: str
    citations: list[Citation] = Field(default_factory=list)


class StructuredAnswer(Contract):
    summary: str
    sections: list[AnswerSection] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    disclaimer: str
    evidence_sufficient: bool = True
    markdown: str = ""


class PipelineTraceRecord(Contract):
    trace_id: str
    workspace_id: str | None = None
    matter_id: str | None = None
    mode: ResearchMode
    question: str
    status: str
    total_ms: float | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    stages: list[TraceStageRecord] = Field(default_factory=list)
    retrieval: RetrievalResponse | None = None
    agent_runs: list[AgentRunRecord] = Field(default_factory=list)
    model_runs: list[ModelRunRecord] = Field(default_factory=list)
    answer: StructuredAnswer | None = None
    capability_snapshot: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class ResearchQueryRequest(Contract):
    question: str
    matter_id: str | None = None
    mode: ResearchMode = ResearchMode.RAG
    retrieval_mode: RetrievalMode = RetrievalMode.COMBINED_LEGAL_RESEARCH
    include_drafting: bool = True
    draft_type: DraftType = DraftType.CASE_SUMMARY
    tone: DraftTone = DraftTone.FORMAL
    length: DraftLength = DraftLength.STANDARD
    instructions: str | None = None


class RagComparison(Contract):
    question: str
    with_rag: PipelineTraceRecord
    without_rag: PipelineTraceRecord
    differences: dict[str, Any] = Field(default_factory=dict)


class ResearchQueryResponse(Contract):
    trace: PipelineTraceRecord
    comparison: RagComparison | None = None


# ---------------------------------------------------------------- documents


class DocumentPageRecord(Contract):
    page_number: int
    text: str
    char_count: int
    headings: list[str] = Field(default_factory=list)


class DocumentRecord(Contract):
    document_id: str
    matter_id: str | None = None
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    doc_class: DocumentClass
    status: str
    page_count: int
    chunk_count: int = 0
    indexed_chunk_count: int = 0
    extraction_method: str | None = None
    is_synthetic: bool = False
    uploaded_at: datetime | None = None
    processed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentDetail(Contract):
    document: DocumentRecord
    pages: list[DocumentPageRecord] = Field(default_factory=list)
    chunks: list[ChunkMetadata] = Field(default_factory=list)
    structure: list[dict[str, Any]] = Field(default_factory=list)


class DocumentProcessRequest(Contract):
    trace_id: str | None = None
    document_id: str
    run_embedding: bool = True


class DocumentProcessResponse(Contract):
    document: DocumentRecord
    pages: int
    chunks: int
    embedded: int
    chunking_ms: float
    embedding_ms: float
    extraction_ms: float
    stages: list[TraceStageRecord] = Field(default_factory=list)


# ---------------------------------------------------------------- knowledge


class KnowledgeSourceRecord(Contract):
    source_id: str
    key: str
    name: str
    authority: str
    source_url: str | None = None
    licence: str | None = None
    attribution: str | None = None
    dataset_version: str | None = None
    retrieved_at: datetime | None = None
    item_count: int = 0
    chunk_count: int = 0


class IngestionRequest(Contract):
    sources: list[str] | None = None
    reindex: bool = False
    limit: int | None = None


class IngestionReport(Contract):
    source_key: str
    items_ingested: int
    chunks_created: int
    vectors_indexed: int
    duration_ms: float
    errors: list[str] = Field(default_factory=list)
    skipped: int = 0


class IngestionResponse(Contract):
    reports: list[IngestionReport]
    total_chunks: int
    total_vectors: int
    total_ms: float


# --------------------------------------------------------------- dashboard


class DashboardMetrics(Contract):
    matters: int
    documents: int
    knowledge_documents: int
    chunks: int
    vectors: int
    cases: int
    provisions: int
    queries_processed: int
    successful_requests: int
    failed_requests: int
    avg_embedding_ms: float | None = None
    avg_retrieval_ms: float | None = None
    avg_agent_ms: float | None = None
    avg_llm_ms: float | None = None
    avg_pipeline_ms: float | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    llm_status: LLMStatus | None = None
    recent_traces: list[dict[str, Any]] = Field(default_factory=list)


ServiceHealth.model_rebuild()
ChunkDetail.model_rebuild()
AgentRunRecord.model_rebuild()
