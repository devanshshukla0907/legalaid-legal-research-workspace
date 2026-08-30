"""Enumerations shared across every service and rendered by the UI."""

from __future__ import annotations

from enum import StrEnum


class Collection(StrEnum):
    """Separate knowledge collections. Never merged into one undifferentiated index."""

    LEGAL_ACTS = "LEGAL_ACTS"
    LEGAL_RULES = "LEGAL_RULES"
    LEGAL_REGULATIONS = "LEGAL_REGULATIONS"
    LEGAL_NOTIFICATIONS = "LEGAL_NOTIFICATIONS"
    CASE_LAW = "CASE_LAW"
    USER_DOCUMENTS = "USER_DOCUMENTS"


LAW_COLLECTIONS: tuple[Collection, ...] = (
    Collection.LEGAL_ACTS,
    Collection.LEGAL_RULES,
    Collection.LEGAL_REGULATIONS,
    Collection.LEGAL_NOTIFICATIONS,
)


class RetrievalMode(StrEnum):
    LAW_SEARCH = "LAW_SEARCH"
    CASE_SEARCH = "CASE_SEARCH"
    USER_DOCUMENT_SEARCH = "USER_DOCUMENT_SEARCH"
    COMBINED_LEGAL_RESEARCH = "COMBINED_LEGAL_RESEARCH"


class ResearchMode(StrEnum):
    RAG = "rag"
    NO_RAG = "no_rag"
    COMPARE = "compare"


class StageKey(StrEnum):
    """The canonical pipeline stages. Ordering here drives the UI trace list."""

    USER_INPUT = "USER_INPUT"
    API_GATEWAY = "API_GATEWAY"
    INTAKE_CLASSIFICATION = "INTAKE_CLASSIFICATION"
    QUERY_REWRITE = "QUERY_REWRITE"

    DOCUMENT_UNDERSTANDING = "DOCUMENT_UNDERSTANDING"
    QUERY_EMBEDDING = "QUERY_EMBEDDING"
    LAW_VECTOR_SEARCH = "LAW_VECTOR_SEARCH"
    CASE_VECTOR_SEARCH = "CASE_VECTOR_SEARCH"
    USER_DOC_SEARCH = "USER_DOC_SEARCH"
    CONTEXT_CONSTRUCTION = "CONTEXT_CONSTRUCTION"
    ORCHESTRATION_PLAN = "ORCHESTRATION_PLAN"

    LEGAL_RESEARCH_AGENT = "LEGAL_RESEARCH_AGENT"
    SIMILAR_CASE_AGENT = "SIMILAR_CASE_AGENT"
    DRAFTING_AGENT = "DRAFTING_AGENT"
    SYNTHESIS_AGENT = "SYNTHESIS_AGENT"
    INPUT_GUARDRAIL = "INPUT_GUARDRAIL"
    OUTPUT_GUARDRAIL = "OUTPUT_GUARDRAIL"
    VERIFICATION_AGENT = "VERIFICATION_AGENT"
    REFINEMENT_LOOP = "REFINEMENT_LOOP"
    CITATION_VALIDATION = "CITATION_VALIDATION"
    FINAL_RESPONSE = "FINAL_RESPONSE"
    # ingestion-side stages
    DOCUMENT_INGESTION = "DOCUMENT_INGESTION"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    VECTOR_UPSERT = "VECTOR_UPSERT"
    LLM_CALL = "LLM_CALL"
    TOKEN_GENERATION = "TOKEN_GENERATION"


STAGE_LABELS: dict[StageKey, str] = {
    StageKey.USER_INPUT: "User input",
    StageKey.API_GATEWAY: "API gateway",
    StageKey.INTAKE_CLASSIFICATION: "Intake classification",
    StageKey.QUERY_REWRITE: "Query rewrite for retrieval",

    StageKey.DOCUMENT_UNDERSTANDING: "Document understanding",
    StageKey.QUERY_EMBEDDING: "Query embedding",
    StageKey.LAW_VECTOR_SEARCH: "Law vector search",
    StageKey.CASE_VECTOR_SEARCH: "Case vector search",
    StageKey.USER_DOC_SEARCH: "User document retrieval",
    StageKey.CONTEXT_CONSTRUCTION: "Context construction",
    StageKey.ORCHESTRATION_PLAN: "Orchestration plan",

    StageKey.LEGAL_RESEARCH_AGENT: "Legal Research Agent",
    StageKey.SIMILAR_CASE_AGENT: "Similar Case Agent",
    StageKey.DRAFTING_AGENT: "Drafting Agent",
    StageKey.SYNTHESIS_AGENT: "Synthesis Agent — writing the answer",
    StageKey.INPUT_GUARDRAIL: "Input guardrails",
    StageKey.OUTPUT_GUARDRAIL: "Output guardrails",
    StageKey.VERIFICATION_AGENT: "Verification Agent",
    StageKey.REFINEMENT_LOOP: "Refinement loop",
    StageKey.CITATION_VALIDATION: "Citation validation",
    StageKey.FINAL_RESPONSE: "Final response",
    StageKey.DOCUMENT_INGESTION: "Document ingestion",
    StageKey.CHUNKING: "Chunking",
    StageKey.EMBEDDING: "Embedding",
    StageKey.VECTOR_UPSERT: "Vector storage",
    StageKey.LLM_CALL: "LLM call",
    StageKey.TOKEN_GENERATION: "Token generation",
}


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentKey(StrEnum):
    ORCHESTRATOR = "orchestrator"
    INTAKE_ROUTER = "intake_router"
    DOCUMENT_UNDERSTANDING = "document_understanding"
    LEGAL_RESEARCH = "legal_research"
    SIMILAR_CASES = "similar_cases"
    DRAFTING = "drafting"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"
    DIRECT_LLM = "direct_llm"


AGENT_LABELS: dict[AgentKey, str] = {
    AgentKey.ORCHESTRATOR: "Orchestrator",
    AgentKey.INTAKE_ROUTER: "Intake Router Agent",
    AgentKey.DOCUMENT_UNDERSTANDING: "Document Understanding Agent",
    AgentKey.LEGAL_RESEARCH: "Legal Research Agent",
    AgentKey.SIMILAR_CASES: "Similar Case Agent",
    AgentKey.DRAFTING: "Legal Drafting Agent",
    AgentKey.SYNTHESIS: "Synthesis Agent",
    AgentKey.VERIFICATION: "Verification Agent",
    AgentKey.DIRECT_LLM: "Direct LLM (no RAG)",
}


class DocumentClass(StrEnum):
    LEGAL_NOTICE = "legal_notice"
    AGREEMENT = "agreement"
    COMPLAINT = "complaint"
    REPLY = "reply"
    COURT_ORDER = "court_order"
    CLIENT_STATEMENT = "client_statement"
    JUDGMENT = "judgment"
    STATUTE = "statute"
    OTHER = "other"


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    CHUNKING = "chunking"
    CHUNKED = "chunked"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    FAILED = "failed"


class SourceBadge(StrEnum):
    """Answer-content classification, rendered with distinct colours in the UI."""

    FACT = "FACT"
    LAW = "LAW"
    CASE_PRECEDENT = "CASE_PRECEDENT"
    AI_INFERENCE = "AI_INFERENCE"
    UNCERTAINTY = "UNCERTAINTY"


class CitationKind(StrEnum):
    LAW = "law"
    CASE = "case"
    USER_DOC = "user_doc"


class ChunkStrategy(StrEnum):
    STATUTE_SECTION = "statute_section"
    JUDGMENT_SECTION = "judgment_section"
    DOCUMENT_PAGE_PARAGRAPH = "document_page_paragraph"
    FALLBACK_FIXED = "fallback_fixed"


class DraftType(StrEnum):
    NOTICE_REPLY = "notice_reply"
    PETITIONER_ARGUMENTS = "petitioner_arguments"
    WRITTEN_SUBMISSION = "written_submission"
    CASE_SUMMARY = "case_summary"
    RESEARCH_QUESTIONS = "research_questions"


class DraftTone(StrEnum):
    FORMAL = "formal"
    ASSERTIVE = "assertive"
    NEUTRAL = "neutral"
    CONCILIATORY = "conciliatory"


class DraftLength(StrEnum):
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"
