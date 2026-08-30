"""Central typed configuration for every LegalAId service.

Every tunable - model names, ports, paths, chunk sizes, top-k values - lives here and is
sourced from the environment. No model name or machine-specific path is hardcoded anywhere
else in the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RepoRoot = Path(__file__).resolve().parents[3]

TransportMode = Literal["http", "inprocess"]


class Settings(BaseSettings):
    """Application settings, populated from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=(RepoRoot / ".env", Path(".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- general
    app_name: str = "LegalAId"
    environment: Literal["dev", "docker", "test"] = "dev"
    log_level: str = "INFO"
    active_phase: int = Field(default=5, ge=1, le=5)

    # ------------------------------------------------------------- llm/ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_provider: str = "ollama"
    llm_model: str = "codellama:7b-instruct"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_request_timeout: float = 300.0
    # 3072 is a measured compromise on a 6 GB GPU: 2048 -> 15.6 tok/s but too little room for
    # retrieved context, 4096 -> 9.5 tok/s because the model stops fully offloading to GPU.
    llm_num_ctx: int = 3072

    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768
    # Ollama embedding latency is ~2.2 s of fixed per-request overhead, almost independent of
    # input length, so batching dominates ingestion speed. Measured on this machine:
    # batch 1 -> 2249 ms/chunk, batch 16 -> 161, batch 32 -> 91, batch 64 -> 54 (a 42x gain).
    # Batched vectors are identical to single-call vectors (cosine 1.000000), so this is a
    # pure throughput win. Requires the /api/embed endpoint.
    embedding_batch_size: int = 32
    embedding_request_timeout: float = 300.0

    # -------------------------------------------------------------- retrieval
    retrieval_top_k_laws: int = 8
    retrieval_top_k_cases: int = 5
    retrieval_top_k_user_docs: int = 6
    retrieval_min_similarity: float = 0.25
    retrieval_candidate_multiplier: int = 4

    # ---------------------------------------------------- hybrid + graph rag
    # Three complementary retrievers, each independently switchable so the UI can demonstrate
    # their contribution (and so RAG ablations are possible):
    #   dense   - semantic similarity over embeddings
    #   keyword - BM25, which is what catches exact tokens like "Section 73"
    #   graph   - symbolic traversal of act/section/judgment relationships, no vectors at all
    retrieval_enable_keyword: bool = True
    retrieval_enable_graph: bool = True
    # Semantic match is the primary signal for natural-language questions; BM25 acts as a
    # precision correction, so it is weighted lower in rank fusion.
    retrieval_dense_weight: float = 1.0
    # 0.15, measured by eval/weight_sweep.py over 22 tasks with known expected sections:
    #
    #   weight  0.00   0.05   0.10   0.15   0.20   0.30   0.45   0.60+
    #   recall  0.614  0.659  0.636  0.682  0.659  0.614  0.568  0.568
    #
    # This was 0.6, chosen by reasoning that semantic match should dominate. The reasoning was
    # right and the number was wrong: at 0.6 BM25 made retrieval *worse than disabling it*
    # (0.568 vs 0.614). Reciprocal Rank Fusion adds both retrievers' contributions, so a
    # document dense-ranked 20th but keyword-ranked 1st outscores one dense-ranked 5th
    # (0.02234 vs 0.01538). That is correct behaviour when both retrievers are strong; on this
    # corpus BM25 without an explicit section number in the query matches generic legal
    # vocabulary, and those spurious agreements promoted noise over good semantic hits.
    # At 0.15 the agreement bonus is small enough to break ties without reordering, and BM25 is
    # worth +0.068 recall over dense alone.
    retrieval_keyword_weight: float = 0.15
    retrieval_graph_hops: int = 2
    retrieval_graph_limit: int = 20

    # ------------------------------------------------------------ orchestration
    # Refinement loop: an agent whose output fails the grounding audit is re-run with
    # corrective guidance. 2 is a deliberate ceiling - at ~12 tok/s every extra iteration
    # costs 30-60 s, and in practice a second attempt fixes citation errors or nothing will.
    orchestrator_max_loop_iterations: int = 2

    # ------------------------------------------------------------- guardrails
    guardrails_enabled: bool = True
    # Best cosine similarity the question must reach against the corpus before the request is
    # treated as answerable. Measured on this corpus with scripts/check_guardrails.py:
    #   0.81, 0.78  statutory questions answerable from the corpus
    #   0.62        a legal question with no supporting material (GST thresholds)
    #   0.49        a question about this repository's own source files
    #   0.43, 0.40, 0.37  general knowledge, nonsense, Python
    # 0.55 is the widest gap in that distribution. It keeps genuine legal questions - including
    # ones the corpus cannot answer, which should proceed and then report insufficient
    # evidence - while excluding everything off-domain. An earlier 0.45 admitted the 0.49
    # code question, which is precisely the confident-but-irrelevant failure Exercise 6 found.
    guardrail_corpus_similarity_floor: float = 0.55
    # Requests per window, per API key. One research request is 4-6 LLM calls on a single GPU,
    # so this is capacity management rather than abuse prevention.
    rate_limit_requests: int = 12
    rate_limit_window_seconds: float = 60.0
    # Only one generation can occupy the GPU at a time; queued callers wait rather than
    # thrashing VRAM and forcing the model to spill to CPU.
    llm_max_concurrency: int = 1
    llm_queue_timeout: float = 240.0

    # ------------------------------------------------------------------- auth
    auth_enabled: bool = True
    # Comma-separated keys. Empty means a development key is generated and logged at startup,
    # so the app is never accidentally deployed wide open with a well-known default.
    api_keys: list[str] = Field(default_factory=list)

    # --------------------------------------------------------------- chunking
    chunk_target_chars: int = 1400
    chunk_max_chars: int = 2600
    chunk_overlap_chars: int = 180
    chunk_min_chars: int = 120

    # --------------------------------------------------------------- database
    database_url: str = ""

    # ------------------------------------------------------------------ paths
    data_dir: Path = RepoRoot / "data"
    upload_dir: Path = RepoRoot / "data" / "uploads"
    index_dir: Path = RepoRoot / "data" / "indexes"
    knowledge_dir: Path = RepoRoot / "data" / "knowledge_sources"
    prompt_dir: Path = RepoRoot / "prompts"

    # --------------------------------------------------------------- services
    service_transport: TransportMode = "inprocess"
    gateway_url: str = "http://localhost:8000"
    application_service_url: str = "http://localhost:8001"
    document_service_url: str = "http://localhost:8002"
    retrieval_service_url: str = "http://localhost:8003"
    llm_service_url: str = "http://localhost:8004"
    agent_service_url: str = "http://localhost:8005"
    embedding_service_url: str = "http://localhost:8006"
    vector_service_url: str = "http://localhost:8007"
    knowledge_service_url: str = "http://localhost:8008"
    service_call_timeout: float = 600.0

    # ------------------------------------------------------------------- cors
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:4173",
            "http://localhost:3000",
        ]
    )

    # ----------------------------------------------------------------- upload
    max_upload_bytes: int = 25 * 1024 * 1024
    allowed_upload_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".txt", ".docx", ".md"]
    )

    # ---------------------------------------------------------------- prompts
    prompt_version_legal_research: str = "v1"
    prompt_version_similar_cases: str = "v1"
    prompt_version_drafting: str = "v1"
    prompt_version_doc_understanding: str = "v1"

    @field_validator("cors_origins", "allowed_upload_extensions", "api_keys", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("database_url")
    @classmethod
    def _default_database_url(cls, value: str) -> str:
        if value:
            return value
        db_path = RepoRoot / "data" / "legalaid.db"
        return f"sqlite+pysqlite:///{db_path.as_posix()}"

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.upload_dir,
            self.index_dir,
            self.knowledge_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def service_url(self, service: str) -> str:
        mapping = {
            "gateway": self.gateway_url,
            "application-service": self.application_service_url,
            "document-service": self.document_service_url,
            "retrieval-service": self.retrieval_service_url,
            "llm-service": self.llm_service_url,
            "agent-service": self.agent_service_url,
            "embedding-service": self.embedding_service_url,
            "vector-service": self.vector_service_url,
            "knowledge-service": self.knowledge_service_url,
        }
        try:
            return mapping[service]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown service: {service}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
