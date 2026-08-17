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
    retrieval_keyword_weight: float = 0.6
    retrieval_graph_hops: int = 2
    retrieval_graph_limit: int = 20

    # ------------------------------------------------------------ orchestration
    # Refinement loop: an agent whose output fails the grounding audit is re-run with
    # corrective guidance. 2 is a deliberate ceiling - at ~12 tok/s every extra iteration
    # costs 30-60 s, and in practice a second attempt fixes citation errors or nothing will.
    orchestrator_max_loop_iterations: int = 2
    orchestrator_parallel_agents: bool = True

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

    @field_validator("cors_origins", "allowed_upload_extensions", mode="before")
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
