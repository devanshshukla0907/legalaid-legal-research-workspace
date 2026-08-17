# LegalAId — Implementation Roadmap

One application. Five exercises. Later phases activate capabilities; they never fork the codebase.

## Phase 0 — Foundation (design + skeleton)
- [x] Architecture proposal, data-source plan, DB schema, service map, agent design, UI IA, roadmap
- [ ] Repo layout, typed config, capability registry, logging, error model
- [ ] Database models + migrations + seed (personal org/user/workspace)

## Phase 1 — Exercise 1: Application + API + Ollama + Code Llama
- [ ] `llm-service` with Ollama provider, model trace recording, health probe
- [ ] `gateway` + OpenAPI + trace-id middleware
- [ ] `application-service`: matters CRUD, document upload
- [ ] `document-service`: validation, PDF/TXT/DOCX extraction, page provenance
- [ ] Frontend shell, navigation, theme, Dashboard, Matters, Documents, Models, System
- [ ] End-to-end: question → gateway → app → llm → Ollama → Code Llama → answer, fully traced

## Phase 2 — Exercise 2: Knowledge base + chunking + embeddings
- [ ] Legislation fetcher + manifest (India Code)
- [ ] Case-law fetcher (CC-BY-4.0 Supreme Court open dataset) + attribution
- [ ] Structure-aware chunkers: statute, judgment, uploaded document
- [ ] `embedding-service` (`/embeddings/document`, `/embeddings/query`) with latency + vector ids
- [ ] `vector-service` FAISS: upsert / delete / search / metadata filter / persist
- [ ] `knowledge-service` ingestion orchestration
- [ ] UI: Knowledge Base, Chunk Explorer, Embeddings, Document Viewer

## Phase 3 — Exercise 3: Retrieval + RAG
- [ ] `retrieval-service`: query embedding → filter → search → rank → context construction
- [ ] Retrieval modes: LAW / CASE / USER_DOCUMENT / COMBINED
- [ ] Why-retrieved explanations
- [ ] RAG ON vs RAG OFF comparison endpoint + UI
- [ ] UI: Research page, Retrieval Inspector

## Phase 4 — Exercise 4: Services + multi-agent orchestration
- [ ] `agent-service`: Legal Research, Similar Case, Drafting agents + typed tools
- [ ] Prompt registry with versioned YAML
- [ ] Orchestrator: plan, parallel fan-out, dependent drafting stage, assembly
- [ ] Agent context isolation with persisted included/excluded manifests
- [ ] Full pipeline trace + SSE live progress
- [ ] Citation validator
- [ ] UI: Pipeline (node graph + stages), Agents, Model Context panel, Similar Cases, Drafting

## Phase 5 — Exercise 5: Docker + deployment
- [ ] Dockerfiles per service, `docker-compose.yml`, Ollama service + model pull documentation
- [ ] Health checks, dependency ordering, volumes for indexes/db/uploads
- [ ] Verify inter-service HTTP communication inside the compose network

## Phase 6 — Testing + documentation
- [ ] Unit tests: chunkers, parsers, citation validator, capability gating, retrieval ranking
- [ ] Service tests: each service independently
- [ ] Integration: the ten required scenarios (direct statute match, multi-provision, similar-case,
      uploaded-document, insufficient evidence, hallucination-prone, irrelevant, RAG on/off,
      poorly-formatted document, multi-page PDF)
- [ ] Docs: ingestion, rag, api, docker, demo; README; known limitations

## Build order (as specified)
architecture → UI shell → Ollama/Code Llama → document upload → knowledge ingestion → chunking →
embeddings → vector search → legal research agent → similar case agent → drafting agent →
orchestrator → pipeline tracing → retrieval inspector → model/context inspector → Docker →
testing → documentation

## Environment notes discovered on this machine
- Python 3.13.12; `faiss-cpu` 1.13.2, `numpy` 2.4.2, `pymupdf` 1.26.5 already present.
- Node 22.22.0, npm 11.6.0.
- Docker CLI 29.2.1 present but **daemon not currently running** → Phase 5 requires Docker Desktop started.
- **Ollama is not installed** → required before Phase 1 can run end-to-end. Setup is documented in README.
- 25.4 GB RAM → `codellama:7b-instruct` selected over 13b.
- Tesseract OCR installed → optional OCR fallback for scanned PDFs is feasible.
