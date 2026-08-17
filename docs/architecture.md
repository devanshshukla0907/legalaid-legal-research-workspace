# Architecture

## Design goals, in priority order

1. **Demonstrability.** Every internal step is inspectable in the UI. If the UI cannot answer
   "what did the model see, and where did this come from?", the feature is incomplete.
2. **Provenance.** Every legal statement carries a chain down to a chunk, a page, and an official
   source URL.
3. **Genuine multi-agent behaviour.** Seven agents with different responsibilities, prompts,
   retrieval scopes and **different context windows** — not one mega-prompt.
4. **Correct boundaries.** Services own one capability each and communicate through documented,
   typed contracts.
5. **Honesty.** No fake services, no invented legal sources, no hardcoded answers. Insufficient
   evidence is a first-class result.

---

## The required pipeline, mapped to real components

Each arrow is a real, traced hop recorded in the database.

| # | Stage | Implemented by | Trace stage |
|---|---|---|---|
| 1 | Application | `frontend` (React) | `USER_INPUT` |
| 2 | API | `gateway` (FastAPI) | `API_GATEWAY` |
| — | *Intent classification* | `agents/intake_router.py` | `INTAKE_CLASSIFICATION` |
| — | *Query rewrite* | `agents/intake_router.py` | `QUERY_REWRITE` |
| 3 | Document ingestion | `gateway/uploads.py` + `ingestion/extraction.py` | `DOCUMENT_INGESTION` |
| 4 | Chunking | `ingestion/chunking.py` | `CHUNKING` |
| 5 | Embeddings | `services/embedding` → Ollama | `EMBEDDING` |
| 6 | Vector database | `services/vector` (FAISS) | `VECTOR_UPSERT` |
| 7 | Query embedding | `services/embedding` | `QUERY_EMBEDDING` |
| 8 | Vector similarity | `services/vector` + `retrieval/keyword.py` + `retrieval/graph.py` | `LAW_VECTOR_SEARCH`, `CASE_VECTOR_SEARCH`, `USER_DOC_SEARCH` |
| 9 | Relevant context | `services/retrieval` (filter → fuse → rank → assemble) | `CONTEXT_CONSTRUCTION` |
| 10 | Multi-agent orchestration | `agents/orchestrator.py` | `ORCHESTRATION_PLAN` |
| 11 | Agent-specific context | `agents/base.py` `AgentContext` | `AGENT_CONTEXT` |
| 12 | Ollama | `services/llm` | `LLM_CALL` |
| 13 | Code Llama | `codellama:7b-instruct` | recorded on every `model_run` |
| 14 | Token generation | token counters per call | `TOKEN_GENERATION` |
| 15 | Final response | `agents/synthesis.py` | `FINAL_RESPONSE` |

The two stages not in the original brief — intent classification and query rewrite — were added
because they measurably improve retrieval. See [rag.md](rag.md#1-the-query-rewrite).

---

## Service map

```
                     ┌──────────────────────────────┐
                     │  frontend  (React + nginx)   │  :3000
                     └───────────────┬──────────────┘
                                     │ HTTP  (nginx proxies /api, SSE unbuffered)
                     ┌───────────────▼──────────────┐
                     │  gateway  :8000              │  OpenAPI, trace ids, orchestrator
                     └──┬──────────┬─────────┬──────┘
          ┌─────────────┘          │         └──────────────┐
          ▼                        ▼                        ▼
 ┌──────────────────┐   ┌────────────────────┐   ┌────────────────────┐
 │ retrieval :8003  │   │ embedding :8006    │   │ llm :8004          │
 │ fuse 3 retrievers│   │ batched vectors    │   │ the ONLY component │
 └────────┬─────────┘   └─────────┬──────────┘   │ that calls Ollama  │
          │                       │              └─────────┬──────────┘
          ▼                       ▼                        │
 ┌──────────────────┐   ┌────────────────────┐             ▼
 │ FAISS (in-proc)  │   │                    │      ┌──────────────┐
 │ BM25 (in-proc)   │   │                    │      │   Ollama     │ :11434
 │ Graph (in-proc)  │   └────────────────────┘      │ codellama:7b │
 └──────────────────┘                               │ nomic-embed  │
                                                    └──────────────┘
 ┌────────────────────────────────────────────────────────────────────┐
 │ SQLite volume: 26 tables — matters, documents, provisions, cases,   │
 │ chunks, embeddings metadata, traces, agent runs, model runs         │
 └────────────────────────────────────────────────────────────────────┘
```

### Why these boundaries

- **`llm` isolates the only component that talks to Ollama for generation.** Swapping models
  touches one file, and every call is recorded in one place.
- **`embedding` is separate from `llm`** because it uses a different model, has a completely
  different batching profile (§ batching below), and is called by both ingestion (write path) and
  retrieval (read path).
- **`retrieval` owns strategy; the vector store owns similarity math.** That split is what makes
  each retrieval mode independently testable.
- **`gateway` hosts the orchestrator** so agent context isolation is enforced by one auditable code
  path rather than scattered across services.

Four containers, not nine. Boundaries are logical, not decorative — a service exists where there is
a real difference in model, failure domain, or scaling profile.

---

## Transport abstraction

Services communicate through `core/transport.py`, which has two implementations selected by
`SERVICE_TRANSPORT`:

- **`inprocess`** — the gateway dispatches directly to the same handler functions. Default, because
  it avoids a network hop on a pipeline already bounded by generation speed.
- **`http`** — real network calls between containers.

Both use **identical Pydantic contracts and the same handler implementations**, so behaviour cannot
drift and there is no mock service anywhere. The Pipeline Trace page displays which transport served
each hop, so the distinction is never hidden.

---

## Observability model

```
pipeline_trace (trace_id)
  ├── trace_stage[]      ordered, timed, service, transport, input, output, status
  ├── retrieval_run[]    → retrieval_result[] → chunk
  ├── agent_run[]        → context manifest: included AND EXCLUDED chunk ids
  ├── model_run[]        → prompt version, params, tokens, latency, raw response
  └── answer / summary
```

The **excluded** chunk ids are persisted deliberately: agent context isolation is only provable if
you can show what each agent was *not* given.

Live progress is pushed over SSE (`/api/research/stream`), which matters because a full run takes
minutes.

---

## Model strategy

| Role | Provider | Model | Why |
|---|---|---|---|
| Generation | Ollama | `codellama:7b-instruct` | Required by the brief. 7B fits alongside Docker and FAISS; the instruct variant follows the structured-output contracts the agents need. |
| Embeddings | Ollama | `nomic-embed-text` (768-d) | Long context suits legal sections; a separate model demonstrates a real embedding stage. |

Every model identifier comes from configuration. Code Llama is the sole generation path — there is
no silent fallback to a hosted API. Models are switchable per request from the UI.

**The trade-off, stated rather than hidden:** Code Llama is code-specialised, so its prose is weaker
than a general chat model's, and it **cannot follow a JSON schema from instructions alone**. The
mitigation is schema-constrained decoding at the Ollama layer, low temperature for extraction, and
strict grounding prompts. See
[developer-guide.md § 11](developer-guide.md#11-constrained-decoding-the-single-most-important-fix).

---

## Hardware-driven decisions

Measured on the development machine (Ryzen 5 7235HS, RTX 3050 6 GB, 25 GB RAM). These numbers
drove real design choices rather than being footnotes.

**Generation throughput vs context window** — the GPU has only 6 GB, so context size decides how
much offloads:

| `num_ctx` | VRAM | CPU/GPU split | tok/s |
|---|---|---|---|
| 2048 | 5.0 GB | 17% / 83% | 15.6 |
| 3072 | ~5.5 GB | — | 12.0 |
| 4096 | 6.1 GB | 31% / 69% | 9.5 |

`llm_num_ctx = 3072`. Consequences: bounded token budgets per agent, prompt context trimmed to 700
chars per chunk, independent agents run concurrently, loop iterations capped at 2, and SSE streaming
so a multi-minute run stays legible.

**Embedding batching** — each request carries ~2.2 s of fixed overhead almost independent of input
length, so cost is driven by request count:

| Batch | ms/chunk |
|---|---|
| 1 | 2249 |
| 32 | 91 |
| 64 | 54 |

`embedding_batch_size = 32`, using the batched `/api/embed`. Verified
`cosine(single, batched) == 1.000000`, so it is pure throughput. Full-corpus embedding: ~13 min →
~20 s.

---

## Progressive activation — one app, five exercises

`capabilities.py` defines 21 capabilities gated by `ACTIVE_PHASE`. The System page renders this as a
live matrix, so an evaluator can verify Exercise 1–5 coverage without reading source.

| Phase | Activates |
|---|---|
| 1 | `app_ui`, `api_gateway`, `matters`, `documents_upload`, `ollama_chat` |
| 2 | `knowledge_ingestion`, `chunking`, `embeddings`, `vector_index` |
| 3 | `query_embedding`, `vector_similarity`, `rag_retrieval`, `rag_compare` |
| 4 | `service_split`, `multi_agent`, `orchestration`, `agent_context_isolation`, `pipeline_trace` |
| 5 | `docker_compose`, `service_health`, `full_deployment` |

Later phases only ever **add**. There are no `exercise1/`, `exercise2/` directories — one
application, progressively activated.

---

## Safety posture

- No agent may cite a source absent from its retrieved context. A **deterministic audit** checks
  every claimed `chunk_id`, section number and case title against what was actually supplied, and
  ungrounded citations are stripped rather than presented as authority.
- **Insufficient evidence is a first-class outcome** (`evidence_sufficient: false`), not a reason to
  guess.
- Repealed and omitted provisions are excluded at ingestion, so dead law can never be cited as
  applicable.
- The synthesis agent's own reasoning is reported under `inference`, visually separated in the UI
  from what sources say.
- Every surface carries the professional-review disclaimer.

---

## Forward compatibility: firm / RBAC mode

Phase 1 ships a single-user personal workspace, but the schema already carries `organizations`,
`roles`, `permissions`, `memberships`, `matter_members`, `document_permissions` and `audit_logs`.
Everything needing tenancy has an `organization_id`, and a personal workspace is simply an
organization of one.

Adding firm mode is therefore **authorisation middleware plus UI, not a migration of core tables**.
The seed already creates 7 roles and 14 permissions with 68 grants.
