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
| 11 | Ollama | `services/llm` | `LLM_CALL` |
| 12 | Generation model | per-request; `llama3.2:3b` measured best | recorded on every `model_run` |
| 13 | Token generation | token counters per call | `TOKEN_GENERATION` |
| 14 | Grounding audit | `agents/verification.py` `audit()` | `CITATION_VALIDATION` |
| 15 | Final response | `agents/synthesis.py` | `FINAL_RESPONSE` |

Two stages precede all of this and are not in the original brief:

| Stage | Implemented by | Trace stage |
|---|---|---|
| Input guardrails | `core/guardrails.py` | `INPUT_GUARDRAIL` |
| Output guardrails | `core/guardrails.py` `screen_output()` | `OUTPUT_GUARDRAIL` |

Intent classification and query rewrite were added because they measurably improve retrieval —
rewriting lifted mean recall from 0.500 to 1.000 on the questions it was tested against. See
[rag.md](rag.md#1-the-query-rewrite).

Guardrails were added because the application had none, and because a legal tool that cannot
decline is dangerous rather than merely incomplete. They cost about **1.5% of request wall-clock**
(measured, `eval/latency_breakdown.py`). See [guardrails.md](guardrails.md).

### Where the time actually goes

Measured over 40 recorded pipeline runs, mean total 70.9 s:

| Stage | Share of total | Mean |
|---|---|---|
| Similar Case Agent | 39.7% | 16.4 s |
| Document Understanding | 17.6% | 12.5 s |
| Legal Research Agent | 17.4% | 12.4 s |
| Synthesis Agent | 17.1% | 12.1 s |
| Intake classification | 6.8% | 4.8 s |
| Verification Agent | 6.6% | 4.1 s |
| Context construction | 3.2% | 2.3 s |
| Input guardrails | 1.5% | 0.9 s |
| Citation validation | ~0% | <1 ms |

**86.2% of wall-clock time is inside model generation.** Orchestration overhead is therefore not
the thing to optimise; a faster model or fewer agent calls is. The deterministic citation audit
costs effectively nothing, which is the argument for doing grounding checks in code rather than
asking a model to do them.

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

### Week 3 exercises, and where to see each one working

| Exercise | Requirement | Where it lives | How to verify |
|---|---|---|---|
| 1 | Application + API + Ollama + a code model | `frontend/` → `gateway` → `services/llm` → Ollama | **Ask LegalAId** page; model selectable per request on **System & models** |
| 2 | Knowledge base + chunking + embeddings | `ingestion/india_code_api.py`, `ingestion/chunking.py`, `services/embedding` | **Law library** (1,783 provisions, 47 judgments); **Chunk explorer** shows strategy per chunk; **Embeddings** page shows vectors |
| 3 | Vector similarity + retrieval + RAG | `services/vector` (FAISS), `services/retrieval` | **Retrieval inspector** — toggle each retriever and watch results change; **RAG on / off** page |
| 4 | APIs + services + orchestration | 9 services, `agents/orchestrator.py` | **Pipeline trace** — every hop, its service, its duration; **Agents** page shows context isolation |
| 5 | Docker + full architecture | `docker-compose.yml`, per-service Dockerfiles | `docker compose up`; **System & models** shows each service's health |

The one deviation from the brief worth stating: the default generation model is no longer
`codellama:7b-instruct`. It remains installed and selectable, but `llama3.2:3b` measured better on
accuracy *and* latency (`docs/evaluation.md`), and Code Llama is the only one of the four that does
not fit entirely in 6 GB of VRAM — it runs at a 22%/78% CPU/GPU split. Choosing the model by
measurement rather than by the brief's example seemed the more defensible reading of the exercise.

---

## Safety posture

Full detail in [guardrails.md](guardrails.md), including an explicit list of what is **not**
protected.

- Six input guardrail layers, deterministic before the LLM, so an out-of-scope request never costs
  a generation. Measured: 19/19 adversarial caught, 0/10 false positives on legitimate legal work.
- No agent may cite a source absent from its retrieved context. A **deterministic audit** checks
  every claimed `chunk_id`, section number and case title against what was actually supplied, and
  ungrounded citations are stripped rather than presented as authority. It also scans the prose of
  `answer_markdown` and `draft_markdown`, because a fabricated section inside a paragraph is
  exactly as damaging as one in a citation list.
- Authentication on every route except health, enforced in middleware so a route added later is
  protected by default.
- One generation on the GPU at a time, plus a per-key request budget.
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
