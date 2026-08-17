# LegalAId — Build Progress

Working notes for continuity across sessions. Updated as phases complete.

## Environment (verified on this machine)
- Python 3.13.12. **Project venv at `.venv`** — global site-packages has corrupted
  distributions (`~jango`, `~ransformers`, `~ttpx`, `~umpy`, `~vicorn`), so always use
  `.\.venv\Scripts\python.exe`.
- `PYTHONPATH=backend\src` required for backend commands.
- Node 22.22.0 / npm 11.6.0.
- Docker CLI 29.2.1 present, **daemon not running** (Phase 5 needs Docker Desktop started).
- **Ollama 0.32.13 IS installed** at `%LOCALAPPDATA%\Programs\Ollama\ollama.exe` (NOT on PATH).
  Both models pulled and verified: `codellama:7b-instruct` (3.8 GB), `nomic-embed-text` (274 MB).
- 25.4 GB RAM, AMD Ryzen 5 7235HS, **NVIDIA RTX 3050 6 GB** → `codellama:7b-instruct`.
- Tesseract OCR installed at `%LOCALAPPDATA%\Programs\Tesseract-OCR`.

### Measured performance (matters for design, not a footnote)
`nomic-embed-text` → **768 dims** confirmed via `/api/embeddings`.
Code Llama 7B generation throughput vs `num_ctx` (GPU has only 6 GB, so context size
decides how much offloads to GPU):

| num_ctx | VRAM footprint | CPU/GPU split | tok/s |
|---------|---------------|---------------|-------|
| 2048    | 5.0 GB        | 17% / 83%     | 15.6  |
| 3072    | ~5.5 GB       | —             | 12.0  |
| 4096    | 6.1 GB        | 31% / 69%     | 9.5   |

Cold start ≈ 89 s (model load); warm load ≈ 0.1 s.

### Embedding throughput — batching is essential (measured)
`nomic-embed-text` runs 100% on GPU, but each HTTP request carries **~2.2 s of fixed
overhead that barely varies with input length** (152 chars → 2221 ms; 3040 chars → 2341 ms).
So per-chunk cost is dominated by request count, not text size. Using the batched
`POST /api/embed` (`input: [...]`) instead of `POST /api/embeddings` (`prompt: ...`):

| batch | ms/chunk | speedup |
|-------|----------|---------|
| 1     | 2249     | 1x      |
| 16    | 161      | 14x     |
| 32    | 91       | 25x     |
| 64    | 54       | 42x     |

Verified `cosine(single, batched) == 1.000000`, i.e. identical vectors — pure throughput win.
Full-corpus embedding goes from ~13 min to ~20 s. `embedding_batch_size` set to 32 (near-optimal
without oversized payloads). **The embedding service must use `/api/embed`, not `/api/embeddings`.**
**Consequences:** `llm_num_ctx` default of 8192 is wrong for this machine — set 3072.
Agent token budgets must be bounded (~500-700 output tokens each), the two independent
agents must run concurrently, and the UI must stream so a ~2-3 min pipeline stays legible.

## Key decisions
- Vector store: **FAISS flat inner-product** on normalised vectors (exact cosine). Corpus is
  <2k vectors so IVF/HNSW is needless complexity.
- Case law: [vanga/indian-supreme-court-judgments](https://github.com/vanga/indian-supreme-court-judgments),
  **CC-BY-4.0**, AWS Open Data, from official scr.sci.gov.in. Redistributable with attribution.
  Rejected IL-PCSR and ILDC/CJPE (CC-BY-**NC**, no redistribution).
- Legislation: India Code official GoI repository. Fetched at setup, not vendored; repo ships a manifest.

### India Code ingestion route — PROVEN on this machine
Reverse-engineered and verified end-to-end. Do not re-derive this.
1. Act page: `https://www.indiacode.nic.in/handle/123456789/{handle}` (GET; **HEAD is refused**).
   Contract Act 1872 = handle `2187`. There is no REST/OAI API (`/rest/*`, `/oai/*` → 404).
2. That page embeds the full section index. Each section is an anchor of the form
   `/show-data?abv=CEN&statehandle=...&actid={actid}&sectionId={sid}&sectionno={n}&orderno={o}`.
   Parsing it yielded **all 268 sections** with numbers + titles.
3. Section **text is AJAX-loaded**, so `/show-data` HTML has an empty body. The real endpoint is
   `GET /SectionPageContent?actid={actid}&sectionID={sid}` → JSON `{"content", "footnote"}`.
   Verified: s.73 returns full text incl. Illustrations; s.74 returns text **plus amendment
   footnotes** ("Subs. by Act 6 of 1899, s. 4..."), which feeds the version/amendment metadata.
4. Full act PDF also downloads (`/bitstream/123456789/2187/2/A187209.pdf`, 568 KB) for the
   page-level provenance viewer.
This gives genuine, section-level, citable statutory text — no synthetic legal data anywhere.
- Transport: `ServiceTransport` with `http` and `inprocess` implementations sharing identical
  Pydantic contracts. UI displays which transport served each hop.
- One app, five exercises via `legalaid.capabilities` gated on `ACTIVE_PHASE`.

## Gift for the RAG ON/OFF demo (Exercise 3) — keep this
Verified via `scripts/smoke_llm.py`. Asked **without** retrieval:
> Q: "what does Section 73 of the Indian Contract Act deal with?"
> Code Llama: *"Section 73 ... deals with the sale of goods."*  ← **wrong**

Ground truth from India Code `/SectionPageContent`: s.73 is *"Compensation for loss or damage
caused by breach of contract."* So the baseline model confidently hallucinates a statutory
provision, and RAG ON fixes it with a citable source. Use this exact question as the headline
RAG OFF vs RAG ON comparison — it is a real, reproducible failure, not a staged one.

## Status
- [x] Phase 0 design docs: architecture, data-sources, database-schema, agents,
      ui-information-architecture, roadmap
- [x] `pyproject.toml` with all 18 pins verified against PyPI (`scripts/check_pins.py`)
- [x] `legalaid.config` — all models/paths/top-k from env, nothing hardcoded
- [x] `legalaid.capabilities` — 21 capabilities across 5 phases
- [x] `legalaid.domain.enums` + `legalaid.domain.contracts` — 64 models, import-verified
- [x] Database models — 26 tables created in `data/legalaid.db` (all empty; seed still to do)
- [x] Observability/trace recorder (`core/tracing.py`), transport, logging, errors
- [x] LLM service (Ollama/Code Llama) — **verified end-to-end**: `scripts/smoke_llm.py`
      reaches `codellama:7b-instruct`, returns tokens/latency, persists a `ModelRun` row
- [x] Chunking (14 tests pass), extraction, FAISS vector store module
- [x] India Code fetcher (`ingestion/india_code.py`) + corpus manifest (`ingestion/corpus.py`),
      17 new tests. Verified against the live site.
- [x] Embedding path verified (`scripts/smoke_embed.py`): 768 dims, batched, and semantically
      ordered — cosine(contract, contract-ish)=0.82 vs cosine(contract, conspiracy)=0.56
- [x] Seed (`legalaid.db.seed`) — idempotent (29 rows created, second run 0). Personal org of
      one + full RBAC vocabulary ready for firm mode: 7 roles, 14 permissions, 68 grants,
      3 knowledge sources with licence/attribution, demo matter `LA-DEMO-001`.
      Deliberately seeds **no password hash** — Phase 1 is single-user and unauthenticated;
      a fake hash would imply an auth system that does not exist.
- [x] **Embedding service** (`services/embedding/`) — `/embeddings/document`, `/embeddings/query`,
      `/embeddings/chunks`, batched, reports model/dims/norm/latency/vector-ids
- [x] **Knowledge service ingestion** (`services/knowledge/ingest.py`) — India Code → provisions
      → chunks → embeddings → FAISS. Idempotent, resumable, version-aware.
- [x] **Retrieval service** (`services/retrieval/`) — query embedding → source filter → vector
      search → metadata filter → rank → context, persisting `RetrievalRun`/`RetrievalResult`
- [x] **Why-retrieved explanations** (`services/retrieval/explain.py`) — 18 legal concept
      groups; reasons are lexical evidence, never model-generated rationalisation
- [ ] Agents + orchestrator, gateway, frontend, Docker

## LIVE KNOWLEDGE BASE (ingested and verified)
**263 live provisions → 355 chunks → 355 vectors** (768-d, `nomic-embed-text`), all in
`LEGAL_ACTS`. `scripts/check_corpus_quality.py` reports **0 problems**: every chunk has a
source URL and a citation, one embedding model, one dimension, vectors == embeddings.

| act | provisions |
|---|---|
| Indian Contract Act, 1872 | 192 |
| Bharatiya Sakshya Adhiniyam, 2023 | 20 |
| Information Technology Act, 2000 | 19 |
| Code of Civil Procedure, 1908 | 18 |
| Bharatiya Nyaya Sanhita, 2023 | 14 |

### Retrieval quality (measured, `scripts/smoke_retrieval.py`)
- "compensation for loss caused by breach" → **s.73 rank 1 @ 0.8182**, then s.75, s.74
- "is a penalty clause enforceable" → **s.74 rank 1 @ 0.7300**
- "agreement void for want of consideration" → **s.25 rank 1 @ 0.7617**
- "are electronic records legally recognised" → **IT Act s.4 @ 0.7680**, s.10A, s.3, BSA s.61
  (correct cross-act retrieval; before the repealed purge this query returned irrelevant
  Contract Act sections at ~0.56)

## Three real defects found by building, and fixed
1. **Chunk fragmentation.** `_split_subsections` split at every marker and only merged pieces
   below `chunk_min_chars` (120), so IT Act s.2 became **28 chunks, a dozen of them 116-300
   chars** — too little context to match a question, and they crowd top-k with near-duplicates.
   Fixed by *packing* subsections up to `chunk_target_chars` while still splitting on
   subsection boundaries. s.2: 28 → 8 chunks; whole act 32 → 12. Pinned by
   `tests/test_chunk_packing.py`.
2. **Repealed provisions indexed as live law.** Asking about "Section 73" ranked
   *"Section 259 — Repealed … Rep. by s. 73"* **third**, purely because the repeal note names
   s.73. Citing dead law is the worst failure mode for this tool. `is_repealed()` now excludes
   them, and because 76 had already been ingested the filter is **retroactive** —
   `_purge_provisions` removes previously stored ones (339 → 263 provisions). Pinned by
   `tests/test_repealed_filter.py`.
3. **FK-ordering bug on delete.** `retrieval_results` and `citations` reference `chunks`, so
   deleting chunks first raised `FOREIGN KEY constraint failed` and the reset silently did
   nothing — after which ingestion reported all zeros because rows already existed. Deletion
   now goes leaves-inward via `_delete_chunks`, and drops the FAISS vector too so search can
   never return a chunk id that no longer resolves.

### Efficiency note
`fetch_act(only_sections=...)` filters **before** fetching. BNS has 358 sections but the
manifest keeps 14; filtering afterwards meant 344 pointless requests to a government server
(~3 min wasted per run).

## THE SINGLE MOST IMPORTANT FIX: constrained decoding
Code Llama 7B **cannot** follow a JSON schema from instructions alone. First end-to-end run:
both research agents failed to parse twice each, and the raw output was the model echoing the
retrieved statute's *Illustrations* back as `{"73": [{"a": "A contracts to sell 50 maunds of
saltpetre to B..."}]}`. It pattern-matched its input instead of obeying the schema. Total run:
**640 s, zero usable output.**

Ollama accepts a **JSON Schema as the `format` parameter** and constrains sampling to it, which
makes off-schema output structurally impossible. After wiring that in (`agents/schemas.py`,
`json_schema` on `LLMGenerateRequest`, passed to `/api/generate`):

| agent | before | after |
|---|---|---|
| Legal Research | 363 s, parse FAILED | **33 s, valid, audit 0 problems** |
| Similar Cases | 172 s, wrong shape | **22 s, valid, audit 0 problems** |

Supporting changes that mattered:
- **Illustrations stripped from prompts** (`trim_for_prompt`). They are kept in the knowledge
  base for citation, but in a prompt they tripled length *and* supplied the lettered-list
  pattern the model copied. Chunks are capped at 700 chars for prompts.
- Context budgets cut (law 8→5, cases 6→4) and `max_tokens` 700→500. Prompt tokens dominate
  cost at ~12 tok/s.
- `additionalProperties: false` on every schema, so the model cannot bolt on invented keys.

## Other real bugs found and fixed this session
- **BM25 citation tokens were being destroyed.** `tokenize` rewrote "Section 73" → `section_73`,
  but `_TOKEN_RE` excluded `_` and split it straight back into `section` + `73`. Since "section"
  appears in 71% of the corpus and "73" in 1.4%, the precise half was swamped. Now one token
  (df=8, 1.1%). Also added a **citation boost (3x)**: plain BM25 rewards short documents, so a
  38-token chunk matching only {indian, contract, act, deal} outranked chunks that actually
  contained s.73.
- **`agent_runs.trace_id` is NOT NULL**, but standalone agent runs passed `None` → crash.
  Rather than weaken the constraint, `_ensure_trace` creates a minimal trace so every agent
  execution is traceable regardless of entry point.
- **e-SCR ligature corruption.** PDFs emit "specifi c performance" and "fi led". Left alone the
  token is "fi", not "specific" — corrupting embeddings and any quoted passage. Repaired via an
  explicit legal-vocabulary map (a generic rule would weld "of the" → "ofthe").
- **S.C.R. gutter letters** (`A B C D E F G H`) and running page headers were polluting chunks.
- **Headnote regex never matched.** It terminated on "CASE LAW", which does not occur in these
  documents; a lazy match on an absent terminator captures nothing. Now enumerates real
  terminators. Headnote extraction is what makes case chunking work at all.
- **Case chunking exploded**: 341 chunks from 6 judgments, one constitutional-bench case alone
  producing 151, which would dominate every search. Now indexes the court-authored **headnote**
  (facts + issues + holding) rather than every page → 33 chunks, 2-8 per case.

### `maxItems` is load-bearing (second truncation bug)
After constrained decoding, one run still failed: `out=500` exactly, i.e. **truncated at
`num_predict`** mid-array. Constrained decoding guarantees the output *shape* but not that a
complete object fits the token budget — the model opened a long `sections` array and ran out.
Fixed by bounding every array (`sections` maxItems 4, `cases` 3, nested lists 3) and raising
research budgets to 620. **Three consecutive full-pipeline runs then passed cleanly** (208 s,
227 s, 180 s; all agents `parse_ok`, all audits 0 problems).

### Drafting authority false positive
The loop rejected `"Section 73 of Indian Contract Act, 1872"` as fabricated even though the
Legal Research Agent had established `section 73 | Indian Contract Act, 1872` — the strings
differed only in format. That wasted a whole 30-60 s iteration *teaching the model to drop a
correct citation*, which is worse than not checking. `_authority_key` now normalises to
`s73:indian`, so equivalent citations compare equal while s.73 vs s.74 still differ.

## FIVE AGENTS, VERIFIED WORKING
| # | agent | sees | withheld | measured |
|---|---|---|---|---|
| 1 | Legal Research | 5 law + 2 doc chunks | **all case law** | ~27-33 s |
| 2 | Similar Cases | 4 case (1 per judgment) + 2 law | **all matter docs** | ~22-40 s |
| 3 | Drafting | verified facts + both agents' findings | **no raw chunks at all** | ~22-50 s |
| 4 | Document Understanding | matter documents only | all statute + case law | (needs uploads) |
| 5 | Verification (critic) | reviewed agent's evidence set | — | deterministic + LLM |

Agents 1 and 2 run **concurrently** (`asyncio.gather`); 3 depends on both. Agent 5 is
deterministic-first (set membership of chunk_ids/sections/titles) because asking a 7B model to
check citation membership is slower and less trustworthy than a set lookup; the LLM layer only
judges overclaiming.

## HYBRID + GRAPH RAG (all three retrievers live)
- **dense** — 768-d cosine over FAISS
- **keyword** — BM25 implemented directly (~40 lines, no new dependency), with legal citation
  normalisation and a 3x exact-citation boost
- **graph** — symbolic traversal, **no vectors**: 322 nodes / 315 edges
  (263 provisions, 54 cases, 5 acts; relations `contains`, `cross_references`,
  `considers_act`, `cites_section`)
Fused with **Reciprocal Rank Fusion** (k=60, dense 1.0 / keyword 0.6) because BM25 and cosine
are not on comparable scales; RRF combines ranks so no score calibration is needed. Each hit
records which retrievers found it, so the UI can show dense-only vs keyword-only vs agreed.

## AGENT 0: INTAKE ROUTER (classify + query rewrite) — working
Runs **before** retrieval. Classifies free-form input (question / document / both), identifies
document type, decides which agents are needed, and **rewrites the input into the text that
actually gets embedded**.

Why the rewrite matters: embedding the raw input embeds the conversational wrapper too, which
dilutes the legal signal in a 768-d vector. Verified working with a pasted FIR:
- original: *"My client supplied goods under a written agreement and the buyer's cheque bounced.
  What provisions apply...?"*
- `document_type: fir`, `input_kind: question_with_document`, agents requested:
  `[document_understanding, legal_research, similar_cases]`
- **search_query**: *"Dishonour of cheque issued towards payment for goods supplied under a
  written supply agreement; cheating and criminal breach of trust by inducing delivery without
  intention to pay; recovery of price and compensation for breach."*
- concepts: `[dishonour of cheque, cheating, criminal breach of trust, recovery of price...]`

Two bugs fixed while getting there:
1. Documents ingested **without** a matter were not passed to the router, so the common
   "paste an FIR then ask" flow classified as a bare question and `document_type: none`. The
   gateway now feeds recent matter-less documents (with a 700-char excerpt) to the router.
2. The router copied the question verbatim into `search_query`. The prompt now forbids that
   explicitly and carries a worked FIR example.
`fallback_decision` keeps the pipeline running if the router itself fails, flagged
`used_fallback` so the UI shows routing was not model-driven.

## API GATEWAY: 27 routes, live at :8000 with OpenAPI /docs
Research (`/api/research/query`, `/stream` SSE, `/compare` for RAG on/off), retrieval
(`/api/retrieval/search` with per-request `enable_keyword` / `enable_graph` toggles),
inspection (`/api/traces/{id}` returns stages + agent runs + model runs + retrievals;
`/api/chunks/{id}` returns text, embedding, vector preview, retrieval history, which agents
used it, and full provenance), knowledge (`/api/knowledge`, `/provisions`, `/graph`),
documents (`/upload`, `/paste`), plus `/api/models`, `/api/agents`, `/api/prompts`,
`/api/metrics`, `/api/system/health`.

**Model switching works**: `/api/models` lists installed Ollama models with
`usable_for_generation` (embedding models excluded), and `model` on a research request flows to
every agent. Both `codellama:7b-instruct` and `qwen2.5:3b-instruct` are installed.

## FRONTEND: built and live at :5173 (12 pages, typecheck clean)
React 18 + TS + Vite, dark/light, professional legal-tech density. Vite proxies `/api` to the
gateway so there is one origin in dev.

| page | shows |
|---|---|
| Research | one input for question / FIR / notice; live SSE stage progress; router decision incl. **original vs rewritten query**; retrieval with per-chunk "why retrieved"; per-agent context **included vs deliberately withheld**; system + user prompt; raw model text; loop iterations |
| Pipeline Trace | trace list, **node graph with parallel agents drawn side by side**, expandable stages with input/output, retrieval detail, "what did the model see?" per model call |
| Retrieval Inspector | query embedding + vector preview, per-collection candidates/top-k/timings, **toggles to switch keyword and graph off** and see each retriever's contribution |
| Chunk Explorer | full text, all metadata, embedding + raw vector, retrieval history, which agents used it, provenance to official source URL + amendment footnotes |
| Embeddings | model, dimensions, norms, latency, vector bar previews |
| Agents | 6-agent roster with orchestration node graph, prompt versions, temp/max-tokens, context policy, recent runs |
| Knowledge Base | acts → provisions (with official source links), 47 judgments with citations, licence/attribution, **graph structure with edge evidence** |
| Documents | upload or paste, extracted text per page, chunks derived per page |
| RAG On/Off | side-by-side same-model comparison with sources and token accounting |
| Dashboard | corpus counts, all three retrievers, measured latencies, per-agent stats, exercise coverage |
| System | service health, installed models, retrieval infra, prompt registry, capability matrix |

**Model switching** is in the top bar (persisted to localStorage) and flows to every agent.
All 17 UI-facing endpoints verified returning real data via `scripts/smoke_ui_endpoints.py`.

## How to run
```powershell
# 1. Ollama (must be running)
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve   # usually already running

# 2. API gateway
$env:PYTHONPATH="backend\src"
.\.venv\Scripts\python.exe -m uvicorn legalaid.services.gateway.app:app --port 8000 --reload --reload-dir backend\src

# 3. Frontend
cd frontend; npm run dev      # http://localhost:5173
```

## SYNTHESIS AGENT (agent 6) — the pipeline now ANSWERS
Previously the pipeline ended with structured findings: a list of sections and a list of cases.
That is research output, not an answer. The synthesis agent writes a colleague's reply — direct
answer, reasoning, what is uncertain, next steps — and like drafting it sees **no raw chunks**,
only verified upstream findings, so it cannot introduce an unretrieved authority. Its own
reasoning is reported separately under `inference` so the UI can keep source and interpretation
visually distinct. `fallback_answer()` guarantees the lawyer never sees an empty result.

## SCHEMA / TOKEN BUDGET: the root cause of repeated truncation
Three separate agents kept truncating at exactly `num_predict`. I fixed it three times by hand
(raising caps, adding `maxItems`) and it kept coming back, because I was treating symptoms.

**The actual cause:** constrained decoding fills whatever the schema permits, and `max_tokens`
was being chosen by hand *independently of the schema*. Any budget below the schema's realistic
output guarantees truncation. Worse, the real culprit was uncapped **strings**, not arrays — the
model transcribed whole statutory passages verbatim into `what_it_provides`, spending 620 tokens
on two sections.

Fixed structurally: `required_max_tokens()` derives the budget from the schema
(`worst_case_chars × 0.55 realistic × 1.25 safety`, capped at 1500), and `Agent._token_budget`
raises the prompt's preference to meet it. Tighten a schema and the budget follows automatically.
`test_schemas.py` (17 tests) now enforces: every array bounded, every summary string capped,
`additionalProperties: false`, budget ≥ realistic output, and budget below the cap.

### Ollama grammar limit — a silent agent killer
With large `maxLength` values Ollama returns **HTTP 400 "failed to parse grammar"**. Measured on
this build: `maxLength: 1600` compiles, **`2000` fails**. It is a *request-time* failure, so an
over-long cap does not degrade output — it **disables the agent entirely**. Both synthesis and
drafting were completely dead (1.0 s, `out=None`) until this was traced.
`MAX_SAFE_STRING_LENGTH = 1400` with a test enforcing it.

### Result
All 5 agents parse; **247 s** end to end (was 632 s with two agents failing). 147 tests pass.

## UI: user / developer split
- **"For you"** — Home (lawyer dashboard), Ask LegalAId, My documents, Law library
- **"Developer / evaluation"** — Pipeline trace, Agents, Retrieval inspector, Chunk explorer,
  Embeddings, RAG on/off, System & models

The dashboard was rebuilt for a practising advocate: a greeting, one large ask box, four
one-click matter starters, recent research, and what the library covers — no vector counts or
token throughput, which moved to the developer pages. `/answer/:traceId` reopens any past
research. Research now leads with the answer; the machinery is behind "Show how this answer was
produced".

## AGENT 6: SYNTHESIS — the pipeline now ANSWERS
Previously the pipeline ended with structured findings (a list of sections, a list of cases). That
is research output, not an answer. `agents/synthesis.py` + `prompts/synthesis/v1.yaml` writes a
colleague's reply: prose answer, key points, **inference labelled separately from what sources
say**, uncertainties, next steps, confidence.

Like drafting it sees **no raw chunks** — only verified findings — so it cannot introduce an
authority that was never retrieved. `fallback_answer()` assembles a plain summary from the
structured findings if the synthesis call fails, flagged `used_fallback` so the UI never presents
it as model-written analysis.

Verified: 57 s, valid output, real key points and next steps.

### Third truncation bug, now impossible to reintroduce
`document_understanding` failed with `out=600` exactly — truncated at `max_tokens` because its
schema had **unbounded arrays**, the same failure class as `sections` earlier.
`tests/test_schemas.py` now asserts **every array in every schema has `maxItems`** and every object
sets `additionalProperties: false`. It immediately caught a fourth case I had missed
(`verification`). 118 tests pass.

## UI RESTRUCTURED for two audiences
- **For you** (lawyer): Home, Ask LegalAId, My documents, Law library
- **Developer / evaluation**: Pipeline trace, Agents, Retrieval inspector, Chunk explorer,
  Embeddings, RAG on/off, System & models

New **Home** is a practitioner's page — a single ask box, four starting points, recent research,
what the library covers. No vector counts or token throughput; those moved to the developer group.
New **`/answer/:traceId`** page re-opens any past research from recorded data.
`AnswerView` leads with prose and labels AI inference distinctly from law.

## DOCKER — built and running
5 services (`gateway`, `llm`, `embedding`, `retrieval`, `frontend`) + optional `ollama` behind a
`bundled` profile. Images: backend 1.25 GB, frontend 74 MB.

**Key decision: Ollama defaults to the HOST, not a container.** Docker Desktop on Windows has no
GPU access without the NVIDIA Container Toolkit, so a containerised Ollama runs on CPU at ~4-6
tok/s versus ~12 on the host GPU — a full run goes from ~4 min to ~12. `--profile bundled` starts
a containerised Ollama for a fully self-contained demo.

Two compose bugs fixed along the way:
1. A busybox stub for a "disabled" ollama service exits immediately, so `condition:
   service_healthy` could never be satisfied and every dependent container refused to start.
   Replaced with **profiles**, which is the correct mechanism.
2. Hard `depends_on: ollama` was wrong once Ollama could live on the host. Removed; reachability is
   reported through `/api/system/health` instead, so a missing model surfaces as degraded health
   rather than a container that won't boot.

Verified in Docker: gateway healthy, host Ollama reachable (4 models visible from inside the
container), nginx serving the UI on :3000 and proxying `/api` including SSE, `llm` and `embedding`
healthy. `retrieval` correctly self-reported `degraded` with *"Knowledge base is empty"* because
the Docker volume starts fresh — that is the health check working, and seed + ingest inside the
container fixes it.

## DOCUMENTATION written
`README.md` (both run paths, verification, limitations), `docs/developer-guide.md` (18 sections,
every component and the reasoning behind it), `docs/architecture.md`, `docs/rag.md`,
`docs/api.md`, `docs/docker.md`, `docs/demo.md`.

## Next action when resuming
Agents (`legal_research`, `similar_cases`, `drafting`) + orchestrator with per-agent context
isolation, then the gateway, then the frontend, then Docker. The RAG substrate they need is
live and verified — call `services.retrieval.service.search` with a `RetrievalMode`.
Case law (`CASE_LAW` collection) is still empty: the CC-BY-4.0 Supreme Court dataset has not
been downloaded yet, so the Similar Case Agent has nothing to retrieve until that is done.

### Verified corpus (handles + section counts read back from the live site)
| key | handle | act | sections | in scope |
|---|---|---|---|---|
| `contract_act_1872` | 2187 | Indian Contract Act, 1872 | 268 | 268 (all) |
| `bns_2023` | 20062 | Bharatiya Nyaya Sanhita, 2023 | 358 | 14 (filtered) |
| `bsa_2023` | 20063 | Bharatiya Sakshya Adhiniyam, 2023 | 170 | 20 (filtered) |
| `it_act_2000` | 1999 | Information Technology Act, 2000 | 125 | 19 (filtered) |
| `cpc_1908` | 2191 | Code of Civil Procedure, 1908 | 171 | 18 (filtered) |

Total ≈ 339 provisions across 5 acts → satisfies the brief's 5-10 acts / 100-300 provisions.
Handles are NOT ordered by subject; `scripts/scan_handles.py` probes ranges and caches results.
BNSS 2023 was not found in the 20060-20110 range — 5 acts is already sufficient, so scanning
was stopped rather than spending more time on it.

### Two India Code gotchas that cost real time — do not rediscover
1. **The WAF 403s any `User-Agent` containing parentheses.** The conventional
   `Product/1.0 (comment)` form fails; `LegalAId/0.1 academic-legal-research` works. Omitting
   the UA entirely gives a **404** instead. This looked exactly like IP rate-limiting at first.
2. Bursts of rapid requests do also get throttled with 403s, so `_get` backs off 3x harder on
   403/429 and ingestion paces itself (~3.5-4 s between act-page probes).

### Statutory text cleaning (`strip_html`) — why it is not naive
Raw `SectionPageContent` HTML has two defects that would silently corrupt embeddings and quoted
citations:
- `\r\n` appear **mid-sentence** ("has been\r\nbroken") — they are PDF line-wrap artifacts, not
  paragraph breaks. Real paragraph breaks are `</br><hr class="hr1"/>`. Wraps are rejoined.
- Inline tags sit flush against text (`<i>Explanation</i>.In estimating`), so naive tag stripping
  welds words together. Tags become spaces and punctuation spacing is repaired.
Pinned by tests in `backend/tests/test_india_code.py` using real markup fixtures.
- [ ] Agents + orchestrator
- [ ] Frontend
- [ ] Docker
- [ ] Tests + docs

## Verification commands
```powershell
$env:PYTHONPATH="backend\src"
.\.venv\Scripts\python.exe -m pytest backend\tests -q
.\.venv\Scripts\python.exe scripts\check_pins.py
```
