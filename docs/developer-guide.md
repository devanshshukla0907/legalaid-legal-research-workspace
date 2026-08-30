# LegalAId — Developer Guide

Everything in the system, what it does, and **why it is built that way**. Where a decision was
driven by something measured on real hardware or discovered from a real failure, the number or
the failure is recorded — those are the parts worth understanding.

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Configuration](#2-configuration)
3. [Request lifecycle](#3-request-lifecycle-end-to-end)
4. [Data acquisition](#4-data-acquisition)
5. [Chunking](#5-chunking)
6. [Embeddings](#6-embeddings)
7. [Vector store](#7-vector-store)
8. [Hybrid retrieval](#8-hybrid-retrieval)
9. [Graph RAG](#9-graph-rag)
10. [Agents](#10-the-agent-layer)
11. [Constrained decoding](#11-constrained-decoding-the-single-most-important-fix)
12. [Orchestration and the refinement loop](#12-orchestration-and-the-refinement-loop)
13. [Observability](#13-observability)
14. [Database](#14-database)
15. [Frontend](#15-frontend)
16. [Testing](#16-testing)
17. [Performance](#17-performance-characteristics)
18. [Extending it](#18-extending-the-system)

---

## 1. Repository layout

```
LegalAId/
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── src/legalaid/
│   │   ├── config.py              Typed settings; every tunable, nothing hardcoded
│   │   ├── capabilities.py        21 capabilities gated by ACTIVE_PHASE (the 5 exercises)
│   │   ├── core/
│   │   │   ├── errors.py          Error taxonomy → HTTP status codes
│   │   │   ├── logging.py         structlog, trace-id bound to context
│   │   │   ├── tokens.py          Token estimation
│   │   │   ├── tracing.py         TraceRecorder + SSE event bus
│   │   │   └── transport.py       ServiceTransport: http | inprocess
│   │   ├── db/
│   │   │   ├── base.py            Engine, session scope, UTC datetimes
│   │   │   ├── models.py          26 tables
│   │   │   └── seed.py            Idempotent seed + RBAC vocabulary
│   │   ├── domain/
│   │   │   ├── enums.py           Collections, stages, agents, statuses
│   │   │   └── contracts.py       64 Pydantic models shared by every service
│   │   ├── ingestion/
│   │   │   ├── india_code.py      Official legislation fetcher
│   │   │   ├── corpus.py          Which acts, verified handles
│   │   │   ├── case_law.py        Supreme Court judgment fetcher
│   │   │   ├── case_curation.py   Two-stage relevance scoring
│   │   │   ├── chunking.py        Three structure-aware strategies
│   │   │   └── extraction.py      PDF/DOCX/TXT with page provenance
│   │   ├── agents/
│   │   │   ├── base.py            Agent ABC, context isolation, JSON recovery
│   │   │   ├── prompts.py         Versioned YAML prompt registry
│   │   │   ├── schemas.py         JSON Schemas for constrained decoding
│   │   │   ├── intake_router.py   Agent 0 — classify + rewrite query
│   │   │   ├── document_understanding.py   Agent 1
│   │   │   ├── legal_research.py           Agent 2
│   │   │   ├── similar_cases.py            Agent 3
│   │   │   ├── drafting.py                 Agent 4
│   │   │   ├── verification.py             Agent 5 — critic
│   │   │   ├── synthesis.py                Agent 6 — writes the answer
│   │   │   └── orchestrator.py    Plans, parallelises, runs the refinement loop
│   │   └── services/
│   │       ├── common.py          Shared FastAPI factory
│   │       ├── gateway/           app.py · inspect.py · uploads.py
│   │       ├── llm/               Ollama client + model-run recording
│   │       ├── embedding/         Batched embedding service
│   │       ├── retrieval/         service.py · keyword.py · fusion.py · graph.py · explain.py
│   │       ├── knowledge/         ingest.py · ingest_cases.py
│   │       └── vector/            FAISS store
│   └── tests/                     118 tests
├── frontend/                      React 18 + TS + Vite, 12 pages
├── prompts/<name>/<version>.yaml  7 versioned prompts
├── scripts/                       Ingestion, smoke tests, diagnostics
└── docker-compose.yml
```

---

## 2. Configuration

`backend/src/legalaid/config.py`. One `Settings` class (pydantic-settings), read from environment
or `.env`. Nothing model-specific or machine-specific is hardcoded anywhere else — that is what
makes `LLM_MODEL=qwen2.5:3b-instruct` work without touching code.

The settings that carry real reasoning:

```python
llm_num_ctx: int = 3072
```
Measured on the dev machine (RTX 3050, 6 GB): at `num_ctx` 2048 the model runs at 15.6 tok/s with
83% GPU offload; at 4096 the footprint grows to 6.1 GB, offload drops to 69%, and throughput falls
to 9.5 tok/s. 3072 keeps enough room for retrieved context without falling off the GPU.

```python
embedding_batch_size: int = 32
```
Ollama embedding calls carry ~2.2 s of **fixed per-request overhead that barely varies with input
length** (152 chars → 2221 ms; 3040 chars → 2341 ms). Cost is therefore driven by request count,
not text size. Batching via `/api/embed`: batch 1 → 2249 ms/chunk, batch 16 → 161, batch 32 → 91,
batch 64 → 54. Verified `cosine(single, batched) == 1.000000`, so it is a pure throughput win —
full-corpus embedding drops from ~13 minutes to ~20 seconds.

```python
retrieval_dense_weight: float = 1.0
retrieval_keyword_weight: float = 0.15
```
Semantic similarity is the primary signal for natural-language questions; BM25 is a precision
correction for exact identifiers, weighted well below 1.0 so it sharpens rather than dominates.

**0.15 is measured, not chosen.** It was 0.6 on the reasoning above, and at 0.6 hybrid retrieval
scored *worse than disabling BM25 entirely* — Reciprocal Rank Fusion's agreement bonus let spurious
keyword matches promote noise over good semantic hits. `eval/weight_sweep.py` swept the parameter;
recall peaks at 0.15 (0.682) and collapses from 0.45 up (0.568). Full analysis in
[rag.md](rag.md#5-hybrid-search).

The transferable lesson: a fusion weight is a hypothesis, not a preference. This one was wrong for
eight months and nothing surfaced it until an ablation existed.

```python
orchestrator_max_loop_iterations: int = 2
```
Each extra refinement iteration costs a full generation — 12–20 s on `llama3.2:3b`, longer on the
7B models. In practice a second attempt fixes a citation error or nothing will.

---

## 3. Request lifecycle, end to end

Following `POST /api/research/query` with `{"question": "..."}`:

**1. Gateway** (`services/gateway/app.py`) resolves matter context. If documents were ingested
without a matter — the common "paste an FIR then ask" flow — the three most recent are collected
with a 700-char excerpt each and passed to the router. *This was a bug: without it the router saw
a bare question and classified `document_type: none`, so the FIR never informed the search.*

**2. Orchestrator** opens a `PipelineTrace`, giving every subsequent event a `trace_id`.

**3. Intake Router (Agent 0)** — runs **before retrieval**, and does two things:

- Decides which agents are needed. Running all of them always would waste 30–60 s per unnecessary
  agent.
- **Rewrites the input into the text that gets embedded.** This matters more than it looks.
  Embedding raw input embeds the conversational wrapper too — "Hi, can you please help, my client
  Mr Sharma is worried about…" — which dilutes legal signal in a 768-dimensional vector.

  Real example. Input: *"My client supplied goods under a written agreement and the buyer's cheque
  bounced. What provisions apply?"* plus a pasted FIR. Rewritten to: *"Dishonour of cheque issued
  towards payment for goods supplied under a written supply agreement; cheating and criminal breach
  of trust by inducing delivery without intention to pay; recovery of price and compensation for
  breach."* That is what gets embedded. The UI shows both, side by side.

  `fallback_decision()` keeps the pipeline running if the router itself fails, flagged
  `used_fallback` so the UI never presents deterministic routing as model-driven.

**4. Retrieval** — the rewritten query is embedded once, then three retrievers run and their
rankings are fused (§8).

**5. Document Understanding (Agent 1)** — only if documents exist and the router asked for it.
Extracts parties, dates, amounts, obligations, each tagged with its source `chunk_id`. Its output
becomes the "verified facts" every downstream agent may rely on.

**6. Legal Research (2) and Similar Cases (3)** run **concurrently** via `asyncio.gather`. They
share no data and each costs 25–60 s, so concurrency roughly halves the slowest phase.

**7. Verification (Agent 5)** audits each output. Deterministic checks first (§12).

**8. Drafting (Agent 4)** — if requested. Sees findings only, never raw chunks.

**9. Synthesis (Agent 6)** writes the answer: prose, key points, inference, uncertainties, next
steps. *Without this the pipeline ended with structured findings, which is research output, not an
answer.*

**10. Trace closed**, answer returned.

---

## 4. Data acquisition

### Legislation — India Code

`ingestion/india_code.py`. India Code has **no REST or OAI API** (`/rest/*`, `/oai/*` → 404), so
this uses the endpoints the site's own front end uses. The route was reverse-engineered and
verified live:

1. `GET /handle/123456789/{handle}` — the act page. **HEAD is refused**, so always GET. The page
   embeds the complete section index as anchors carrying `actid`, `sectionId`, `sectionno`.
2. `GET /SectionPageContent?actid=…&sectionID=…` → JSON `{content, footnote}`. The section text is
   **AJAX-loaded**, so the `/show-data` HTML body is empty — scraping that page yields nothing.
   The `footnote` field carries amendment history ("Subs. by Act 6 of 1899, s. 4…").
3. `GET /bitstream/.../A187209.pdf` — the consolidated PDF, kept for the document viewer.

**Two gotchas that cost real time:**

- **The WAF 403s any `User-Agent` containing parentheses.** The conventional
  `Product/1.0 (comment)` form fails; `LegalAId/0.1 academic-legal-research` works. Omitting the
  UA entirely gives a **404** instead. This looked exactly like IP rate-limiting at first.
- Bursts of rapid requests *also* get 403s. `_get()` backs off 3× harder on 403/429 and ingestion
  paces itself.

**Text cleaning is not naive.** Raw content has two defects that would silently corrupt
embeddings and quoted citations:

- `\r\n` appear **mid-sentence** ("has been\r\nbroken") — they are PDF line-wrap artifacts, not
  paragraph breaks. Real paragraph breaks are `</br><hr class="hr1"/>`. Wraps are rejoined.
- Inline tags sit flush against text (`<i>Explanation</i>.In estimating`), so naive tag stripping
  welds words together. Tags become spaces; punctuation spacing is repaired.

Pinned by `tests/test_india_code.py` using real markup fixtures.

**Repealed provisions are excluded.** Asking about "Section 73" once ranked *"Section 259 —
Repealed … Rep. by s. 73"* **third**, purely because the repeal note mentions s.73. Citing dead law
is the worst failure mode for this tool. `is_repealed()` excludes them, and because 76 had already
been ingested the filter is **retroactive** — `_purge_provisions()` removes previously stored ones
(339 → 263 provisions, on the corpus as it stood at the time).

The DSpace API later made this cheaper as well as more reliable: `dc.identifier.repealed` is a
publisher-supplied flag, so repeal status no longer has to be inferred from heading text. The
heading inference is kept as a second line of defence.

**Section filters are now the exception, not the rule.** With the corpus at 1,783 provisions only
BNSS is filtered (61 of 531 sections), because 531 sections of criminal procedure are mostly
outside the application's scope. Everything else is held in full: filtering a statute means a
question about an unfiltered section retrieves nothing, and "the corpus does not contain it" is a
far less useful answer than the provision itself.

**Empty provisions are dropped.** India Code carries the Arbitration Council sections (43A–43M)
with headings and no body text. Indexed, they were retrievable and citable while saying nothing —
worse than absent. Found by `scripts/analyse_corpus.py`, which flagged 13 of them as exact
duplicates of one another.

### Case law — Supreme Court e-SCR

`ingestion/case_law.py`. The full dataset is ~52 GB; **we never download it**. Per-year metadata
parquet is under 1 MB, and only the PDFs actually kept are fetched. Object naming
(`{path}_EN.pdf`) was discovered by listing the public bucket — the README does not state it.

Chosen for licensing: **CC-BY-4.0**, so redistributable with attribution. IL-PCSR and ILDC/CJPE
were rejected as CC-BY-**NC** with no redistribution rights.

Cleaning fixes two more real defects:

- **Ligature splitting.** PDFs emit "specifi c performance" and "fi led". Left alone the token is
  "fi", not "specific" — corrupting embeddings *and* any quoted passage. Repaired via an explicit
  legal-vocabulary map, because a general rule would weld "of the" → "ofthe".
- **S.C.R. gutter letters** (`A B C D E F G H`) and running page headers pollute chunks.

**Curation is two-stage.** The `description` column is empty and only 3 of 856 judgments look
contract-related by title, so titles alone are useless. Stage 1 screens on title (free); stage 2
scores the **court-authored headnote** after fetching, which is where e-SCR states the issues and
statutes in play. Judgments failing verification are discarded, not indexed.

> Do **not** score on the dataset's `raw_html` column — it is page furniture (language selector,
> disclaimer modal) identical across judgments. Screening on it reported 450 of 856 as
> contract-related, which is nonsense.

---

## 5. Chunking

`ingestion/chunking.py`. Three strategies, because one fixed-length splitter destroys the
provenance a legal citation depends on.

| Document type | Hierarchy |
|---|---|
| Statute | Act → Chapter → Section → Subsection → Paragraph |
| Judgment | Metadata → Facts → Issues → Reasoning → Ratio → Decision |
| Upload | Document → Page → Heading → Paragraph |

Overlap is applied **only** when a single structural unit exceeds the maximum size. Units that fit
are never overlapped — duplicating statutory text across chunks corrupts citation provenance.

**Subsection packing.** `_split_subsections` splits at subsection markers and then **packs back up
to the target size**. Without packing, IT Act s.2 (Definitions) became **28 chunks, a dozen of them
116–300 characters**. Fragments that small carry too little context to match a question and flood
top-k with near-duplicates. After packing: s.2 → 8 chunks; the whole act 32 → 12. Pinned by
`tests/test_chunk_packing.py`.

**Judgments index the headnote, not the full text.** Measured on a 6-case sample, full-text
chunking produced **341 chunks, one constitutional-bench judgment alone producing 151** — that
single case would dominate every similarity search. The court-authored headnote states facts,
issues and holding: exactly what precedent matching needs, authoritative rather than our
summarisation, a few thousand characters instead of hundreds of thousands. Result: 33 chunks, 2–8
per case.

---

## 6. Embeddings

`services/embedding/service.py`. Separate from the LLM service on purpose: different model,
different batching profile, and it sits on both the write path (ingestion) and read path (query).

Endpoints: `/embeddings/document`, `/embeddings/query`, `/embeddings/chunks`.

**Batching is load-bearing, not an optimisation** — see §2. The service must use `/api/embed`
(batched) rather than `/api/embeddings` (single).

Per-item latency is reported as the batch cost shared out, with the batch total separate, so the
UI never implies each vector cost the full request time.

Semantic sanity is verified rather than assumed: `cosine(contract, contract-ish) = 0.8237` vs
`cosine(contract, conspiracy) = 0.5561`.

---

## 7. Vector store

`services/vector/store.py`. FAISS, **one index per collection** — `LEGAL_ACTS`, `CASE_LAW`,
`USER_DOCUMENTS` and the rest are never merged, so retrieval always knows what it is searching.

- **Flat inner-product on L2-normalised vectors.** With normalised vectors, inner product *is*
  cosine, and flat search is *exact*. The corpus is <1k vectors, so IVF/HNSW would add
  approximation error and tuning burden for no measurable gain.
- **`IndexIDMap2`** maps our stable chunk ids to FAISS positions, which is what makes delete work.
- **Metadata sidecar** as JSON. Filters are applied by over-fetching candidates *before* top-k
  selection, so a filter can never silently shrink results below `top_k` when matches exist.
- `candidate_count` reported to the UI is the honest total in the collection, not the number FAISS
  happened to scan.

---

## 8. Hybrid retrieval

Three retrievers, fused. Each is independently switchable so the UI can demonstrate its
contribution.

### Dense
Cosine over embeddings. Good at meaning, bad at exact identifiers.

### Keyword — BM25 (`retrieval/keyword.py`)
Implemented directly (~40 lines) rather than adding a dependency. Okapi BM25 with `k1=1.5`,
`b=0.75`.

**Citation normalisation.** "Section 73", "S. 73" and "sec 73" all become the token `section_73`;
"Order VII Rule 11" additionally emits the combined `order_vii_rule_11`.

> **A subtle bug worth knowing about.** `tokenize` produced `section_73`, but `_TOKEN_RE` excluded
> `_` and split it straight back into `section` + `73`. Since "section" appears in **71%** of the
> corpus and "73" in **1.4%**, the precise half was swamped by the generic half. Underscore must be
> a word character. After the fix `section_73` has df=8 (1.1%).

**Citation boost (3×).** Plain BM25 rewards short documents, so a 38-token chunk matching only
{indian, contract, act, deal} outranked chunks that actually contained s.73. An exact statutory
citation is the strongest lexical evidence available, so it is weighted accordingly.

### Fusion — Reciprocal Rank Fusion (`retrieval/fusion.py`)

```
RRF(d) = Σ  weight / (k + rank_r(d))       k = 60
```

Cosine sits in [-1,1]; BM25 is unbounded and corpus-dependent. Normalising them into a weighted
sum needs a scale that shifts whenever the corpus changes. RRF combines **ranks**, so no
calibration is needed, it degrades gracefully when one retriever returns nothing, and it rewards
documents both retrievers agree on. Ties break toward agreement, then dense similarity.

Every hit records which retrievers found it, which the UI renders as badges.

### Explanations (`retrieval/explain.py`)
Similarity alone is not an explanation. 18 legal concept groups map surface terms to concepts, so
the UI can say "matched legal concepts: breach, compensation" rather than listing word overlaps.

These are **lexical evidence derived from the text, never model-generated** — an LLM-written
justification could sound convincing for a bad match.

---

## 9. Graph RAG

`retrieval/graph.py`. A third retriever that uses **no vectors at all**.

Dense finds text that *means* something similar; BM25 finds text containing the same *words*.
Neither can answer "which judgments interpret this section?", because that is a question about
relationships.

```
Act ──contains──> Section ──cited_by──> Judgment ──considers_act──> Act
                     │
                     └──cross_references──> Section
```

Edges come from real evidence captured during ingestion: statutory cross-references parsed out of
provision text, and the statutes a judgment's headnote says it considered. Current graph: **1,845
nodes, 2,649 edges** (1,783 provisions, 47 cases, 15 acts).

`find_seeds()` locates entry points named in the question; `seed_from_chunks()` seeds from what
dense search already found; `expand()` does bounded BFS. Traversal is bidirectional — "what does
this cite?" and "what cites this?" are both meaningful.

Measured contribution to retrieval recall: **0.000**. See [rag.md](rag.md#6-graph-rag) for why, and
why it is reported rather than quietly dropped.

---

## 10. The agent layer

Seven agents. Each has its own prompt file, its own model parameters, and its own **context
policy**.

| # | Agent | Sees | Deliberately withheld |
|---|---|---|---|
| 0 | Intake Router | raw input, document excerpts | the knowledge base (runs before retrieval) |
| 1 | Document Understanding | matter documents only | all statute and case law |
| 2 | Legal Research | 5 law + 2 document chunks | **all case law** |
| 3 | Similar Cases | 4 judgments (1 per case) + 2 provisions | **all matter documents** |
| 4 | Drafting | verified facts + both agents' findings | **all raw chunks** |
| 5 | Verification | the reviewed agent's evidence set | nothing beyond it |
| 6 | Synthesis | all findings | **all raw chunks** |

The withholding is deliberate, not incidental:

- Legal Research is denied case law because its job is statute identification; precedent would
  pull it toward predicting outcomes.
- Similar Cases gets **one chunk per judgment**, so four *different* precedents are in view rather
  than four fragments of the same one.
- Drafting and Synthesis see **no raw chunks**. Giving a drafting model raw statutory text invites
  it to paraphrase provisions into the document as if they were its own findings — which is
  exactly how fabricated authority gets in. Working from verified findings keeps every authority
  traceable to an agent that cited a chunk for it.

`AgentContext` records both `included` and `excluded` items **with reasons**, because isolation is
only provable if you can show what an agent was *not* given.

### Prompts
`prompts/<name>/<version>.yaml`, never inline in Python. Each file carries its own temperature and
token budget — extraction wants determinism (0.0–0.1), drafting needs latitude (0.2). Rendering
raises if a variable is missing rather than substituting blank: a silently empty context section
would make the model answer confidently with no evidence while the trace looked normal.

---

## 11. Constrained decoding — the single most important fix

Code Llama 7B **cannot follow a JSON schema from instructions alone.**

First full run: both research agents failed to parse **twice each**, and the raw output was the
model echoing the retrieved statute's *Illustrations* back as:

```json
{"73": [{"a": "A contracts to sell 50 maunds of saltpetre to B..."}]}
```

It pattern-matched its input instead of obeying the schema. Total: **640 s, zero usable output.**

Ollama accepts a **JSON Schema as the `format` parameter** and constrains token sampling to it,
making off-schema output structurally impossible.

| Agent | Before | After |
|---|---|---|
| Legal Research | 363 s, parse **FAILED** | **33 s, valid, audit 0 problems** |
| Similar Cases | 172 s, wrong shape | **22 s, valid, audit 0 problems** |

Supporting changes that mattered:

- **Illustrations stripped from prompts** (`trim_for_prompt`). Kept in the knowledge base for
  citation, but in a prompt they tripled length *and* supplied the lettered-list pattern the model
  was copying. Chunks capped at 700 chars for prompts.
- **`additionalProperties: false`** everywhere, so the model cannot bolt on invented keys.
- **`maxItems` on every array.** Constrained decoding guarantees the output *shape* but not that a
  complete object fits the budget. One run still failed at exactly `out=500` — truncated
  mid-array, because the model opened a long `sections` list and ran out. `tests/test_schemas.py`
  now fails the build if any array is unbounded, and it caught one I had missed.

---

## 12. Orchestration and the refinement loop

`agents/orchestrator.py`.

**Verification is deterministic-first.** Set membership of cited `chunk_id`s, section numbers and
case titles is exact, cheap, and cannot itself hallucinate — so it runs first and is
authoritative. The LLM critic only judges what a set operation cannot: overclaiming, conclusions
beyond the evidence, confident language over thin material. Asking a 7B model to check citation
membership would be both slower and less trustworthy than a set lookup.

**The loop.** An agent whose output fails the audit is re-run with concrete corrective guidance
naming the offending citations. Every iteration is recorded, including when a retry fails to fix
the problem. On the final iteration, ungrounded citations are **stripped** rather than presented as
authority — better to return fewer verifiable authorities than one the system cannot substantiate.

> **A false positive worth knowing about.** The loop once rejected `"Section 73 of Indian Contract
> Act, 1872"` as fabricated even though the Legal Research Agent had established
> `section 73 | Indian Contract Act, 1872`. The strings differed only in *format*. That wasted a
> full 30–60 s iteration **teaching the model to drop a correct citation** — worse than not
> checking at all. `_authority_key()` now normalises to `s73:indian`, so equivalent citations
> compare equal while s.73 and s.74 still differ.

---

## 13. Observability

`core/tracing.py`. Every request creates a `PipelineTrace`; every event references its `trace_id`.

```
pipeline_trace
  ├── trace_stage[]      ordered, timed, service, transport, input, output
  ├── retrieval_run[]    → retrieval_result[] → chunk
  ├── agent_run[]        → context manifest (included AND excluded)
  ├── model_run[]        → prompts, params, tokens, latency
  └── answer / summary
```

`TraceRecorder.stage()` is a context manager that records failures and re-raises, so a broken
stage appears in the UI instead of vanishing. Live progress is pushed over SSE
(`/api/research/stream`).

Two constraint decisions worth noting:

- `agent_runs.trace_id` is `NOT NULL` **by design** — every agent execution must be traceable. But
  agents can legitimately run standalone (tests, single-agent calls), so rather than weaken the
  constraint, `_ensure_trace()` creates a minimal trace.
- Deletion runs **leaves-inward**. `retrieval_results` and `citations` reference `chunks`, so
  deleting chunks first raises `FOREIGN KEY constraint failed` — and the reset then silently did
  nothing, after which ingestion reported all zeros because rows already existed. That looked like
  success in the log. `_delete_chunks()` now clears dependents first and drops the FAISS vector
  too, so search can never return a chunk id that no longer resolves.

---

## 14. Database

26 tables, SQLite for development, schema Postgres-ready. Full detail in
[database-schema.md](database-schema.md).

The tenancy model is forward-looking: `organizations`, `roles`, `permissions`, `memberships`,
`matter_members`, `document_permissions`, `audit_logs` all exist now. A personal workspace is
simply an organization of one, so **firm mode is authorisation logic plus UI, not a migration of
core tables**. The seed creates 7 roles and 14 permissions with 68 grants.

The seed deliberately stores **no password hash** — Phase 1 is single-user and unauthenticated, and
a fake hash would imply an auth system that does not exist.

Embedding vectors live in FAISS, not in the relational database; `embeddings_metadata` stores
bookkeeping and a 12-value preview for the UI.

---

## 15. Frontend

React 18 + TypeScript + Vite. No component library — the CSS is purpose-built for information
density. Vite proxies `/api` to the gateway so dev has one origin and no CORS handling.

Navigation is split deliberately:

- **For you** — Home, Ask LegalAId, My documents, Law library. A practising lawyer's workspace.
- **Developer / evaluation** — Pipeline trace, Agents, Retrieval inspector, Chunk explorer,
  Embeddings, RAG on/off, System & models.

The answer comes first on the Research page; the machinery is behind "Show how this answer was
produced". `AnswerView` labels **AI inference** separately from what the sources say, so a reader
can always tell interpretation from law.

---

## 16. Testing

```powershell
$env:PYTHONPATH="backend\src"
.\.venv\Scripts\python.exe -m pytest backend\tests -q
```

118 tests. Most exist because something actually broke:

| File | Pins |
|---|---|
| `test_india_code.py` | Real-markup fixtures: ligatures, PDF wraps, headnote terminators |
| `test_chunking.py` | Statute/judgment/document strategies |
| `test_chunk_packing.py` | The 28-fragment definitions-section regression |
| `test_repealed_filter.py` | The "Section 259 — Repealed" ranking failure |
| `test_hybrid_retrieval.py` | Citation tokenisation, the underscore bug, RRF properties |
| `test_case_law.py` | Gutter letters, running headers, ligature vocabulary |
| `test_agents.py` | Context isolation, grounding audit, authority normalisation |
| `test_schemas.py` | **Every array bounded, every object closed** |

Smoke scripts hit the live stack: `smoke_answer.py`, `smoke_orchestrator.py`,
`smoke_ui_endpoints.py`, `smoke_retrieval.py`, `smoke_hybrid.py`, `check_corpus_quality.py`.

---

## 17. Performance characteristics

Measured on the dev machine (Ryzen 5 7235HS, RTX 3050 6 GB, 25 GB RAM).

| Operation | Time |
|---|---|
| Query embedding | 1–7 s (cold model load dominates) |
| Vector search (651 vectors) | 2–4 ms |
| BM25 search | <5 ms |
| Graph expansion | <5 ms |
| Intake router | 30–40 s |
| Legal Research | 25–80 s |
| Similar Cases | 20–140 s |
| Synthesis | 50–60 s |
| **Full pipeline** | **170–460 s** |

Generation is the bottleneck at every step — measured at **39.8 tok/s** on `llama3.2:3b` and
**86.2% of total request wall-clock** (`eval/latency_breakdown.py`). Code Llama 7B runs closer to
12 tok/s because it is the only installed model that does not fit entirely in 6 GB of VRAM; it
offloads at a 22%/78% CPU/GPU split. Consequences designed around it: bounded
token budgets, trimmed prompt context, concurrent independent agents, capped loop iterations, and
SSE streaming so a multi-minute run stays legible.

`qwen2.5:3b-instruct` is installed as a faster alternative, selectable in the UI.

---

## 18. Extending the system

**Add an act.** Find its handle with `scripts/find_handles.py`, add an `ActSpec` to
`ingestion/corpus.py` (with `section_filter` if it is large), then
`python scripts/ingest_legislation.py --act <key>`.

**Add more case law.** `python scripts/ingest_cases.py --limit 200 --years 2023 2022 2021`.
Adjust `TOPIC_WEIGHTS` in `case_curation.py` to change what counts as relevant.

**Add an agent.** Write `prompts/<name>/v1.yaml`, add a bounded schema to `agents/schemas.py`,
subclass `Agent` implementing `build_context` / `render` / `validate`, then wire it into the
orchestrator. Add its `AgentKey` and `StageKey` so it appears in traces.

**Swap the model.** `LLM_MODEL=<name>` in the environment, or pick it in the UI per request.

**Switch to real HTTP between services.** `SERVICE_TRANSPORT=http`. The same handlers serve both
transports through identical Pydantic contracts, so behaviour cannot drift and there is no mock
service anywhere. The UI displays which transport served each hop.
