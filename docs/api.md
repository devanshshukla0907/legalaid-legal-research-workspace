# API

One documented surface at `http://localhost:8000`. Interactive docs at
[`/docs`](http://localhost:8000/docs); machine-readable schema at `/openapi.json`.

Every response carries an `X-Trace-Id` header. Errors are uniform:

```json
{ "error": { "code": "not_found", "message": "Chunk chk_x not found", "detail": {}, "trace_id": "LA-1a2b3c" } }
```

---

## Research

### `POST /api/research/query`
Runs the full multi-agent pipeline. Takes minutes — see the streaming endpoint for progress.

```json
{
  "question": "My client supplied goods and the buyer refused payment. What applies?",
  "pasted_text": "",
  "matter_id": null,
  "model": null,
  "enable_drafting": null,
  "enable_verification": true,
  "instructions": ""
}
```

`model` overrides the LLM for this request only and flows to every agent. `enable_drafting: null`
lets the intake router decide.

The response is deliberately verbose — it is what makes the UI's observability possible:

| Field | Contents |
|---|---|
| `trace_id` | Look anything up later with this |
| `answer` | **The synthesised answer**: `answer_markdown`, `key_points`, `inference`, `uncertainties`, `next_steps`, `confidence` |
| `routing` | Router decision: task, document type, agents chosen, `original_input` vs `search_query` |
| `plan` | Execution steps with which ran in parallel |
| `facts` / `legal` / `cases` / `draft` | Structured per-agent output |
| `retrieval` | Query embedding, per-collection searches, every retrieved chunk with match reasons |
| `agents` | Per agent: model, prompt version, tokens, latency, attempts, **full context manifest (included *and* excluded)**, system prompt, user prompt, raw text |
| `loop_iterations` | Each refinement pass, its audit and corrective guidance |
| `warnings` | Anything that did not go cleanly |

### `GET /api/research/stream?trace_id=…`
Server-sent events for live progress: `trace_started`, `stage_started`, `stage_finished`,
`trace_completed`.

### `POST /api/research/compare`
The same question through the same model with and without retrieval.

```json
{ "question": "What does Section 73 of the Indian Contract Act deal with?" }
```

Returns `without_rag` and `with_rag` (each with answer, tokens, latency, sources) plus the exact
`retrieved_context` injected. The default question is a real, reproducible failure: without
retrieval Code Llama says s.73 concerns "the sale of goods"; it is compensation for breach.

---

## Retrieval

### `POST /api/retrieval/search`
Retrieval with no agents.

```json
{
  "question": "compensation for breach of contract",
  "mode": "COMBINED_LEGAL_RESEARCH",
  "top_k": 8,
  "enable_keyword": true,
  "enable_graph": true
}
```

Modes: `LAW_SEARCH`, `CASE_SEARCH`, `USER_DOCUMENT_SEARCH`, `COMBINED_LEGAL_RESEARCH`.

The `enable_keyword` / `enable_graph` toggles are per-request, which is what makes each retriever's
contribution demonstrable rather than asserted. Each returned chunk carries
`metadata.retrievers` — `["dense","keyword","graph"]` — and `match_reasons`.

---

## Inspection

### `GET /api/traces?limit=30`
### `GET /api/traces/{trace_id}`
Everything recorded for one request: `stages` (ordered, with service, transport, duration, input,
output), `agent_runs` (with context manifests), `model_runs` (**with the exact system and user
prompts and the raw response** — this is the "what did the model see?" data), and `retrievals`
(candidate counts, query vector preview, per-result similarity and match reasons).

### `GET /api/chunks?collection=&q=&limit=&offset=`
### `GET /api/chunks/{chunk_id}`
One chunk in full: text, all metadata, its embedding record, the **first 64 components of the
actual vector**, retrieval history, which agents have used it, and provenance — for a provision the
act, section, official source URL and amendment footnotes; for a judgment the citation, court,
bench, outcome and licence.

### `GET /api/embeddings?limit=50`
Model, dimensions, L2 norm, latency, token count and a 12-value preview per vector.

---

## Knowledge base

### `GET /api/knowledge`
Sources with licence and attribution, acts with provision counts, all judgments, vectors per
collection.

### `GET /api/knowledge/provisions?act=&q=`
Full statutory text with source URLs and amendment notes.

### `GET /api/knowledge/graph?limit=160`
Graph-RAG nodes and edges with the evidence string for each edge.

---

## Documents

### `POST /api/documents/upload`
`multipart/form-data`: `file`, optional `matter_id`, optional `doc_class`. PDF, DOCX, TXT, MD up to
25 MB.

### `POST /api/documents/paste`
```json
{ "text": "FIRST INFORMATION REPORT …", "filename": "FIR-0412.txt" }
```

Both extract with page provenance, classify, chunk, embed and index, returning page/chunk/vector
counts and the detected `doc_class`. Pasting is a first-class path because in practice a lawyer
pastes an FIR rather than attaching a PDF.

### `GET /api/documents` · `GET /api/documents/{id}`
The document with its extracted pages and the chunks derived from each.

---

## Platform

### `GET /api/system/health`
Status, Ollama reachability and model availability, all three retrievers' sizes, row counts, and
the capability matrix.

### `GET /api/models`
Installed Ollama models with `usable_for_generation` (embedding-only models are flagged so the UI
does not offer them for generation).

### `GET /api/agents`
The seven-agent roster: responsibility, what each sees, **what each is withheld**, prompt version,
temperature, token budget, and measured recent performance.

### `GET /api/prompts`
Every versioned prompt with its full system and user template.

### `GET /api/metrics`
LLM/retrieval/pipeline averages and per-agent breakdown, computed from recorded runs.

### `GET /api/stages`
The canonical pipeline stage list, so the UI can render the full graph.

---

## Matters

### `GET /api/matters`
Matters with document counts. Matter CRUD beyond this is Phase 1 scope; the schema already supports
organisations, roles and per-matter membership.
