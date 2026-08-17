# LegalAId — UI Information Architecture

Guiding principle: **make the invisible visible.** The UI is a legal-research workspace with a
built-in observability layer, not a chat window.

## 1. Shell

```
┌──────────────────────────────────────────────────────────────────────────┐
│ LegalAId          Matter: ABC v XYZ ▾     Ollama ●  codellama ●   ☾/☀    │
├───────────────┬──────────────────────────────────────────────────────────┤
│ WORKSPACE     │                                                          │
│  Dashboard    │                                                          │
│  Matters      │                    Main Workspace                        │
│  Documents    │                                                          │
│  Research     │                                                          │
│  Similar Cases│                                                          │
│  Drafting     │                                                          │
│  Knowledge    │                                                          │
│ ───────────── │                                                          │
│ OBSERVABILITY │                                                          │
│  Pipeline     │                                                          │
│  Retrieval    │                                                          │
│  Chunks       │                                                          │
│  Embeddings   │                                                          │
│  Agents       │                                                          │
│  Models       │                                                          │
│  System       │                                                          │
├───────────────┴──────────────────────────────────────────────────────────┤
│ AI-generated legal research for professional review. Verify before use.  │
└──────────────────────────────────────────────────────────────────────────┘
```

The sidebar is split into **Workspace** (what a lawyer does) and **Observability** (what an evaluator
inspects). This separation is deliberate: the professor's checklist maps onto the second group,
while the first group demonstrates a genuine legal workflow.

The disclaimer bar is persistent and not dismissible.

## 2. Pages

### Dashboard
Matter/document/chunk/vector/case counters; Ollama + Code Llama status; capability checklist by
exercise phase; performance metrics (avg embedding / retrieval / agent / LLM / pipeline latency,
queries processed, success vs failure); recent traces with drill-in.

### Matters → Matter Workspace
Metadata, parties, uploaded files, extracted facts, identified issues, research history, drafts,
activity timeline. This is the lawyer's home.

### Documents
Upload with validation; processing status; per-page extracted text; detected structure; chunk list.
**Document Viewer** shows original ↔ extracted text ↔ structure ↔ chunks side by side, and clicking a
chunk highlights its source page.

### Research
The primary query surface. Question box, retrieval-mode selector, and a **RAG ON / RAG OFF / COMPARE**
toggle. During execution it renders the **live pipeline animation** (stage-by-stage, currently-running
highlighted, via SSE). Result is the structured answer with FACT / LAW / CASE / AI-INFERENCE /
UNCERTAINTY tagging and inline citations that open the underlying chunk.

### Similar Cases
Case comparison table: title, court, citation, similarity, why-similar, principle, outcome,
differences, source link.

### Drafting
Draft type / tone / length / purpose controls, generated draft, and a grounding panel showing exactly
which research items and cases supported it. `[INFORMATION NOT AVAILABLE]` markers render as warnings.

### Knowledge Base
Tabs per collection — Acts, Rules, Regulations, Notifications, Cases. Shows source authority,
licence, attribution, retrieval date, item and chunk counts. Ingestion trigger and progress.

### Pipeline (Trace)
Two synchronised views:
- **Node graph** — the visual architecture diagram, each node clickable, coloured by status/latency.
- **Stage list** — `01 USER INPUT ✓ … 10 FINAL RESPONSE ✓`, each expandable to real recorded
  input/output JSON, service name, transport and duration.

### Retrieval Inspector
Per trace: query text, embedding model, dimensions, then per collection the candidate count, top-k,
filters and ranked results with similarity scores. Each result expands into **Why retrieved?** —
matched legal concepts, matched terms, similarity — and links to the chunk.

### Chunk Explorer
Searchable/filterable table over all chunks. Detail view: id, document, page, section, heading,
collection, source type, text, char + token counts, embedding model, dimensions, vector id, created
timestamp, strategy — plus retrieval history (`Retrieved? YES · similarity 0.9241 · used by Legal
Research Agent, Drafting Agent`).

### Embeddings
Model, dimensions, source chunk, vector id, latency, and a vector preview (values + sparkline).

### Agents
Per agent: responsibility, model, prompt name/version, tools, status, avg latency, recent runs. Each
run opens the **Agent Context panel** (included vs excluded context) and the **Model Context panel**:
system prompt, user prompt, retrieved context grouped by law/case/document, model parameters,
token counts, latency.

### Models
Configured LLM and embedding models, provider, live Ollama availability, installed model list,
parameters, and aggregate usage/latency/token stats.

### System
Service health for all nine services with transport mode, Ollama connectivity, FAISS index stats,
database stats, Docker status, capability/phase matrix.

## 3. Cross-cutting components

- `TraceStageCard` — expandable stage with status, timing, payload viewer.
- `ChunkCard` / `ChunkDrawer` — used anywhere a chunk is referenced, so provenance is always one click away.
- `CitationChip` — click opens the citation object and its chunk.
- `WhyRetrieved` — similarity + matched concepts explanation.
- `ModelContextPanel` — the "what did the model see?" viewer.
- `EvidenceChain` — answer → supporting evidence → chunks → documents → agents → model calls.
- `SourceBadge` — FACT / LAW / CASE / AI INFERENCE / UNCERTAINTY, consistently coloured everywhere.

## 4. Visual language

Professional legal-tech: dark and light themes, restrained palette (slate/ink neutrals, a single
indigo accent, semantic status colours), dense tables, tabs, timelines, expandable panels. Monospace
for identifiers, vectors and JSON. No gimmicky AI flourishes; the only animation is the pipeline
progress indicator, which conveys real state.

Accessibility: semantic landmarks, keyboard-navigable disclosure widgets, visible focus rings,
ARIA live region for pipeline progress, status conveyed by icon + text rather than colour alone,
WCAG AA contrast targets in both themes.
