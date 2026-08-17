# Retrieval and RAG

How text becomes a vector, how a question finds the right provision, and why there are three
retrievers instead of one.

---

## The pipeline

```
Question or document
      │
      ▼
Intake Router ──── rewrites the text into dense legal language for embedding
      │
      ▼
Query embedding (nomic-embed-text, 768-d, L2-normalised)
      │
      ├─────────────────┬─────────────────┐
      ▼                 ▼                 ▼
Dense (FAISS)      BM25 keyword       Graph traversal
cosine similarity  exact tokens       no vectors at all
      │                 │                 │
      └────────┬────────┴─────────────────┘
               ▼
    Reciprocal Rank Fusion
               ▼
    Metadata filter → top-k → explanations
               ▼
    Per-agent context assembly (with exclusions recorded)
```

---

## 1. The query rewrite

Before anything is embedded, the intake router rewrites the input. This is the cheapest large win
in the whole retrieval path.

Embedding raw user input embeds the conversational wrapper too. "Hi, can you please help me, my
client Mr Sharma is very worried about…" spends most of the vector's capacity on politeness and a
party name. The router strips that and keeps the legal substance:

> **Input:** "My client supplied goods under a written agreement and the buyer's cheque bounced.
> What provisions apply?" (plus a pasted FIR)
>
> **Embedded:** "Dishonour of cheque issued towards payment for goods supplied under a written
> supply agreement; cheating and criminal breach of trust by inducing delivery without intention to
> pay; recovery of price and compensation for breach."

The UI shows both, so the transformation is visible rather than hidden preprocessing.

---

## 2. Chunking

One fixed-length splitter is wrong for legal text: it cuts across section boundaries and destroys
the provenance a citation depends on. Three strategies:

| Type | Hierarchy |
|---|---|
| Statute | Act → Chapter → Section → Subsection → Paragraph |
| Judgment | Metadata → Facts → Issues → Reasoning → Ratio → Decision |
| Upload | Document → Page → Heading → Paragraph |

Overlap is applied **only** when a single structural unit exceeds the maximum. Units that fit are
never overlapped — duplicating statutory text across chunks corrupts citation provenance.

Every chunk carries: `chunk_id`, `document_id`/`provision_id`/`case_id`, `collection`,
`source_type`, `source_title`, `section`, `chapter`, `page`, `heading`, `char_count`,
`token_count`, `ordinal`, `chunk_strategy`, `source_url`, `version`, and a metadata blob including
a ready-made `citation` string.

### Two chunking failures worth knowing

**Definitions sections fragmented.** `_split_subsections` split at every subsection marker and only
merged pieces below `chunk_min_chars` (120). IT Act s.2 became **28 chunks, a dozen of them
116–300 characters**. Chunks that small carry too little context to match a natural-language
question, and they flood top-k with near-duplicates. Fixed by *packing* subsections up to
`chunk_target_chars` while still splitting on subsection boundaries, so a chunk never begins
mid-clause. s.2: 28 → 8 chunks; the whole act 32 → 12.

**Judgments exploded.** Full-text chunking of 6 judgments produced **341 chunks, one
constitutional-bench case alone producing 151** — that case would then dominate every similarity
search, and its page-by-page procedural narrative is noise for "find me a similar case". Now the
court-authored **headnote** is indexed: it states the facts, the issues for consideration and what
was held. Authoritative rather than our summarisation, and a few thousand characters instead of
hundreds of thousands. Result: 33 chunks, 2–8 per case.

---

## 3. Embeddings

`nomic-embed-text` via Ollama, 768 dimensions, L2-normalised so inner product *is* cosine.

**Batching is essential, not an optimisation.** Each Ollama embedding request carries ~2.2 s of
fixed overhead that barely varies with input length:

| Input | Latency |
|---|---|
| 152 chars | 2221 ms |
| 456 chars | 2231 ms |
| 1520 chars | 2255 ms |
| 3040 chars | 2341 ms |

So cost is driven by *request count*, not text size. Using the batched `/api/embed`:

| Batch | ms/chunk | Speed-up |
|---|---|---|
| 1 | 2249 | 1× |
| 16 | 161 | 14× |
| 32 | 91 | 25× |
| 64 | 54 | 42× |

Verified `cosine(single, batched) == 1.000000` — identical vectors, pure throughput. Full-corpus
embedding: ~13 minutes → ~20 seconds.

Semantic ordering is verified rather than assumed:
`cosine(contract, contract-ish) = 0.8237` vs `cosine(contract, conspiracy) = 0.5561`.

---

## 4. Vector store

FAISS, **one index per collection**. `LEGAL_ACTS`, `LEGAL_RULES`, `LEGAL_REGULATIONS`,
`LEGAL_NOTIFICATIONS`, `CASE_LAW`, `USER_DOCUMENTS` are never merged, so retrieval always knows
what source type it is searching.

Flat inner-product with `IndexIDMap2`:

- Normalised vectors + inner product = **exact** cosine. No approximation error.
- The corpus is <1k vectors; IVF/HNSW would add tuning burden and error for no measurable gain.
- `IndexIDMap2` maps our stable chunk ids to FAISS positions, which is what makes delete work.
- Metadata filters over-fetch candidates *before* top-k selection, so a filter can never silently
  shrink results below `top_k` when matches exist.

---

## 5. Hybrid search

### Why not dense alone

Dense embeddings are good at meaning and bad at exact identifiers. "Section 73" and "Order VII
Rule 11" are precise tokens; an embedding will happily return a semantically similar but *legally
different* provision. Observed: a query naming s.73 also scored a **repealed** s.259 at 0.7975,
purely because its repeal note mentioned s.73. Dense similarity could not distinguish a provision
from a reference to it.

### BM25

Okapi BM25 (`k1=1.5`, `b=0.75`), implemented directly in ~40 lines rather than adding a dependency.

**Citation normalisation** makes "Section 73", "S. 73" and "sec 73" the same token `section_73`;
"Order VII Rule 11" also emits the combined `order_vii_rule_11`.

> **The bug that made this matter.** `tokenize` produced `section_73`, but the token pattern
> excluded `_` and split it straight back into `section` + `73`. "section" appears in **71%** of the
> corpus; "73" in **1.4%**. The precise half was swamped by the generic half. After the fix,
> `section_73` has df=8 (1.1%).

**Citation boost (3×).** Plain BM25 rewards short documents, so a 38-token chunk matching only
{indian, contract, act, deal} outranked chunks that actually contained `section_73`. An exact
statutory citation is the strongest lexical evidence available.

### Reciprocal Rank Fusion

```
RRF(d) = Σ  weight / (k + rank_r(d))       k = 60,  dense 1.0,  keyword 0.6
```

Cosine sits in [-1,1]; BM25 is unbounded and corpus-dependent. Normalising them into a weighted sum
requires a scale that shifts whenever the corpus changes. RRF combines **ranks**, so no calibration
is needed, it degrades gracefully when one retriever returns nothing useful, and it rewards
documents both retrievers agree on. Ties break toward agreement, then dense similarity.

---

## 6. Graph RAG

A third retriever using **no vectors**. Dense finds similar meaning; BM25 finds shared words.
Neither answers "which judgments interpret this section?", because that is a question about
relationships.

```
Act ──contains──> Section ──cited_by──> Judgment ──considers_act──> Act
                     │
                     └──cross_references──> Section
```

Edges are built from evidence captured during ingestion — statutory cross-references parsed out of
provision text, and the statutes a judgment's headnote says it considered. Nothing is inferred by a
model.

Current graph: **322 nodes, 315 edges** (263 provisions, 47 cases, 5 acts; relations `contains`
263, `cross_references` 27, `cites_section` 13, `considers_act` 12).

Traversal is bidirectional bounded BFS — "what does this cite?" and "what cites this?" are both
legally meaningful. Seeds come from provisions named explicitly in the question.

---

## 7. Why was this retrieved?

Similarity alone is not an explanation. `retrieval/explain.py` maps surface terms to 18 legal
concept groups, so the UI can report:

```
Found by BOTH retrievers — dense rank 1, keyword rank 2
High semantic similarity to the question (0.8182)
Shared legal concepts: breach, compensation, contract, remedy
Overlapping terms: breach, caused, compensation, contract, loss
```

These are **lexical facts derived from the text, never model-generated.** An LLM-written
justification could sound convincing for a bad match, which would be worse than no explanation.

---

## 8. Measured quality

`scripts/smoke_retrieval.py`, against 651 vectors:

| Question | Top result |
|---|---|
| "compensation for loss caused by breach of contract" | **s.73 Contract Act @ 0.8182**, then s.75, s.74 |
| "is a penalty clause enforceable" | **s.74 @ 0.7300** |
| "agreement void for want of consideration" | **s.25 @ 0.7617** |
| "are electronic records legally recognised" | **IT Act s.4 @ 0.7680**, s.10A, s.3, BSA s.61 |

The last one is worth noting: before repealed provisions were purged, that query returned
irrelevant Contract Act sections at ~0.56. Corpus hygiene moved the needle more than any tuning.

---

## 9. RAG on vs off

`POST /api/research/compare`, or the **RAG on / off** page.

The headline example is a genuine reproducible failure. Asked *"What does Section 73 of the Indian
Contract Act deal with?"* with no retrieval, Code Llama answers that it concerns **"the sale of
goods"**. It is compensation for breach of contract. With retrieval the model is grounded in the
official text and cites it.

That failure was not staged — it surfaced during a smoke test of the LLM path, before any RAG
existed.

---

## 10. Tuning

All in `config.py` / environment:

```bash
RETRIEVAL_TOP_K_LAWS=8
RETRIEVAL_TOP_K_CASES=5
RETRIEVAL_TOP_K_USER_DOCS=6
RETRIEVAL_MIN_SIMILARITY=0.25
RETRIEVAL_CANDIDATE_MULTIPLIER=4      # over-fetch before filtering
RETRIEVAL_ENABLE_KEYWORD=true
RETRIEVAL_ENABLE_GRAPH=true
RETRIEVAL_DENSE_WEIGHT=1.0
RETRIEVAL_KEYWORD_WEIGHT=0.6
RETRIEVAL_GRAPH_HOPS=2
CHUNK_TARGET_CHARS=1400
CHUNK_MAX_CHARS=2600
CHUNK_OVERLAP_CHARS=180
```

Note that agent context budgets are tighter than retrieval top-k (5 law chunks, 4 cases). Retrieval
casts a wider net; the agents get the best of it, because prompt tokens are the dominant cost at
~12 tok/s.
