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
RRF(d) = Σ  weight / (k + rank_r(d))       k = 60,  dense 1.0,  keyword 0.15
```

Cosine sits in [-1,1]; BM25 is unbounded and corpus-dependent. Normalising them into a weighted sum
requires a scale that shifts whenever the corpus changes. RRF combines **ranks**, so no calibration
is needed, it degrades gracefully when one retriever returns nothing useful, and it rewards
documents both retrievers agree on. Ties break toward agreement, then dense similarity.

### The keyword weight was wrong, and the ablation caught it

`retrieval_keyword_weight` was **0.6**, chosen by reasoning that semantic match should dominate for
natural-language questions. The reasoning was sound. The number was wrong, and it made hybrid
retrieval **worse than turning BM25 off entirely**.

The mechanism is RRF's agreement reward. A document both retrievers found accumulates both
contributions, so one dense-ranked 20th but keyword-ranked 1st beats one dense-ranked 5th:

```
dense rank 5,  keyword absent   →  1/(60+5)                = 0.01538
dense rank 20, keyword rank 1   →  1/(60+20) + 0.6/(60+1)  = 0.02234
```

That is exactly what RRF is designed to do, and it is correct when both retrievers are strong. On a
2,575-chunk legal corpus BM25 is not: with no section number in the question it matches generic
legal vocabulary — *contract*, *party*, *compensation* — and those spurious agreements promoted
noise over good semantic hits.

`eval/weight_sweep.py` replaced the guess with a measurement over the 22 evaluation tasks that have
known expected sections:

| `keyword_weight` | 0.00 | 0.05 | 0.10 | **0.15** | 0.20 | 0.30 | 0.45 | 0.60+ |
|---|---|---|---|---|---|---|---|---|
| Mean recall | 0.614 | 0.659 | 0.636 | **0.682** | 0.659 | 0.614 | 0.568 | 0.568 |

The curve peaks at 0.15 and is flat from 0.45 up. So BM25 does earn its place — worth **+0.068
recall over dense alone** — but only at roughly a quarter of the configured weight. At 0.6 it was
costing 0.046 recall and doubling latency.

Full before/after, all three retrievers enabled:

| | Before | After |
|---|---|---|
| Recall | 0.568 | **0.682** |
| vs dense alone | −0.045 | **+0.068** |
| Zero-recall tasks | 8 | **6** |

The general lesson is not about BM25. It is that a fusion weight is not a matter of taste, and
"semantic should matter more than lexical" is a hypothesis rather than a setting. See
`docs/evaluation.md` for the full seven-configuration ablation.

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

Current graph: **1,845 nodes, 2,649 edges** (1,783 provisions, 47 cases, 15 acts; relations
`contains` 1783, `cross_references` 803, `cites_section` 43, `considers_act` 20).

Traversal is bidirectional bounded BFS — "what does this cite?" and "what cites this?" are both
legally meaningful.

### Seeding, and an honest measurement

Seeds originally came only from provisions or acts **named explicitly in the question**. The
retriever ablation (`eval/ablation.py`) showed the consequence: graph retrieval contributed
**exactly zero** to recall, because conversational questions name neither. `dense+graph` equalled
`dense` to three decimal places.

Two changes followed. Act-name matching now tries the short title, because node labels carry the
year (`Indian Contract Act, 1872`) and no question includes it. And `seed_from_chunks` adds
dense-anchored seeding — take the provisions semantic search already surfaced and traverse out from
those, which is the graph's actual useful role.

Seeding now fires (graph-alone MRR rose 0.011 → 0.023) but **recall contribution is still 0.000**.
Reported as measured rather than dressed up: graph expansion reaches provisions that
cross-reference what dense found, and on the evaluation task set those are not the expected
sections. It adds candidates without displacing anything, so it is neutral rather than harmful, at
roughly 190 ms.

The honest reading is that a 2,649-edge graph over one jurisdiction's statutes is not yet dense
enough for traversal to beat embeddings at finding the right provision. It would matter more with
the precedent graph populated — `ingestion/citations.py` extracts treatment edges, but all 45
citations in the current corpus resolve outside it.

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

`scripts/smoke_retrieval.py`, spot checks:

| Question | Top result |
|---|---|
| "compensation for loss caused by breach of contract" | **s.73 Contract Act @ 0.8182**, then s.75, s.74 |
| "is a penalty clause enforceable" | **s.74 @ 0.7300** |
| "agreement void for want of consideration" | **s.25 @ 0.7617** |
| "are electronic records legally recognised" | **IT Act s.4 @ 0.7680**, s.10A, s.3, BSA s.61 |

The last one is worth noting: before repealed provisions were purged, that query returned
irrelevant Contract Act sections at ~0.56. Corpus hygiene moved the needle more than any tuning.

### Systematic measurement

Spot checks show the system working; they cannot show where it fails. `eval/ablation.py` scores all
seven retriever configurations against 22 tasks with known expected sections:

| Configuration | Recall | MRR | Latency |
|---|---|---|---|
| dense | 0.614 | 0.567 | 1,222 ms |
| keyword | 0.386 | 0.187 | 2,149 ms |
| graph | 0.045 | 0.023 | 1,357 ms |
| dense + keyword | 0.682 | 0.572 | 2,280 ms |
| dense + graph | 0.614 | 0.567 | 1,411 ms |
| keyword + graph | 0.386 | 0.187 | 2,555 ms |
| **all three** | **0.682** | **0.572** | 2,485 ms |

Two things this makes visible that spot checks could not:

- **Dense carries the system.** It uniquely retrieves 7 tasks that neither other retriever finds;
  BM25 uniquely retrieves 1; the graph uniquely retrieves none.
- **Recall is 0.682, not 1.0.** Six tasks still retrieve none of their expected sections. Those are
  the conversationally-phrased ones, and they are why the query rewrite exists — see §1, where
  rewriting takes recall on the affected subset from 0.500 to 1.000.

---

## 9. RAG on vs off

`POST /api/research/compare`, or the **RAG on / off** page.

The headline example is a genuine reproducible failure. Asked *"What does Section 73 of the Indian
Contract Act deal with?"* with no retrieval, Code Llama answers that it concerns **"the sale of
goods"**. It is compensation for breach of contract. With retrieval the model is grounded in the
official text and cites it.

That failure was not staged — it surfaced during a smoke test of the LLM path, before any RAG
existed.

### Measured across the whole task set

One anecdote is not a result. `eval/rag_ablation.py` runs all 27 tasks in both conditions with the
same model and prompt shape, scoring every section the model names by looking it up in the corpus —
**verified** if that Act contains it, **fabricated** if it does not. That is a lookup rather than a
judgement about phrasing.

| Metric | No RAG | With RAG |
|---|---|---|
| **Accuracy (answerable)** | 16.7% | **54.2%** |
| Fabrication rate | 4.3% | 6.0% |
| Fabricated on *unanswerable* tasks | 0 | 2 |
| Declined on unanswerable | 0/3 | 0/3 |
| Mean output tokens | 269 | 192 |

**Retrieval triples accuracy. It does not reduce fabrication — it relocates it.**

Without retrieval the model invents Contract Act sections 380, 123 and 111 while answering
answerable questions. With retrieval it gets those right, then fabricates on the deliberately
unanswerable ones instead — attributing s.498A to the Indian Contract Act three times when handed
context that cannot answer the question.

Neither condition declined on a single unanswerable task. So grounding fixes *what a model knows*,
not *whether it knows when to stop*. Those are separate problems, which is why refusal discipline
lives in the guardrail layer and in model choice (`qwen2.5:3b` declines 3/3; `gemma2:2b` declines
0/3) rather than being treated as something RAG delivers for free.

Grounded answers are also **shorter** — 192 tokens against 269 — and correspondingly faster. Having
evidence to point at removes the need to hedge at length.

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
RETRIEVAL_KEYWORD_WEIGHT=0.15         # measured; see §5. Was 0.6, which hurt recall
RETRIEVAL_GRAPH_HOPS=2
CHUNK_TARGET_CHARS=1400
CHUNK_MAX_CHARS=2600
CHUNK_OVERLAP_CHARS=180
```

Note that agent context budgets are tighter than retrieval top-k (5 law chunks, 4 cases). Retrieval
casts a wider net; the agents get the best of it, because prompt tokens are the dominant cost —
generation is **86.2% of request wall-clock**, measured over 40 recorded runs.
