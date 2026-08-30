# Model Evaluation — LegalAId

Comparison of three locally hosted models on the same application, prompts, knowledge base and question set. Every number here is produced by `eval/report.py` from the recorded runs in `eval/results/`, so the report cannot drift from the measurements.

## Summary of findings

1. **`llama3.2:3b` wins on quality and speed simultaneously** — 55.6% accuracy at 11.6 s mean latency. Among these three there is no accuracy-versus-latency trade-off to manage; the weakest model was also the slowest.
2. **Model choice moved accuracy by 3.7%** (51.9% → 55.6%) on an identical application, prompt, corpus and retrieved context. The generation model is not an interchangeable component.
3. **Refusal discipline diverged far more sharply than accuracy.** On the three questions the corpus genuinely cannot answer, scores ranged from 0/3 to 3/3. `qwen2.5:3b` declined correctly every time; the weakest answered all three with fabricated confidence. For a legal tool this matters more than a few points of accuracy.
4. **Retrieval phrasing mattered more than model choice.** Rewriting the query into dense legal language lifted mean retrieval recall from 0.500 to 1.000 (+0.500) on the same corpus — a larger swing than any model substitution produced. Where retrieval recall was zero, **no model recovered.**
5. **The ablation found the hybrid retrieval actively harmful, and that was the most useful result in the project.** Fusing BM25 at its configured weight scored *worse* than disabling it (0.568 against 0.614 recall) because Reciprocal Rank Fusion's agreement bonus promoted spurious keyword matches over good semantic hits. Replacing the guessed weight with a measured one took the full stack to 0.682 — now genuinely better than dense alone.
6. **Retrieval triples accuracy but does not reduce fabrication — it relocates it.** Accuracy on answerable tasks went 16.7% → 54.2% with retrieval, while the fabrication rate barely moved (4.3% → 6.0%). Without retrieval the model invents sections on answerable questions; with retrieval it gets those right and then fabricates on the *unanswerable* ones instead. Grounding fixes what a model knows, not whether it knows when to stop.
7. **86% of wall-clock time is model generation**, so orchestration overhead is not what to optimise. The entire input guardrail stack costs about 1.5% of a request — cheap insurance.
8. **The system still cannot answer codebase questions** — 0 of 61 relevant source files retrieved, because there is no code in the corpus. What changed is that it no longer fails *confidently*: those questions score 0.52–0.59 against the corpus, below the 0.55 guardrail floor, so they are now declined with a reason instead of answered with statutory text.

## How to reproduce

```bash
# model comparison
python scripts/verify_ground_truth.py   # expected sections still resolve
python eval/runner.py                   # 3 models x 27 tasks
python eval/resources.py                # per-model memory / VRAM / load time

# retrieval
python eval/ablation.py                 # 7 retriever configurations
python eval/weight_sweep.py             # fusion weight, measured
python eval/rewrite_experiment.py       # query rewriting A/B

# pipeline behaviour
python eval/rag_ablation.py             # RAG on vs off
python eval/latency_breakdown.py        # where the time goes
python eval/codebase_probe.py           # Exercise 6

# guardrails and corpus (gateway must be running for the first)
python scripts/check_probe_suite.py     # adversarial suite scorecard
python scripts/analyse_corpus.py        # duplicates and citation graph
python scripts/corpus_stats.py          # reconciliation table

python eval/report.py                   # regenerate this document
```

Corpus at time of measurement: **1,783 provisions, 47 judgments, 2,575 chunks** across 15 Acts. Earlier results against the 263-provision corpus are archived in `eval/results/corpus_263/` and compared near the end of this document.

## Exercise 1 — Models compared

Three different model families were chosen at the light end of what is installed, so the comparison isolates architecture and training rather than parameter count:

| Model | Family | Size | Params |
|---|---|---|---|
| `gemma2:2b` | Google | 1.6 GB | 2B |
| `qwen2.5:3b-instruct` | Alibaba | 1.9 GB | 3B |
| `llama3.2:3b` | Meta | 2.0 GB | 3B |
| `codellama:7b-instruct` | Meta / Code Llama | 3.8 GB | 7B |

**Held constant across all models:**

| Held constant | Detail |
|---|---|
| Application | LegalAId Legal Research Agent, unchanged |
| Prompt | `prompts/legal_research/v1.yaml`, same version and temperature |
| Questions | the same 27 tasks, in the same order |
| Knowledge base | 1,783 provisions, 47 judgments, 2,575 vectors, 15 Acts |
| Retrieved context | computed **once per question** and replayed byte-identical to every model |
| Decoding | schema-constrained JSON via Ollama `format`, `num_ctx` 3072 |

Replaying one retrieval to all models is deliberate. Retrieval depends only on the embedding model and the corpus, so re-running it per model would add noise to a comparison that is supposed to isolate the generation model.

## Exercise 2 — Evaluation dataset

27 tasks covering what a practitioner actually asks:

| Category | Tasks | What it tests |
|---|---|---|
| `statutory_lookup` | 6 | direct provision retrieval and accurate statement of its effect |
| `remedy_analysis` | 6 | mapping a client scenario to the governing provisions |
| `cross_act` | 4 | combining provisions across two or more statutes |
| `procedural` | 3 | civil procedure questions (jurisdiction, interest, notice) |
| `document_analysis` | 3 | reasoning from facts in an FIR or legal notice |
| `precedent` | 2 | finding relevant Supreme Court authority |
| `negative` | 3 | **declining** when the corpus cannot support an answer |

**24 answerable, 3 deliberately unanswerable.** The unanswerable tasks ask about the IPC, GST and the Motor Vehicles Act — none of which are in the corpus. They are load-bearing: without them a model that always answers confidently would score the same as one that knows when to stop, and hallucination rate could not be measured at all.

Ground-truth sections were verified to exist in the live knowledge base before the run (`scripts/list_sections.py`), so a model is never penalised for a gap in the corpus.

## Exercise 3 — Quantitative results

### Metric definitions

| Metric | Definition |
|---|---|
| **Correctness** | Answerable task: cites ≥1 expected section **and** every citation is grounded in supplied context. Unanswerable task: declines (`evidence_sufficient=false` or no citations). Citing the right section number while inventing its source is **not** counted correct — an unverifiable citation is worse than none in legal work. |
| **Relevance** | Citation precision on answerable tasks: `|cited ∩ expected| / |cited|`, averaged. Separates a focused answer from one that lists everything it saw. |
| **Retrieval recall** | `|expected ∩ retrieved| / |expected|`. Model-independent. |
| **Retrieval MRR** | `1 / rank` of the first expected section, 0 if absent. Rewards ranking the right provision first, not merely inside top-k. |
| **Hallucination rate** | Fraction of tasks citing ≥1 `chunk_id` that was never supplied. Fully automatic — the grounding audit knows exactly which chunk ids the agent received. |
| **Schema compliance** | Output parsed as JSON **and** passed the agent's structural validator. This app generates no code, so a literal unit-test-pass rate does not apply; this is the equivalent objective pass/fail gate the pipeline depends on. |
| **Latency** | Wall-clock ms for the agent call: render, HTTP, generation, parse, retries. Excludes the shared retrieval step. |
| **Tokens** | `input_tokens` / `output_tokens` as reported by Ollama. |
| **Throughput** | `total output tokens / total generation seconds` — the fair speed comparison, since a more verbose model is not necessarily slower per token. |
| **Resources** | Measured with one model resident at a time (Ollama evicts on switch): cold load, `ollama ps` resident size and CPU/GPU split, VRAM delta from `nvidia-smi`. |

### Retrieval quality (shared by all models)

Scored on the 22 answerable tasks that name specific expected sections.

| Metric | Value |
|---|---|
| Mean recall | 0.682 |
| Mean MRR | 0.572 |
| Tasks with perfect recall | 63.6% |
| Tasks with zero recall | 27.3% |
| Mean retrieval latency | 2823 ms |

### Quality

| Model | Accuracy (all) | Accuracy (answerable) | Correct refusals | Relevance | Hallucination | Schema compliance |
|---|---|---|---|---|---|---|
| `llama3.2:3b` | **55.6%** | 54.2% | 2/3 | 40.3% | 0.0% | 100.0% |
| `gemma2:2b` | **51.9%** | 58.3% | 0/3 | 24.0% | 11.1% | 88.9% |
| `qwen2.5:3b` | **51.9%** | 45.8% | 3/3 | 32.6% | 0.0% | 100.0% |

### Performance

| Model | Mean latency | p50 | p95 | Mean out tokens | Mean in tokens | Throughput | Retries |
|---|---|---|---|---|---|---|---|
| `llama3.2:3b` | 11.6 s | 12.2 s | 17.5 s | 373 | 1331 | 32.2 tok/s | 0 |
| `gemma2:2b` | 18.3 s | 16.8 s | 22.8 s | 641 | 1494 | 35.0 tok/s | 0 |
| `qwen2.5:3b` | 15.1 s | 15.0 s | 23.0 s | 550 | 1461 | 36.4 tok/s | 0 |

### Resource consumption

Measured one model at a time. GPU is an RTX 3050 (6 GB); `num_ctx` 3072 throughout.

| Model | Cold load | Warm call | Resident | CPU/GPU split | VRAM delta |
|---|---|---|---|---|---|
| `gemma2:2b` | 0.9 s | 3.4 s | 1.9 GB | 100% GPU | 402 MB |
| `qwen2.5:3b` | 12.7 s | 0.3 s | 1.9 GB | 100% GPU | 1390 MB |
| `llama3.2:3b` | 4.9 s | 0.4 s | 2.4 GB | 100% GPU | 2411 MB |
| `codellama:7b` | 9.7 s | 0.6 s | 5.6 GB | 22%/78% CPU/GPU | 2167 MB |

### Accuracy by category

| Model | `statutory_lookup` | `remedy_analysis` | `cross_act` | `procedural` | `document_analysis` | `precedent` | `negative` |
|---|---|---|---|---|---|---|---|
| `llama3.2:3b` | 6/6 | 2/6 | 2/4 | 2/3 | 1/3 | 0/2 | 2/3 |
| `gemma2:2b` | 6/6 | 3/6 | 2/4 | 2/3 | 1/3 | 0/2 | 0/3 |
| `qwen2.5:3b` | 4/6 | 2/6 | 2/4 | 2/3 | 1/3 | 0/2 | 3/3 |

## Exercise 4 — Analysis

| Question | Answer | Evidence |
|---|---|---|
| Which model is most accurate? | `llama3.2:3b` | 55.6% overall |
| Fewest hallucinations? | `llama3.2:3b` | 0.0% |
| Best citation relevance? | `llama3.2:3b` | 40.3% |
| Lowest latency? | `llama3.2:3b` | 11.6 s mean |
| Highest throughput? | `qwen2.5:3b` | 36.4 tok/s |

**Is the most accurate model also the fastest?** Yes — the same model leads on both axes, so there is no trade-off to manage here.

**Cost of accuracy.** Accuracy against mean latency, ordered by accuracy:

| Model | Accuracy | Mean latency | Accuracy per second |
|---|---|---|---|
| `llama3.2:3b` | 55.6% | 11.6 s | 0.048 |
| `gemma2:2b` | 51.9% | 18.3 s | 0.028 |
| `qwen2.5:3b` | 51.9% | 15.1 s | 0.034 |

**Test-pass rate for generated code.** Not applicable: LegalAId generates legal analysis and draft documents, not executable code, so there is nothing to run tests against. Schema compliance is reported instead as the equivalent objective gate — it is pass/fail, automatically checked, and the pipeline genuinely fails when it fails.

## Exercise 5 — RAG pipeline analysis

For each case below the chain is `QUESTION → RETRIEVED CONTEXT → LLM RESPONSE`, taken from the recorded runs rather than reconstructed.

### Where retrieval succeeded

**L01 — What does Section 73 of the Indian Contract Act provide for?**

- retrieval: recall `1.00`, MRR `1.00`, top similarity `0.8087`
  - `Section 73` Indian Contract Act, 1872 — sim `0.8087`, via `dense,keyword,graph`
  - `Section 73` Indian Contract Act, 1872 — sim `0.7951`, via `dense,keyword,graph`
  - `Section 73` Indian Contract Act, 1872 — sim `0.7915`, via `dense,keyword,graph`
- `llama3.2:3b`: correct, cited `indiancontract:73`
- `gemma2:2b`: correct, cited `indiancontract:73`
- `qwen2.5:3b`: correct, cited `indiancontract:73`

**L02 — Under the Indian Contract Act, when is an agreement without consideration still valid?**

- retrieval: recall `1.00`, MRR `1.00`, top similarity `0.8515`
  - `Section 25` Indian Contract Act, 1872 — sim `0.8515`, via `dense,keyword`
  - `Section 10` Indian Contract Act, 1872 — sim `0.8344`, via `dense,keyword,graph`
  - `Section 185` Indian Contract Act, 1872 — sim `0.8337`, via `dense,keyword`
- `llama3.2:3b`: correct, cited `indiancontract:25`
- `gemma2:2b`: correct, cited `indiancontract:10, indiancontract:185, indiancontract:25`
- `qwen2.5:3b`: correct, cited `indiancontract:25`

**L03 — What are the essential requirements for an agreement to be a contract?**

- retrieval: recall `1.00`, MRR `1.00`, top similarity `0.7498`
  - `Section 10` Indian Contract Act, 1872 — sim `0.7498`, via `dense,graph`
  - `Section 4` Sale of Goods Act, 1930 — sim `0.7018`, via `dense,keyword,graph`
  - `Section 7` Indian Contract Act, 1872 — sim `0.6966`, via `dense,graph`
- `llama3.2:3b`: correct, cited `indiancontract:10`
- `gemma2:2b`: correct, cited `indiancontract:10, indiancontract:25, indiancontract:7, saleofgoodsact:4`
- `qwen2.5:3b`: incorrect, cited `indiancontract:indian contract act, 1872 -  10, indiancontract:indian contract act, 1872 -  7`

### Where retrieval missed important information

6 of the section-specific tasks retrieved **none** of their expected provisions. This is the most instructive result in the whole evaluation:

**R01 — My client supplied goods under a written agreement and the buyer refused payment after delivery. Which provisions support a claim for compensation?**

- expected but missing: `indiancontract:73`
- what came back instead: `Section 44` (`0.739`), `Section 38` (`0.738`), `Section 56` (`0.734`)
- `llama3.2:3b`: incorrect — cited `saleofgoodsact:43, saleofgoodsact:44, saleofgoodsact:56, saleofgoodsact:57`
- `gemma2:2b`: incorrect — cited `saleofgoodsact:44, saleofgoodsact:56, saleofgoodsact:57`
- `qwen2.5:3b`: incorrect — cited `saleofgoodsact:44, saleofgoodsact:56, saleofgoodsact:57`

**R06 — The buyer told us in advance they would not perform. What are our options?**

- expected but missing: `indiancontract:39`
- what came back instead: `Section 44` (`0.607`), `Section 56` (`0.606`), `Section 43` (`0.604`)
- `llama3.2:3b`: incorrect — cited `saleofgoodsact:43, saleofgoodsact:44, saleofgoodsact:56, saleofgoodsact:57`
- `gemma2:2b`: incorrect — cited `saleofgoodsact:chk_43, saleofgoodsact:chk_44, saleofgoodsact:chk_56, saleofgoodsact:chk_57`
- `qwen2.5:3b`: incorrect — cited `saleofgoodsact:sale of goods act, 1930 -  43, saleofgoodsact:sale of goods act, 1930 -  44, saleofgoodsact:sale of goods act, 1930 -  56, saleofgoodsact:sale of goods act, 1930 -  57`

**X02 — A buyer took delivery of goods intending never to pay. Is that only a contract breach or also a criminal offence?**

- expected but missing: `bharatiyanyaya:318, indiancontract:73`
- what came back instead: `Section 38` (`0.775`), `Section 56` (`0.749`), `Section 44` (`0.748`)
- `llama3.2:3b`: incorrect — cited `saleofgoodsact:38, saleofgoodsact:44, saleofgoodsact:55, saleofgoodsact:60`
- `gemma2:2b`: incorrect — cited `saleofgoodsact:38, saleofgoodsact:44, saleofgoodsact:55, saleofgoodsact:56`
- `qwen2.5:3b`: incorrect — cited `saleofgoodsact:38, saleofgoodsact:44, saleofgoodsact:55, saleofgoodsact:56`

**X03 — We handed over machinery for repair and it was damaged in the workshop. What duty did they owe and are they liable?**

- expected but missing: `indiancontract:151, indiancontract:148`
- what came back instead: `Section 73` (`0.659`), `Section 150` (`0.650`), `Section 225` (`0.648`)
- `llama3.2:3b`: incorrect — cited `indiancontract:161, indiancontract:70, indiancontract:73`
- `gemma2:2b`: incorrect — cited `indiancontract:150, indiancontract:161, indiancontract:225, indiancontract:73`
- `qwen2.5:3b`: incorrect — cited `indiancontract:161, indiancontract:73`

**P01 — In which court should we institute a suit for breach of a supply contract?**

- expected but missing: `codeofcivilpro:20, codeofcivilpro:15`
- what came back instead: `Section 222` (`0.677`), `Section 19` (`0.676`), `Section 58` (`0.674`)
- `llama3.2:3b`: incorrect — cited `codeofcivilpro:17, codeofcivilpro:19, saleofgoodsact:58, saleofgoodsact:60`
- `gemma2:2b`: incorrect — cited `codeofcivilpro:19, saleofgoodsact:58, saleofgoodsact:60`
- `qwen2.5:3b`: incorrect — cited `codeofcivilpro:19 | code of civil procedure, 1908, saleofgoodsact:58 | sale of goods act, 1930, saleofgoodsact:60 | sale of goods act, 1930`

**D01 — An FIR alleges our client induced delivery of goods worth Rs 18,40,000 and then issued a cheque that bounced. Which provisions are in play and what is the exposure?**

- expected but missing: `bharatiyanyaya:318, bharatiyanyaya:316`
- what came back instead: `FIR-0412-2024.txt` (`0.785`), `Section 72` (`0.771`), `judgment_extract` (`0.732`)
- `llama3.2:3b`: incorrect — cited `bharatiyasaksh:13, indiancontract:222, indiancontract:72, indiancontract:73`
- `gemma2:2b`: incorrect — cited `indiancontract:222, indiancontract:223, indiancontract:72`
- `qwen2.5:3b`: incorrect — cited `indiancontract:222, indiancontract:72`

**Why this happened, and it is not a model failure.** These questions are phrased the way a client speaks — *"a legal notice claims we breached..."* — so the embedding spends its capacity on the narrative wrapper rather than the legal concept. In the full application the intake router rewrites the question into dense legal language **before** embedding, precisely to prevent this. The evaluation harness calls retrieval directly with the raw question, which isolates raw retrieval and therefore exposes what the rewrite is buying.

#### Controlled test: does rewriting actually fix it?

The explanation above is a testable claim, so it was tested. Same corpus, same retriever, same `top_k` — only the query text changes. Rewrites were written in the router's style (legal concepts, no narrative) **without** reference to the expected sections, so the answer is not smuggled in.

| Task | Raw recall | Raw MRR | Rewritten recall | Rewritten MRR | Result |
|---|---|---|---|---|---|
| `D01` | 0.00 | 0.00 | **1.00** | 1.00 | REWRITE FIXES IT |
| `D02` | 1.00 | 1.00 | **1.00** | 1.00 | no change |
| `D03` | 1.00 | 0.17 | **1.00** | 1.00 | better ranking |
| `R05` | 1.00 | 0.09 | **1.00** | 1.00 | better ranking |
| `P01` | 0.00 | 0.00 | **1.00** | 1.00 | REWRITE FIXES IT |
| `X03` | 0.00 | 0.00 | **1.00** | 1.00 | REWRITE FIXES IT |

**Mean recall went from 0.500 to 1.000 — an absolute gain of +0.500, with 3 of 6 tasks fixed outright.** The one task that already had full recall still improved in ranking (MRR 0.25 → 1.00), i.e. the right provision moved from fourth place to first.

This is the clearest quantitative result in the evaluation. **The same underlying question, against the same corpus and the same model, succeeds or fails entirely on how the query was phrased before embedding.** Retrieval is not a preprocessing step that either runs or does not — its quality sets a ceiling that no downstream model can exceed.

Worked example:

- **Raw:** *"An FIR alleges our client induced delivery of goods worth Rs 18,40,000 and then issued a cheque that bounced. Which provisions are in play and what is the exposure?"*
  - retrieved: `Section 72 (Indian Contract Act, 1872)`, `judgment_extract (National Insurance Company Ltd. v.`, `Section 13 (Bharatiya Sakshya Adhiniyam, 2023)` → recall 0.00
- **Rewritten:** *"Cheating and criminal breach of trust by dishonestly inducing delivery of goods without intention to pay; dishonour of cheque; entrustment and misappropriation of property."*
  - retrieved: `Section 316 (Bharatiya Nyaya Sanhita, 2023)`, `Section 316 (Bharatiya Nyaya Sanhita, 2023)`, `Section 318 (Bharatiya Nyaya Sanhita, 2023)` → recall 1.00

**The causal chain is visible.** Where retrieval recall was 0, no model recovered — every one of them either cited the wrong provision or declined. No amount of model quality compensates for a provision that was never placed in the context window. That is the concrete demonstration that `RETRIEVAL QUALITY → CONTEXT QUALITY → RESPONSE QUALITY` is a chain, not a checkbox.

### Hallucination despite retrieved context

**`gemma2:2b`** — 3 task(s):

- `R01` cited chunk ids that were never supplied: `chk_44, chk_56`
- `R06` cited chunk ids that were never supplied: `chk_43, chk_44`
- `X02` cited chunk ids that were never supplied: `chk_55`

### Behaviour on unanswerable questions

| Model | Correctly declined | Notes |
|---|---|---|
| `llama3.2:3b` | 2/3 | N01✓ N02✓ N03✗ |
| `gemma2:2b` | 0/3 | N01✗ N02✗ N03✗ |
| `qwen2.5:3b` | 3/3 | N01✓ N02✓ N03✓ |

### Retriever ablation — what each retriever actually contributes

The application claims hybrid retrieval: dense embeddings, BM25 and a symbolic graph traversal fused by reciprocal rank. That is architecture, not evidence. Every combination was run over the same questions.

| Configuration | Recall | MRR | Precision | Latency | Zero-recall tasks |
|---|---|---|---|---|---|
| `dense` | 0.614 | 0.567 | 0.077 | 1222 ms | 7 |
| `keyword` | 0.386 | 0.187 | 0.043 | 2149 ms | 13 |
| `graph` | 0.045 | 0.023 | 0.006 | 1357 ms | 21 |
| `dense+keyword` | 0.682 | 0.572 | 0.088 | 2280 ms | 6 |
| `dense+graph` | 0.614 | 0.567 | 0.077 | 1411 ms | 7 |
| `keyword+graph` | 0.386 | 0.187 | 0.043 | 2555 ms | 13 |
| `all three` | 0.682 | 0.572 | 0.088 | 2485 ms | 6 |

**Hybrid retrieval beats dense alone by +0.068 recall.** Dense alone uniquely retrieves 7 tasks that neither other retriever finds; BM25 uniquely retrieves 1.

#### This did not start out true

The first run of this ablation found the full stack retrieving **worse** than dense alone, which is the most useful thing the experiment produced.

| | Before | After |
|---|---|---|
| Full stack recall | 0.568 | **0.682** |
| vs dense alone | -0.045 | **+0.068** |
| Zero-recall tasks | 8 | **6** |

The cause was `retrieval_keyword_weight`, set to 0.6 on the reasoning that semantic match should dominate. Reciprocal Rank Fusion adds both retrievers' contributions, so a document dense-ranked 20th but keyword-ranked 1st scores `1/80 + 0.6/61 = 0.02234` against `1/65 = 0.01538` for one dense-ranked 5th. That is exactly what RRF is designed to do, and it is right when both retrievers are strong. On this corpus BM25 without an explicit section number in the question matches generic legal vocabulary, and those spurious agreements promoted noise over good semantic hits.

#### The weight, measured rather than reasoned

| `keyword_weight` | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.45 | 0.60 | 0.80 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|---|
| Recall | 0.614 | 0.659 | 0.636 | 0.682 | 0.659 | 0.614 | 0.568 | 0.568 | 0.568 | 0.568 |
| MRR | 0.567 | 0.573 | 0.567 | 0.572 | 0.572 | 0.566 | 0.522 | 0.522 | 0.522 | 0.522 |

The curve peaks at **0.15** and is flat from 0.45 upward. BM25 earns its place — worth +0.068 recall over dense alone — but only at roughly a quarter of the weight originally configured.

#### The graph retriever contributes nothing, and that is the honest result

`dense+graph` recall (0.614) equals `dense` (0.614) exactly. Graph alone reaches 0.045 recall and returns almost nothing (0.3 distinct acts per query).

The cause is structural rather than a tuning problem: `find_seeds` only fires when a question names a section or an act explicitly, and conversational questions name neither. Dense-anchored seeding was added — take the provisions semantic search already surfaced and traverse out from those — and seeding now fires (graph-alone MRR rose from 0.011 to 0.023), but **recall contribution is still 0.000**. Graph expansion reaches provisions that cross-reference what dense found, and on this task set those are not the expected sections. It adds candidates without displacing anything, so it is neutral rather than harmful, at roughly 190 ms.

### RAG on versus off — the hallucination control group

Hallucination was previously measured only *with* retrieval, which is a measurement without a control. Identical questions, identical model (`llama3.2:3b`), identical prompt shape; the only difference is whether retrieved statutory text is in the context.

Scoring cannot use the grounding audit here, because the no-RAG condition has no supplied context to check against. Instead every section the model names is looked up in the corpus: **verified** if that act really contains it, **fabricated** if it does not. That is not a judgement about phrasing, it is a lookup that fails.

| Metric | No RAG | With RAG |
|---|---|---|
| Sections named | 69 | 67 |
| Verified in corpus | 55 | 53 |
| **Fabricated** | 3 | 4 |
| **Fabrication rate** | 4.3% | 6.0% |
| Accuracy (answerable) | 16.7% | 54.2% |
| Declined on unanswerable | 0 | 0 |
| Fabricated on unanswerable | 0 | 2 |
| Mean latency | 7237 ms | 6198 ms |

**Retrieval changes the fabrication rate by -1.6%.** On this task set the difference is smaller than expected, which is reported as measured.

### Where the time goes

Aggregated from 40 recorded pipeline runs. Mean total 70.9 s, median 70.1 s.

| Stage | Share | Mean | Median | p95 |
|---|---|---|---|---|
| Similar Case Agent | 39.7% | 16.43 s | 17.23 s | 28.36 s |
| Document understanding | 17.6% | 12.49 s | 12.37 s | 13.14 s |
| Legal Research Agent | 17.4% | 12.36 s | 11.90 s | 18.00 s |
| Synthesis Agent — writing the answer | 17.1% | 12.12 s | 12.28 s | 14.68 s |
| Intake classification | 6.8% | 4.82 s | 4.87 s | 5.21 s |
| Verification Agent | 6.6% | 4.12 s | 3.96 s | 7.05 s |
| Drafting Agent | 5.3% | 3.27 s | 0.00 s | 15.86 s |
| Context construction | 3.2% | 2.25 s | 2.21 s | 2.73 s |
| Input guardrails | 1.5% | 0.93 s | 0.91 s | 1.09 s |
| Citation validation | 0.0% | 0.00 s | 0.00 s | 0.00 s |
| User input | 0.0% | 0.00 s | 0.00 s | 0.01 s |
| Final response | 0.0% | 0.00 s | 0.00 s | 0.01 s |

**86.2% of wall-clock time is inside model generation.** Orchestration overhead is therefore not the thing to optimise; a faster model or fewer agent calls is.

Mean throughput: 39.8 tokens/second.

### Guardrail effectiveness

29 probe cases: 19 adversarial and 10 legitimate. The legitimate cases matter more — several deliberately contain denylist trigger words, because a legal tool that refuses "draft a written statement in reply to the notice" is worse than one with no guardrails at all.

| | Adversarial caught | False positives | Accuracy | Mean cost |
|---|---|---|---|---|
| **Full stack** | 19/19 (100.0%) | 0/10 (0.0%) | 100.0% | 589 ms |
| Deterministic only | 15/19 (79.0%) | 0/10 (0.0%) | 86.2% | 0 ms |

**The embedding layer is worth +21.1% of catch rate**, and the cases it alone catches carry no keyword trigger whatsoever — capital of France, sorting a list, pizza toppings, yesterday's cricket. That is the evidence it generalises rather than pattern-matching a wordlist.

Refusals by layer: `{'empty': 2, 'too_long': 1, 'injection': 6, 'denylist': 6, 'corpus_distance': 4}`

### Corpus quality

1783 provisions checked for exact, near (SimHash, 64-bit, Hamming ≤ 3) and structural duplication.

| Metric | Value |
|---|---|
| Exact duplicates | 2 |
| Near duplicates | 1 |
| Structural duplicates | 0 |
| Duplicate rate | 0.17% |
| Chunk duplicate rate | 0.00% |

Duplicate detection earned its place immediately: it flagged 13 Arbitration Act provisions as identical to one another because India Code carries the Arbitration Council sections (43A–43M) with headings and **no body text**. Indexed, they were retrievable and citable while saying nothing — worse than absent, because the system would offer a lawyer a provision reference carrying no law. Filtering them took the duplicate rate from 0.84% to 0.17%.

The remaining exact duplicates are genuine statutory parallels: Arbitration Act s.46 and s.55 are word-for-word identical across the New York and Geneva Convention chapters. Not an error, and not removed.

Citation extraction found 45 citations and 38 section references across 47 judgments, with treatment classified as `{'cited': 15, 'distinguished': 7, 'followed': 2, 'referred': 14, 'relied_on': 5, 'reversed': 2}`. All 45 resolve **outside** the corpus, which is honest rather than broken: the judgments are a 2023 slice citing earlier authority the corpus does not hold. Unresolved citations are reported, not dropped.

## Exercise 6 — Repository / codebase understanding

**Question: can the existing LLM + RAG system answer questions that require understanding multiple files or modules? Measured answer: no, for a structural reason rather than a tuning one.**

Five genuine multi-file questions about this repository were put to the retriever. Ground truth was established independently by static analysis (AST parsing), giving **61 relevant files** across the five questions and **216 internal import edges** across 45 modules.

| Question | Kind | Ground-truth files | Found in retrieved context | Top similarity |
|---|---|---|---|---|
| Q1 — Which files are involved in agent context isolation? | `multi_file_location` | 10 | **0** | 0.5269 |
| Q2 — Which modules import the retrieval service? | `dependency_graph` | 19 | **0** | 0.5296 |
| Q3 — What happens after a user submits a research request? | `control_flow` | 8 | **0** | 0.5899 |
| Q4 — Which components would be affected if the chunking strategy changed? | `impact_analysis` | 23 | **0** | 0.5231 |
| Q5 — Which tests cover the hybrid retrieval fusion logic? | `test_mapping` | 1 | **0** | 0.5443 |

**0 of 61 relevant files were retrieved.** Every query returned Indian statute and case law, because that is the only thing in the knowledge base.

**Why it fails, concretely:**

1. **No code is indexed.** The corpus is 263 statutory provisions and 47 judgments. There is no source file in any collection, so no amount of retrieval tuning can surface `orchestrator.py`.
2. **The graph retriever models the wrong relationships.** It has real edges — 216 import edges exist in the code — but the graph it built has `contains`, `cross_references`, `cites_section` and `considers_act`. There is no `imports`, `calls`, `defines` or `tested_by`. Answering *"which modules import the retrieval service?"* needs an import edge, and the schema has no way to express one.
3. **Chunking is structured for statute.** The three strategies split on act/section/subsection, judgment headnote, and document page. A function spanning 40 lines with its decorator and docstring has no corresponding unit.
4. **Worst of all, it fails confidently.** Top similarities were 0.52–0.59, comfortably above the 0.25 `min_similarity` floor. The system therefore returns plausible-looking statutory text for a question about Python imports, with no signal that it has nothing relevant. Cosine similarity is computed within whatever corpus exists; it cannot report that the corpus is the wrong one.

**What would be required.** Not a better model — a second retrieval domain: index source files with code-aware chunking (function and class granularity), add `imports` / `calls` / `defines` / `tested_by` edge types to the graph, and route code questions to it. That is precisely the capability Sourcegraph provides, which is the point of the exercise.

**One honest caveat.** The existing graph retriever is already a symbolic, non-vector traversal over typed edges, so the architecture generalises to code — the gap is the corpus and the edge vocabulary, not the retrieval design.

**What changed as a result.** The confident-failure half of this finding is now handled. The corpus-distance guardrail refuses questions whose nearest corpus neighbour falls below 0.55, and the code questions here score 0.37–0.49. So the system no longer answers them with statutory text; it declines and says why. The retrieval gap is unchanged — it still could not answer a codebase question if asked — but it no longer pretends otherwise, which was the more dangerous half.

## The corpus itself is a variable

Every number above comes from the 1,783-provision corpus. An earlier run of the identical harness against the 263-provision corpus is archived in `eval/results/corpus_263/`, and comparing them is instructive:

| Model | Accuracy @ 263 provisions | Accuracy @ 1,783 | Change |
|---|---|---|---|
| `llama3.2:3b` | 66.7% | 55.6% | -11.1 pp |
| `gemma2:2b` | 48.1% | 51.9% | +3.8 pp |
| `qwen2.5:3b` | 59.3% | 51.9% | -7.4 pp |

**Accuracy fell for the two strongest models as the corpus grew almost sevenfold.** That is not a regression in the system; it is the task becoming harder. A larger corpus means more plausible-but-wrong provisions competing for the same top-k, so retrieval has to discriminate among 1,783 candidates rather than 263. The relative ranking of the models is unchanged, and the refusal-discipline finding is unchanged.

The practical lesson is that a model comparison is only meaningful against a stated corpus. Quoting an accuracy figure without saying what it was measured over is close to meaningless, which is why every table here names the corpus size.

## Threats to validity

- **One agent, not the full pipeline.** The seven-agent pipeline takes 3–7 minutes per question; 27 questions × 3 models would be roughly 9 hours. The Legal Research Agent was used instead. It exercises retrieval, prompt rendering, constrained decoding and the grounding audit, but multi-agent effects (verification retries, synthesis quality) are not captured.
- **Raw questions, not router-rewritten.** Retrieval was called directly, so the reported recall is a floor rather than what the deployed app achieves. This was useful — it is what exposed the missed-retrieval cases in Exercise 5 — but it understates end-to-end quality.
- **Single run per pair.** Temperature is low (0.1) but non-zero, so figures carry run-to-run variance. No repeats were averaged.
- **Correctness is section-level.** A model citing the right provision is scored correct even if its prose summary is thin. Legal soundness of the wording is not assessed.
- **Small corpus for precedent.** All 47 judgments are from 2023 and skew commercial, so the two precedent tasks test a thin slice. This is also why all 45 extracted citations resolve outside the corpus.
- **The ablation scores retrieval, not answers.** Exercise 5's configuration table measures recall and MRR against known expected sections. It does not establish that better retrieval produced a better answer in every case, only that the evidence reached the context window.
- **E6 uses corpus-existence as its fabrication test.** A section is counted fabricated when the act it is attributed to does not contain it. That is objective and automatic, but it cannot catch a citation to a real section that says something other than what the model claimed.
- **The latency breakdown aggregates whatever traces exist.** It is measured from real recorded runs rather than a controlled benchmark, so the stage mix reflects the questions that happened to be asked.

### What this evaluation does not establish

No metric here shows that an answer is legally **correct**. Every measurement is about grounding, scope and retrieval: whether a cited provision exists, was supplied, and is the one the question needed. Whether the reasoning built on it would persuade a court is outside what an automated harness can assess, and it is the reason the disclaimer and lawyer review exist rather than being decoration.

