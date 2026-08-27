# Model Evaluation — LegalAId

Comparison of three locally hosted models on the same application, prompts, knowledge base and question set. Every number here is produced by `eval/report.py` from the recorded runs in `eval/results/`, so the report cannot drift from the measurements.

## Summary of findings

1. **`llama3.2:3b` wins on quality and speed simultaneously** — 66.7% accuracy at 7.3 s mean latency. Among these three there is no accuracy-versus-latency trade-off to manage; the weakest model was also the slowest.
2. **Model choice moved accuracy by 18.5%** (48.1% → 66.7%) on an identical application, prompt, corpus and retrieved context. The generation model is not an interchangeable component.
3. **Refusal discipline diverged far more sharply than accuracy.** On the three questions the corpus genuinely cannot answer, scores ranged from 0/3 to 3/3. `qwen2.5:3b` declined correctly every time; the weakest answered all three with fabricated confidence. For a legal tool this matters more than a few points of accuracy.
4. **Retrieval phrasing mattered more than model choice.** Rewriting the query into dense legal language lifted mean retrieval recall from 0.167 to 1.000 (+0.833) on the same corpus — a larger swing than any model substitution produced. Where retrieval recall was zero, **no model recovered.**
5. **The system cannot answer codebase questions** — 0 of 55 relevant source files retrieved. Worse, it fails *confidently*: similarities of 0.52–0.57 sit above the relevance floor, so it returns statutory text for questions about Python imports.

## How to reproduce

```bash
python eval/runner.py            # all models, all 27 tasks
python eval/resources.py         # per-model memory / VRAM / load time
python eval/codebase_probe.py    # Exercise 6
python eval/report.py            # regenerate this document
```

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
| Knowledge base | 263 provisions, 47 judgments, 651 vectors |
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
| Mean recall | 0.659 |
| Mean MRR | 0.572 |
| Tasks with perfect recall | 59.1% |
| Tasks with zero recall | 27.3% |
| Mean retrieval latency | 1721 ms |

### Quality

| Model | Accuracy (all) | Accuracy (answerable) | Correct refusals | Relevance | Hallucination | Schema compliance |
|---|---|---|---|---|---|---|
| `llama3.2:3b` | **66.7%** | 66.7% | 2/3 | 40.6% | 0.0% | 100.0% |
| `qwen2.5:3b` | **59.3%** | 54.2% | 3/3 | 37.8% | 0.0% | 100.0% |
| `gemma2:2b` | **48.1%** | 54.2% | 0/3 | 30.6% | 11.1% | 88.9% |

### Performance

| Model | Mean latency | p50 | p95 | Mean out tokens | Mean in tokens | Throughput | Retries |
|---|---|---|---|---|---|---|---|
| `llama3.2:3b` | 7.3 s | 6.7 s | 11.4 s | 345 | 1304 | 47.4 tok/s | 0 |
| `qwen2.5:3b` | 9.7 s | 9.0 s | 16.1 s | 480 | 1433 | 49.3 tok/s | 0 |
| `gemma2:2b` | 10.3 s | 11.3 s | 14.5 s | 551 | 1466 | 53.3 tok/s | 0 |

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
| `llama3.2:3b` | 5/6 | 6/6 | 3/4 | 2/3 | 0/3 | 0/2 | 2/3 |
| `qwen2.5:3b` | 5/6 | 4/6 | 2/4 | 2/3 | 0/3 | 0/2 | 3/3 |
| `gemma2:2b` | 5/6 | 3/6 | 3/4 | 2/3 | 0/3 | 0/2 | 0/3 |

## Exercise 4 — Analysis

| Question | Answer | Evidence |
|---|---|---|
| Which model is most accurate? | `llama3.2:3b` | 66.7% overall |
| Fewest hallucinations? | `llama3.2:3b` | 0.0% |
| Best citation relevance? | `llama3.2:3b` | 40.6% |
| Lowest latency? | `llama3.2:3b` | 7.3 s mean |
| Highest throughput? | `gemma2:2b` | 53.3 tok/s |

**Is the most accurate model also the fastest?** Yes — the same model leads on both axes, so there is no trade-off to manage here.

**Cost of accuracy.** Accuracy against mean latency, ordered by accuracy:

| Model | Accuracy | Mean latency | Accuracy per second |
|---|---|---|---|
| `llama3.2:3b` | 66.7% | 7.3 s | 0.092 |
| `qwen2.5:3b` | 59.3% | 9.7 s | 0.061 |
| `gemma2:2b` | 48.1% | 10.3 s | 0.047 |

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
- `qwen2.5:3b`: correct, cited `indiancontract:73`
- `gemma2:2b`: correct, cited `indiancontract:73`

**L02 — Under the Indian Contract Act, when is an agreement without consideration still valid?**

- retrieval: recall `1.00`, MRR `1.00`, top similarity `0.8515`
  - `Section 25` Indian Contract Act, 1872 — sim `0.8515`, via `dense,keyword`
  - `Section 10` Indian Contract Act, 1872 — sim `0.8344`, via `dense,keyword`
  - `Section 185` Indian Contract Act, 1872 — sim `0.8337`, via `dense,keyword`
- `llama3.2:3b`: correct, cited `indiancontract:25`
- `qwen2.5:3b`: correct, cited `indiancontract:10, indiancontract:185, indiancontract:25`
- `gemma2:2b`: correct, cited `indiancontract:10, indiancontract:185, indiancontract:25`

**L04 — Are electronic records legally recognised in India, and under which provision?**

- retrieval: recall `1.00`, MRR `1.00`, top similarity `0.7785`
  - `Section 4` Information Technology Act, 2000 — sim `0.7785`, via `dense,keyword`
  - `Section 2` Information Technology Act, 2000 — sim `0.7647`, via `dense,keyword`
  - `Section 63` Bharatiya Sakshya Adhiniyam, 2023 — sim `0.7639`, via `dense,keyword`
- `llama3.2:3b`: correct, cited `informationtec:4`
- `qwen2.5:3b`: correct, cited `informationtec:2, informationtec:4`
- `gemma2:2b`: correct, cited `bharatiyasaksh:63, informationtec:2, informationtec:4`

### Where retrieval missed important information

6 of the section-specific tasks retrieved **none** of their expected provisions. This is the most instructive result in the whole evaluation:

**L03 — What are the essential requirements for an agreement to be a contract?**

- expected but missing: `indiancontract:10`
- what came back instead: `Section 25` (`0.692`), `Section 25` (`0.689`), `Section 23` (`0.685`)
- `llama3.2:3b`: incorrect — cited `indiancontract:25`
- `qwen2.5:3b`: incorrect — cited `indiancontract:23 | indian contract act, 1872, indiancontract:25 | indian contract act, 1872`
- `gemma2:2b`: incorrect — cited `indiancontract:2, indiancontract:23, indiancontract:25`

**X03 — We handed over machinery for repair and it was damaged in the workshop. What duty did they owe and are they liable?**

- expected but missing: `indiancontract:151, indiancontract:148`
- what came back instead: `Section 73` (`0.659`), `Section 150` (`0.650`), `Section 225` (`0.648`)
- `llama3.2:3b`: incorrect — cited `indiancontract:150, indiancontract:73`
- `qwen2.5:3b`: incorrect — cited `indiancontract:73`
- `gemma2:2b`: incorrect — cited `indiancontract:150, indiancontract:225, indiancontract:73`

**P01 — In which court should we institute a suit for breach of a supply contract?**

- expected but missing: `codeofcivilpro:20, codeofcivilpro:15`
- what came back instead: `Section 222` (`0.677`), `Section 17` (`0.669`), `Section 10` (`0.668`)
- `llama3.2:3b`: incorrect — cited `codeofcivilpro:10, codeofcivilpro:17, indiancontract:73`
- `qwen2.5:3b`: incorrect — cited `codeofcivilpro:17 | code of civil procedure, 1908, indiancontract:73 | indian contract act, 1872`
- `gemma2:2b`: incorrect — cited `codeofcivilpro:10, codeofcivilpro:17, indiancontract:222, indiancontract:73`

**D01 — An FIR alleges our client induced delivery of goods worth Rs 18,40,000 and then issued a cheque that bounced. Which provisions are in play and what is the exposure?**

- expected but missing: `bharatiyanyaya:318, bharatiyanyaya:316`
- what came back instead: `FIR-0412-2024.txt` (`0.785`), `Section 73` (`0.718`), `Section 74` (`0.718`)
- `llama3.2:3b`: incorrect — cited `indiancontract:220, indiancontract:50, indiancontract:73, indiancontract:74`
- `qwen2.5:3b`: incorrect — cited `indiancontract:50, indiancontract:73`
- `gemma2:2b`: incorrect — cited `indiancontract:220, indiancontract:50, indiancontract:73, indiancontract:74`

**D02 — A legal notice claims we breached a supply agreement and demands damages. What provisions govern the quantum they can claim?**

- expected but missing: `indiancontract:73, indiancontract:74`
- what came back instead: `Section 222` (`0.693`), `reasoning` (`0.670`), `judgment_extract` (`0.664`)
- `llama3.2:3b`: incorrect — cited `indiancontract:229`
- `qwen2.5:3b`: incorrect — cited `indiancontract:229 | indian contract act, 1872`
- `gemma2:2b`: incorrect — cited `indiancontract:145, indiancontract:229, indiancontract:24`

**D03 — The agreement was signed by someone described as an agent. How do we test whether the principal is bound?**

- expected but missing: `indiancontract:182`
- what came back instead: `Section 228` (`0.742`), `Section 231` (`0.739`), `Section 232` (`0.728`)
- `llama3.2:3b`: incorrect — cited `indiancontract:227, indiancontract:228, indiancontract:230`
- `qwen2.5:3b`: incorrect — cited `indiancontract:227 | indian contract act, 1872`
- `gemma2:2b`: incorrect — cited `indiancontract:227, indiancontract:228, indiancontract:230, indiancontract:231`

**Why this happened, and it is not a model failure.** These questions are phrased the way a client speaks — *"a legal notice claims we breached..."* — so the embedding spends its capacity on the narrative wrapper rather than the legal concept. In the full application the intake router rewrites the question into dense legal language **before** embedding, precisely to prevent this. The evaluation harness calls retrieval directly with the raw question, which isolates raw retrieval and therefore exposes what the rewrite is buying.

#### Controlled test: does rewriting actually fix it?

The explanation above is a testable claim, so it was tested. Same corpus, same retriever, same `top_k` — only the query text changes. Rewrites were written in the router's style (legal concepts, no narrative) **without** reference to the expected sections, so the answer is not smuggled in.

| Task | Raw recall | Raw MRR | Rewritten recall | Rewritten MRR | Result |
|---|---|---|---|---|---|
| `D01` | 0.00 | 0.00 | **1.00** | 1.00 | REWRITE FIXES IT |
| `D02` | 0.00 | 0.00 | **1.00** | 1.00 | REWRITE FIXES IT |
| `D03` | 0.00 | 0.00 | **1.00** | 1.00 | REWRITE FIXES IT |
| `R05` | 1.00 | 0.25 | **1.00** | 1.00 | better ranking |
| `P01` | 0.00 | 0.00 | **1.00** | 1.00 | REWRITE FIXES IT |
| `X03` | 0.00 | 0.00 | **1.00** | 1.00 | REWRITE FIXES IT |

**Mean recall went from 0.167 to 1.000 — an absolute gain of +0.833, with 5 of 6 tasks fixed outright.** The one task that already had full recall still improved in ranking (MRR 0.25 → 1.00), i.e. the right provision moved from fourth place to first.

This is the clearest quantitative result in the evaluation. **The same underlying question, against the same corpus and the same model, succeeds or fails entirely on how the query was phrased before embedding.** Retrieval is not a preprocessing step that either runs or does not — its quality sets a ceiling that no downstream model can exceed.

Worked example:

- **Raw:** *"An FIR alleges our client induced delivery of goods worth Rs 18,40,000 and then issued a cheque that bounced. Which provisions are in play and what is the exposure?"*
  - retrieved: `Section 73 (Indian Contract Act, 1872)`, `Section 74 (Indian Contract Act, 1872)`, `Section 220 (Indian Contract Act, 1872)` → recall 0.00
- **Rewritten:** *"Cheating and criminal breach of trust by dishonestly inducing delivery of goods without intention to pay; dishonour of cheque; entrustment and misappropriation of property."*
  - retrieved: `Section 316 (Bharatiya Nyaya Sanhita, 2023)`, `Section 316 (Bharatiya Nyaya Sanhita, 2023)`, `Section 318 (Bharatiya Nyaya Sanhita, 2023)` → recall 1.00

**The causal chain is visible.** Where retrieval recall was 0, no model recovered — every one of them either cited the wrong provision or declined. No amount of model quality compensates for a provision that was never placed in the context window. That is the concrete demonstration that `RETRIEVAL QUALITY → CONTEXT QUALITY → RESPONSE QUALITY` is a chain, not a checkbox.

### Hallucination despite retrieved context

**`gemma2:2b`** — 3 task(s):

- `R02` cited chunk ids that were never supplied: `chk_60, chk_74`
- `R06` cited chunk ids that were never supplied: `chk_39, chk_47`
- `N01` cited chunk ids that were never supplied: `null`

### Behaviour on unanswerable questions

| Model | Correctly declined | Notes |
|---|---|---|
| `llama3.2:3b` | 2/3 | N01✓ N02✓ N03✗ |
| `qwen2.5:3b` | 3/3 | N01✓ N02✓ N03✓ |
| `gemma2:2b` | 0/3 | N01✗ N02✗ N03✗ |

## Exercise 6 — Repository / codebase understanding

**Question: can the existing LLM + RAG system answer questions that require understanding multiple files or modules? Measured answer: no, for a structural reason rather than a tuning one.**

Five genuine multi-file questions about this repository were put to the retriever. Ground truth was established independently by static analysis (AST parsing), giving **55 relevant files** across the five questions and **197 internal import edges** across 41 modules.

| Question | Kind | Ground-truth files | Found in retrieved context | Top similarity |
|---|---|---|---|---|
| Q1 — Which files are involved in agent context isolation? | `multi_file_location` | 10 | **0** | 0.5269 |
| Q2 — Which modules import the retrieval service? | `dependency_graph` | 14 | **0** | 0.5156 |
| Q3 — What happens after a user submits a research request? | `control_flow` | 8 | **0** | 0.5677 |
| Q4 — Which components would be affected if the chunking strategy changed? | `impact_analysis` | 22 | **0** | 0.5197 |
| Q5 — Which tests cover the hybrid retrieval fusion logic? | `test_mapping` | 1 | **0** | 0.5409 |

**0 of 55 relevant files were retrieved.** Every query returned Indian statute and case law, because that is the only thing in the knowledge base.

**Why it fails, concretely:**

1. **No code is indexed.** The corpus is 263 statutory provisions and 47 judgments. There is no source file in any collection, so no amount of retrieval tuning can surface `orchestrator.py`.
2. **The graph retriever models the wrong relationships.** It has real edges — 197 import edges exist in the code — but the graph it built has `contains`, `cross_references`, `cites_section` and `considers_act`. There is no `imports`, `calls`, `defines` or `tested_by`. Answering *"which modules import the retrieval service?"* needs an import edge, and the schema has no way to express one.
3. **Chunking is structured for statute.** The three strategies split on act/section/subsection, judgment headnote, and document page. A function spanning 40 lines with its decorator and docstring has no corresponding unit.
4. **Worst of all, it fails confidently.** Top similarities were 0.52–0.57, comfortably above the 0.25 `min_similarity` floor. The system therefore returns plausible-looking statutory text for a question about Python imports, with no signal that it has nothing relevant. Cosine similarity is computed within whatever corpus exists; it cannot report that the corpus is the wrong one.

**What would be required.** Not a better model — a second retrieval domain: index source files with code-aware chunking (function and class granularity), add `imports` / `calls` / `defines` / `tested_by` edge types to the graph, and route code questions to it. That is precisely the capability Sourcegraph provides, which is the point of the exercise.

**One honest caveat.** The existing graph retriever is already a symbolic, non-vector traversal over typed edges, so the architecture generalises to code — the gap is the corpus and the edge vocabulary, not the retrieval design.

## Threats to validity

- **One agent, not the full pipeline.** The seven-agent pipeline takes 3–7 minutes per question; 27 questions × 3 models would be roughly 9 hours. The Legal Research Agent was used instead. It exercises retrieval, prompt rendering, constrained decoding and the grounding audit, but multi-agent effects (verification retries, synthesis quality) are not captured.
- **Raw questions, not router-rewritten.** Retrieval was called directly, so the reported recall is a floor rather than what the deployed app achieves. This was useful — it is what exposed the missed-retrieval cases in Exercise 5 — but it understates end-to-end quality.
- **Single run per pair.** Temperature is low (0.1) but non-zero, so figures carry run-to-run variance. No repeats were averaged.
- **Correctness is section-level.** A model citing the right provision is scored correct even if its prose summary is thin. Legal soundness of the wording is not assessed.
- **Small corpus for precedent.** All 47 judgments are from 2023 and skew commercial, so the two precedent tasks test a thin slice.

