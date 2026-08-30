# Demo walkthrough

A 10–12 minute script that shows every element the brief asks for, in an order that builds.

**Before you start**

```powershell
curl http://localhost:8000/api/system/health   # expect status ok, vectors > 2500
```

Health is the one endpoint that needs no API key. Everything else does; in local development the
Vite proxy injects the key from `data/.dev-api-key`, so the UI just works. If the UI shows 401s,
the backend has not been started yet — it writes that file on first run.

Warm the model first — the very first generation of a session pays a ~90 s load penalty that will
otherwise look like a hang:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" run llama3.2:3b "ok" --verbose
```

`llama3.2:3b` is the default because it measured best on accuracy *and* latency. Code Llama is
still installed and selectable if you want to show the comparison live.

Have a second browser tab on `http://localhost:8000/docs`.

**Timing note.** A full seven-agent run takes 60–90 seconds on this hardware, of which 86% is model
generation. Start it talking, then narrate the trace while it works rather than waiting in silence.

---

## 1. The lawyer's view (1 min)

Open **http://localhost:5173** → lands on **Home**.

Point out what a practising advocate sees: a single box that accepts anything, four common starting
points, recent research, and what the library covers — 1,783 provisions across 15 Acts plus 47
Supreme Court judgments, each labelled with its official source.

Note the sidebar split: **For you** versus **Developer / evaluation**. The observability lives in
the second group deliberately, so the practitioner view stays uncluttered.

---

## 2. Paste an FIR (1 min)

Go to **My documents** and paste this, then *Ingest pasted text*:

```
FIRST INFORMATION REPORT
Police Station: Sector 20, Gurugram        FIR No: 0412/2024
Date: 14/03/2024

Complainant: M/s Ravi Traders, through its proprietor Shri Ravi Kumar, Gurugram.
Accused: M/s Sunrise Distributors Pvt. Ltd., through its director Shri Anil Mehta.

Brief facts: Under a written supply agreement dated 02/01/2024, goods (industrial fasteners)
worth Rs. 18,40,000 were delivered to the accused on 20/02/2024 against invoice No. RT/2024/118.
The accused issued cheque No. 556213 dated 05/03/2024 for Rs. 18,40,000 drawn on ICICI Bank,
which was returned unpaid marked "funds insufficient" on 11/03/2024. Despite telephonic demands
and a written notice, the accused has neither paid nor returned the goods. The complainant alleges
the accused induced delivery with a dishonest intention of not paying.
```

It reports pages, chunks, vectors and the **detected document class** — `complaint`, inferred from
"FIRST INFORMATION REPORT". Open it to show extracted text per page and the chunks derived from it.

---

## 3. Ask a question (3–5 min of runtime)

**Ask LegalAId**:

> My client supplied goods under a written agreement and the buyer's cheque bounced. What
> provisions apply to claiming compensation, and are there similar Indian cases?

While it runs, the **live pipeline** streams each stage over SSE. Talk through it:

1. `USER_INPUT`
2. `INTAKE_CLASSIFICATION` — deciding what work is needed
3. `QUERY_REWRITE` — rewriting the text that will be embedded
4. `CONTEXT_CONSTRUCTION` — hybrid retrieval
5. `DOCUMENT_UNDERSTANDING` — because an FIR is present
6. `LEGAL_RESEARCH_AGENT` + `SIMILAR_CASE_AGENT` — concurrent
7. `REFINEMENT_LOOP` — if a citation fails its audit
8. `SYNTHESIS_AGENT` — writing the answer

---

## 4. The answer (2 min)

The result leads with prose, not a data dump: a direct answer, key points, **AI inference** labelled
separately from what the sources say, uncertainties, and next steps. Confidence is stated.

Below it, supporting authorities — provisions and judgments, each with *"View the exact text this
came from →"*.

**The point to make:** inference is visually separated from law. A reader can always tell what the
statute says from what the system concluded.

---

## 5. How it was produced (2 min)

Click **Show how this answer was produced**.

**The routing panel is the thing to dwell on.** It shows your original words next to the text that
was actually embedded:

> **You:** "…the buyer's cheque bounced. What provisions apply…?"
>
> **Embedded:** "Dishonour of cheque issued towards payment for goods supplied under a written
> supply agreement; cheating and criminal breach of trust by inducing delivery without intention to
> pay; recovery of price and compensation for breach."

Plus the document type it detected and which agents it selected.

Then expand an agent. Two columns: **Context received** and **Deliberately withheld, with reasons**
— e.g. *"case law is out of scope for the Legal Research Agent"*. That is context isolation being
shown rather than claimed.

Inside each agent: the system prompt, the user prompt (with the retrieved context inline), and the
raw model text.

---

## 6. Retrieval Inspector — prove hybrid search (2 min)

**Developer → Retrieval inspector.** Query: `Section 73 compensation for breach of contract`.

Run three times:

1. keyword **off**, graph **off** → dense only
2. keyword **on** → badges now show `dense,keyword`; note the ranking shift
3. graph **on** → `dense,keyword,graph`

Also on the page: the embedding model and dimensions, the query vector preview, candidate counts
per collection, and per-chunk "why this was retrieved" with matched legal concepts.

---

## 7. Chunk Explorer — provenance (1 min)

Click any retrieved chunk.

Full text, every metadata field, the embedding record, **the first 64 components of the actual
vector**, retrieval history, which agents used it, and provenance: act, section, official India Code
URL, and amendment footnotes. For a judgment: citation, court, bench, outcome, CC-BY-4.0 licence.

Follow the source link to indiacode.nic.in to show the text matches.

---

## 8. RAG on / off (2 min)

**Developer → RAG on / off**, default question:

> What does Section 73 of the Indian Contract Act deal with?

**Without retrieval** Code Llama answers that s.73 concerns *"the sale of goods."* It does not — it
is compensation for breach. **With retrieval** the model is grounded in the official text and cites
it.

Worth saying: that failure was not staged. It surfaced while smoke-testing the LLM path before any
retrieval existed.

---

## 9. Agents and System (1 min)

**Agents** — the seven-agent roster as a node graph showing which run concurrently, each with its
prompt version, temperature, token budget, context policy and measured latency. Expand a prompt to
show it is a versioned file, not a string in Python.

**System & models** — Ollama status, installed models with `usable_for_generation`, all three
retrievers' sizes, the prompt registry, and the capability matrix mapping features to Exercises 1–5.

Switch the model in the top bar to `qwen2.5:3b-instruct` to show generation is swappable at runtime.

---

## 10. Guardrails — what it refuses (2 min)

**Developer → Guardrails.** This is the segment worth the most, because it shows the system
declining rather than performing.

First the layer table: six layers, deterministic ones before the model, so an out-of-scope request
never costs a generation. Point at the similarity-floor panel — 0.55 was chosen from a measured
distribution, not picked.

Then press **Run suite**. 29 adversarial and legitimate probes go through the live stack:

- **19/19 adversarial caught, 0/10 false positives.**
- Filter to `legitimate` and read a few aloud: *"draft a written statement in reply to this legal
  notice"*, *"can the court ignore a defective notice under s.80 CPC"*. Both contain words the
  denylist screens for. Both are allowed, because a legal tool that refuses those is worse than one
  with no guardrails.

Now untick **embedding layer** and re-run. Catch rate drops to **15/19 (79%)**. The four cases that
escape — capital of France, sorting a list, pizza toppings, yesterday's cricket — contain no keyword
trigger at all. That is the evidence the guardrail generalises rather than matching a wordlist.

Finish by trying one live in **Ask LegalAId**: `Write me a poem about the monsoon`. It is refused,
and the refusal names the layer that caught it.

---

## 11. The evaluation (2 min)

Open `docs/evaluation.md`. Do not read it out; point at three things.

1. **The model comparison** — three families, one corpus, identical prompts. `llama3.2:3b` leads on
   accuracy *and* latency, so there is no trade-off among these three.
2. **The refusal column.** `qwen2.5:3b` declined all 3 unanswerable questions; `gemma2:2b` declined
   none and was the only model to cite chunk ids it was never given. For legal work that matters
   more than a few points of accuracy.
3. **The retriever ablation** — and specifically that its first run found the hybrid retrieval
   *worse* than dense alone. Explain the fix: the fusion weight was set by reasoning, the sweep
   measured it, recall went 0.568 → 0.682. A negative finding that was diagnosed and corrected is
   worth more than a table of favourable numbers.

If asked what surprised you: retrieval triples accuracy (16.7% → 54.2%) but does **not** reduce
fabrication. It relocates it — from answerable questions to the unanswerable ones. Grounding fixes
what a model knows, not whether it knows when to stop.

---

## Mapping to the exercises

| Exercise | Where to point |
|---|---|
| 1 — App + API + Ollama + a code model | `/docs`, System page, any model call in a trace |
| 2 — Knowledge base + chunking + embeddings | Law library, Chunk explorer, Embeddings |
| 3 — Vector similarity + retrieval + RAG | Retrieval inspector, RAG on/off |
| 4 — APIs + services + orchestration | Pipeline trace (service + transport per stage), Agents |
| 5 — Docker | `docker compose ps`, `docs/docker.md` |

Week 4 additions: model comparison and all metrics in `docs/evaluation.md`; guardrails on the
**Guardrails** page and in `docs/guardrails.md`; the codebase-understanding probe in Exercise 6 of
the evaluation.

---

## Timing and honesty

A full run is **60–90 seconds** on the host GPU, longer on CPU in Docker, and **86% of that is
inside model generation** — measured, not estimated. If time is tight, open a previously completed
trace from **Home → recent research** and walk through that instead; every page reads recorded data,
so nothing has to be re-run live.

Things worth saying out loud rather than hiding:

- **Case-law coverage is thin.** 47 judgments from 2023 is a small library, so the Similar Case
  Agent sometimes returns a loosely related authority — and marks `evidence_sufficient: false`
  rather than overselling it. It is also why all 45 extracted citations resolve *outside* the
  corpus: a 2023 slice cites earlier authority we do not hold.
- **Accuracy fell when the corpus grew.** 263 → 1,783 provisions dropped `llama3.2:3b` from 66.7%
  to 55.6%. The task got harder, not the system worse: retrieval now discriminates among seven
  times as many candidates. It is the reason every accuracy figure in the report names its corpus.
- **The system cannot answer questions about its own codebase** — 0 of 61 relevant files retrieved,
  because there is no code in the corpus. It now declines those instead of returning statute at a
  confident-looking similarity, which was the dangerous half of that failure.
- **Nothing here establishes legal correctness.** Every metric is about grounding and scope. That is
  what the disclaimer and lawyer review are for.
