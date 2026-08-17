# Demo walkthrough

A 10–12 minute script that shows every element the brief asks for, in an order that builds.

**Before you start**

```powershell
curl http://localhost:8000/api/system/health   # expect status ok, vectors > 600
```

Warm the model first — the very first generation of a session pays a ~90 s load penalty that will
otherwise look like a hang:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" run codellama:7b-instruct "ok" --verbose
```

Have a second browser tab on `http://localhost:8000/docs`.

---

## 1. The lawyer's view (1 min)

Open **http://localhost:5173** → lands on **Home**.

Point out what a practising advocate sees: a single box that accepts anything, four common starting
points, recent research, and what the library covers — 263 provisions from five acts plus 47
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

## Mapping to the exercises

| Exercise | Where to point |
|---|---|
| 1 — App + API + Ollama + Code Llama | `/docs`, System page, any model call in a trace |
| 2 — Knowledge base + chunking + embeddings | Law library, Chunk explorer, Embeddings |
| 3 — Vector similarity + retrieval + RAG | Retrieval inspector, RAG on/off |
| 4 — APIs + services + orchestration | Pipeline trace (service + transport per stage), Agents |
| 5 — Docker | `docker compose ps`, `docs/docker.md` |

---

## Timing and honesty

A full run is **3–7 minutes** on the host GPU (~12 tok/s), longer on CPU in Docker. If time is
tight, either switch to `qwen2.5:3b-instruct`, or open a previously completed trace from **Home →
recent research** and walk through that instead — every page reads recorded data, so nothing has to
be re-run live.

Two things worth saying out loud rather than hiding:

- **Case-law coverage is thin.** 47 judgments from 2023 is a small library for contract disputes,
  so the Similar Case Agent sometimes returns a loosely related authority — and marks
  `evidence_sufficient: false` rather than overselling it. That honesty is the designed behaviour.
- **Code Llama is code-specialised**, so its prose is weaker than a chat model's. That is why output
  is schema-constrained at the decoder and grounded strictly in retrieved text.
