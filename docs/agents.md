# LegalAId — Agent Design

Four cooperating components. Three are LLM agents; one is a deterministic orchestrator.
Each agent has its own prompt file, its own retrieval scope, its own output schema, and — critically —
its own **isolated context**.

## 1. Orchestrator (deterministic controller, not an LLM prompt)

The orchestrator is code, not a prompt. It is deterministic, inspectable and testable, which is what
makes the multi-agent claim defensible.

Responsibilities:
1. Create the `trace_id` and open the trace.
2. Run **document understanding** to extract facts/issues from matter documents (LLM call, Code Llama).
3. Plan which agents to run and what each one needs (`OrchestrationPlan`, persisted to the trace).
4. Dispatch retrieval, then run Legal Research + Similar Case agents **concurrently** (`asyncio.gather`)
   because they have no mutual dependency.
5. Await both, then run the Drafting Agent, which **depends** on both outputs.
6. Validate citations, assemble the final structured answer, close the trace.

```
                       question + matter
                              │
                    ┌─────────▼─────────┐
                    │  Doc Understanding│  (facts, issues, provisions mentioned)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Plan + Retrieval │  law / case / user-doc searches
                    └────┬─────────┬────┘
              parallel   │         │
              ┌──────────▼──┐  ┌───▼───────────────┐
              │ Legal       │  │ Similar Case      │
              │ Research    │  │ Agent             │
              └──────────┬──┘  └───┬───────────────┘
                         └────┬────┘
                        ┌─────▼─────┐
                        │ Drafting  │  (depends on both)
                        └─────┬─────┘
                              │
                     ┌────────▼────────┐
                     │ Citation validate│
                     │ + assemble       │
                     └─────────────────┘
```

## 2. Agent 1 — Legal Research Agent

- **Purpose:** identify applicable Acts, sections, rules, regulations, notifications.
- **Retrieval scope:** `LEGAL_ACTS`, `LEGAL_RULES`, `LEGAL_REGULATIONS`, `LEGAL_NOTIFICATIONS`.
- **Context it receives:** user question, extracted facts, law chunks, a small number of
  matter-document chunks.
- **Context it is denied:** case-law chunks (recorded in `excluded_chunk_ids`).
- **Prompt:** `prompts/legal_research/legal-research-v1.yaml`
- **Temperature:** 0.1 (extraction, not creativity)
- **Output schema:**

```json
{ "legal_issues": [], "applicable_laws": [], "sections": [],
  "reasoning": [], "uncertainties": [], "citations": [],
  "evidence_sufficient": true }
```

Every entry in `sections` must carry the `chunk_id` it came from. Entries whose `chunk_id` was not in
the supplied context are marked `verified: false` by the validator and surfaced as such in the UI.

## 3. Agent 2 — Similar Case Agent

- **Purpose:** find factually or legally similar Indian precedents.
- **Retrieval scope:** `CASE_LAW` only, with metadata filters (court, provisions, date range).
- **Context it receives:** case facts, legal issues, candidate case chunks, a few law provisions
  for grounding.
- **Context it is denied:** the full law corpus and unrelated document chunks.
- **Prompt:** `prompts/similar_cases/similar-cases-v1.yaml`
- **Temperature:** 0.15
- **Output schema:**

```json
{ "cases": [ { "case_title": "", "court": "", "citation": "", "similarity_score": 0.0,
               "why_similar": [], "relevant_principle": "", "outcome": "",
               "differences": [], "practical_relevance": "", "chunk_ids": [] } ],
  "evidence_sufficient": true }
```

`similarity_score` is taken from the **vector search**, not invented by the model; the agent may not
overwrite it. This keeps a numeric claim honest.

## 4. Agent 3 — Legal Drafting Agent

- **Purpose:** produce a lawyer-reviewable draft grounded strictly in supplied evidence.
- **Retrieval scope:** none of its own. It consumes upstream agent outputs plus verified facts.
- **Context it receives:** verified facts, Legal Research output, Similar Case output, user
  instructions, draft type/tone/length.
- **Prompt:** `prompts/drafting/drafting-v1.yaml`
- **Temperature:** 0.2
- **Hard constraints, enforced in prompt and re-checked after generation:** must not invent cases,
  sections, facts, dates, parties or quotations. Missing information must be emitted as an explicit
  `[INFORMATION NOT AVAILABLE: ...]` marker, which the UI renders as a warning chip.
- **Output:** markdown draft plus `grounding` (which upstream items were relied on) and
  `unverified_flags`.

## 5. Context isolation, made visible

`AgentContext` is built by an explicit builder per agent. It records both what was included and what
was withheld, and the UI renders it verbatim:

```
Agent: Legal Research Agent
Included:  user question, extracted facts, 6 law chunks, 2 user-document chunks
Excluded:  14 case-law chunks  (reason: outside agent retrieval scope)
```

Because inclusion and exclusion are persisted per run, this is auditable after the fact rather than
being a claim made in a slide.

## 6. Tools

Agents call typed tools rather than having ambient access to everything. Every invocation is
recorded in `agent_runs.tools_used` with arguments and timing.

| Tool | Available to |
|---|---|
| `search_laws` | Legal Research |
| `search_rules` | Legal Research |
| `search_cases` | Similar Case |
| `search_user_documents` | Legal Research, Similar Case |
| `get_chunk` | all |
| `validate_citation` | all |

## 7. Prompt management

Versioned YAML under `prompts/<agent>/<name>-v<version>.yaml`, each with `name`, `version`,
`description`, `system_prompt`, `variables`, `created_at`, `model_defaults`. Loaded by
`PromptRegistry`, which resolves the active version from config and records the exact version used
in `agent_runs` and `model_runs`. No large prompt strings live inside Python functions.

## 8. Structured output from a code model

Code Llama is code-specialised, so JSON adherence is decent but not guaranteed. The parsing strategy
is layered and every layer is visible in the trace:

1. Direct `json.loads`.
2. Fenced-block extraction, then balanced-brace scan.
3. One **repair pass**: the raw text is returned to the model with the schema and a fix instruction.
4. On continued failure, the agent returns `status: parse_failed` with the raw text preserved.

Step 4 is a real outcome shown in the UI, not a silent fallback to fabricated structure.
