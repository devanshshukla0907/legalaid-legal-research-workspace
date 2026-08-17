# LegalAId — Database Schema

SQLAlchemy 2.0 typed models. SQLite for dev (zero setup, matches the "no unnecessary infrastructure"
rule), Postgres-compatible: no SQLite-only types, UUID-as-string primary keys, JSON columns via the
portable `JSON` type.

## 1. Tenancy and identity (firm-ready from day one)

```
organizations       id, name, slug, kind('personal'|'firm'), created_at
users               id, email, full_name, password_hash, is_active, created_at
roles               id, organization_id, name, is_system            -- Firm Admin/Partner/Associate/Paralegal/Researcher/Read-only
permissions         id, code, description                            -- matter.read, document.upload, draft.create, ...
role_permissions    role_id, permission_id
memberships         id, organization_id, user_id, role_id, status, created_at
workspaces          id, organization_id, owner_user_id, name, kind('personal'|'shared')
```

Phase 1 seeds one `personal` organization, one user, one role (`Owner`) with all permissions, and
one workspace. Nothing in the query layer assumes single-tenancy: every scoped read goes through
`scope_to_workspace()`, so firm mode adds authorization checks rather than schema surgery.

## 2. Matters and documents

```
matters             id, workspace_id, title, reference_no, matter_type, court, parties(JSON),
                    description, status, extracted_facts(JSON), legal_issues(JSON), created_at
matter_members      id, matter_id, user_id, role_id, access_level        -- future firm mode
documents           id, matter_id, workspace_id, filename, stored_path, content_type, size_bytes,
                    sha256, doc_class, page_count, status, extraction_method, metadata(JSON),
                    is_synthetic, uploaded_at, processed_at
document_pages      id, document_id, page_number, text, char_count, headings(JSON), extraction_conf
document_permissions id, document_id, role_id, user_id, access_level      -- future firm mode
```

`sha256` gives idempotent re-upload detection. `is_synthetic` makes fixture data impossible to
mistake for genuine legal material.

## 3. Knowledge base

```
knowledge_sources   id, key, name, authority, source_url, licence, attribution, dataset_version,
                    retrieved_at, item_count
legal_provisions    id, source_id, collection, act_name, act_year, chapter, section_number,
                    section_heading, subsection, text, source_url, version, amendment_note
cases               id, source_id, case_id, case_title, court, citation, neutral_citation,
                    decision_date, judges(JSON), legal_provisions(JSON), keywords(JSON),
                    facts_summary, issues, decision, ratio, source_url, document_path, licence
```

## 4. Chunks and vectors

```
chunks              id, document_id?, provision_id?, case_id?, collection, source_type,
                    source_title, section, chapter, page, heading, text, char_count, token_count,
                    ordinal, parent_chunk_id, chunk_strategy, metadata(JSON), created_at, version
embeddings_metadata id, chunk_id, vector_id, collection, embedding_provider, embedding_model,
                    dimensions, norm, latency_ms, input_tokens, vector_preview(JSON), created_at
```

Full vectors live in FAISS, **not** in the relational DB. Only a 16-value preview is stored for the
UI inspector — enough to demonstrate the representation without duplicating the index.

## 5. Observability (this is what powers the whole UI)

```
pipeline_traces     id(trace_id), workspace_id, matter_id?, mode('rag'|'no_rag'|'compare'),
                    question, status, started_at, completed_at, total_ms, error,
                    capability_snapshot(JSON), summary(JSON)
trace_stages        id, trace_id, ordinal, stage_key, label, service, transport, status,
                    started_at, completed_at, duration_ms, input(JSON), output(JSON), error
retrieval_runs      id, trace_id, mode, collection, query_text, query_vector_id, top_k,
                    candidate_count, filters(JSON), embedding_model, dimensions,
                    embedding_ms, search_ms, rank_ms
retrieval_results   id, retrieval_run_id, chunk_id, rank, similarity, selected,
                    match_reasons(JSON), matched_terms(JSON)
agent_runs          id, trace_id, agent_key, status, started_at, completed_at, duration_ms,
                    prompt_name, prompt_version, included_chunk_ids(JSON),
                    excluded_chunk_ids(JSON), context_manifest(JSON), tools_used(JSON),
                    input_summary(JSON), output(JSON), error
model_runs          id, trace_id, agent_run_id?, provider, model, role, temperature, max_tokens,
                    system_prompt, user_prompt, raw_response, input_tokens, output_tokens,
                    total_duration_ms, load_ms, eval_ms, tokens_per_second, status, error,
                    request_id, created_at
drafts              id, matter_id, trace_id?, draft_type, tone, length, title, content,
                    grounding(JSON), unverified_flags(JSON), created_at
citations           id, trace_id, kind('law'|'case'|'user_doc'), chunk_id?, label, detail(JSON),
                    verified, agent_key
audit_logs          id, organization_id, user_id?, action, entity_type, entity_id,
                    trace_id?, detail(JSON), created_at
```

`agent_runs.excluded_chunk_ids` is the single most important academic column in this schema: it is
the persisted proof that agents received *isolated*, non-identical context.

`model_runs` stores prompt text so the "What did the model see?" panel reads real recorded data
rather than reconstructing a guess.

## 6. Indices

`trace_stages(trace_id, ordinal)`, `retrieval_results(retrieval_run_id, rank)`,
`chunks(collection)`, `chunks(document_id)`, `embeddings_metadata(chunk_id)`,
`agent_runs(trace_id)`, `model_runs(trace_id)`, `documents(matter_id)`, `matters(workspace_id)`.
