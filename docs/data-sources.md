# LegalAId — Data Source Plan

Decided **before** implementation, because licence terms constrain the architecture (what may be
redistributed in this repo vs. what must be fetched by the user at setup time).

## 1. Principles

1. Authoritative sources only — Government of India or the courts themselves.
2. Small and traceable beats large and vague. Target ~5–10 Acts, 100–300 provisions, 50–200 judgments.
3. Licence-checked before ingestion. Non-commercial / no-redistribution datasets are **not** vendored.
4. No commercial legal database scraping, regardless of how easy access might be.
5. The final demo runs on genuine legal material. Synthetic data is permitted only as a temporary
   UI development fixture and is clearly labelled `synthetic: true`.

## 2. Legislation corpus — primary knowledge base

**Source: [India Code](https://www.indiacode.nic.in/)**, the official GoI legislation repository
(Legislative Department, Ministry of Law and Justice).

Status: Indian statutory text is government-authored legal text. Under §52(1)(q) of the Copyright
Act, 1957 reproduction of Acts and of government-published material is permitted, and India Code
is the official publication channel. National Data Sharing policy material is issued under the
Government Open Data Licence – India. We nonetheless treat every item as **attribution-required**
and record the source URL, retrieval timestamp and authority on every chunk.

**Not vendored into git.** The Acts are fetched by `scripts/fetch_legislation.py` into
`data/knowledge_sources/india_code/`, which is gitignored. The repo ships a **manifest** describing
exactly which Acts, which version, and which URL — so the corpus is reproducible without
redistributing bulk PDFs.

### MVP corpus (10 Acts, chosen to serve the contract-dispute demo plus breadth)

| # | Act | Year | Why included |
|---|---|---|---|
| 1 | Indian Contract Act | 1872 | Core of the demo matter (ss. 73, 74 damages) |
| 2 | Specific Relief Act | 1963 | Contract remedies, injunctions |
| 3 | Consumer Protection Act | 2019 | Frequent commercial disputes |
| 4 | Information Technology Act | 2000 | Electronic records, evidence of digital notices |
| 5 | Arbitration and Conciliation Act | 1996 | Dispute-resolution clauses in agreements |
| 6 | Bharatiya Nyaya Sanhita | 2023 | Current criminal law (replaces IPC) |
| 7 | Bharatiya Nagarik Suraksha Sanhita | 2023 | Current criminal procedure (replaces CrPC) |
| 8 | Bharatiya Sakshya Adhiniyam | 2023 | Current evidence law (replaces Evidence Act) |
| 9 | Code of Civil Procedure | 1908 | Civil suit procedure |
| 10 | Constitution of India | 1950 | Fundamental rights, writ jurisdiction |

Deliberately excluded from the MVP: Companies Act 2013 (very large, low relevance to the demo
matter, would dominate the index without improving demonstrability).

### Preserved per provision

`act_name`, `act_year`, `chapter`, `section_number`, `section_heading`, `subsection`,
`provision_text`, `source_url`, `source_authority`, `retrieved_at`, `version`, `amendment_note`.

## 3. Case-law corpus

**Source: [Indian Supreme Court Judgments open dataset](https://github.com/vanga/indian-supreme-court-judgments)**
— judgments obtained from the official [Supreme Court Reports portal](https://scr.sci.gov.in),
published via the AWS Open Data Sponsorship Program.

**Licence: CC-BY-4.0** — use, share and adapt permitted with attribution. This is the decisive
reason it was selected. Attribution is rendered in the UI Knowledge Base page and in `NOTICE.md`.

Access: public S3, no credentials (`--no-sign-request`), also plain HTTPS. We pull **metadata
parquet for selected years plus a filtered subset of judgment PDFs**, targeting 50–200 judgments
relevant to the MVP legal areas. The dataset's own guidance to avoid high-concurrency scraping is
respected: we read the published bulk artifacts, we do not scrape the court site.

### Rejected alternatives, with reasons

| Dataset | Licence | Decision |
|---|---|---|
| [IL-PCSR](https://huggingface.co/datasets/Exploration-Lab/IL-PCSR) | CC-BY-NC-SA 4.0, gated, redistribution forbidden | **Rejected** — cannot redistribute; access gate breaks reproducible setup |
| [ILDC / CJPE](https://github.com/Exploration-Lab/CJPE) | CC-BY-NC | **Rejected** — non-commercial restriction |
| Indian High Court judgments open dataset | CC-BY-4.0 | Available and licence-compatible, but ~1 TB. **Deferred** — out of scope for MVP size target; the ingestion pipeline accepts it later without code changes |
| Commercial legal databases | Proprietary | **Rejected on principle** |

### Preserved per case

`case_id`, `case_title`, `court`, `citation`, `neutral_citation`, `date`, `judges`,
`legal_provisions`, `keywords`, `facts_summary`, `issues`, `decision`, `ratio`, `source_url`,
`document_path`, `dataset`, `licence`, `attribution`, `retrieved_at`.

## 4. Rules / regulations / notifications

10–50 items, from the rules and notifications already published alongside the selected Acts on
India Code (e.g. IT Rules under the IT Act). Kept small; these mainly exist to prove the retrieval
layer distinguishes source types rather than flattening everything into one index.

## 5. User documents

Lawyer uploads per matter. Never leaves the local deployment, never mixed into the shared knowledge
collections, always scoped to `matter_id`. Stored under `data/uploads/{workspace}/{matter}/`.

## 6. Knowledge collections (separate indexes, not one blob)

```
LEGAL_ACTS          statutory provisions
LEGAL_RULES         subordinate rules
LEGAL_REGULATIONS   regulations
LEGAL_NOTIFICATIONS gazette notifications
CASE_LAW            judgments
USER_DOCUMENTS      matter documents, scoped per matter
```

The retrieval layer always knows which collection it is searching, and the UI shows the collection
name, candidate count and top-k for every search.

## 7. Layout

```
data/
  knowledge_sources/
    india_code/          manifest.json + fetched Acts (gitignored)
    supreme_court/       metadata + selected judgments (gitignored)
    ecourts/             reserved for later expansion
    uploaded_documents/  per-matter uploads
  indexes/               FAISS indexes + metadata sidecars
  legalaid.db            relational store
```

Only manifests, schemas and a small clearly-labelled synthetic fixture set are committed.

## 8. Corpus sizing sanity check

10 Acts ≈ 100–300 indexed provisions; 50–200 judgments chunked into ~6 sections each ≈ 300–1200
chunks; rules/notifications ≈ 50–150 chunks. Total well under ~2,000 vectors at 768 dimensions
≈ 6 MB of float32. FAISS flat inner-product search over that is exact and sub-millisecond, so
approximate indexing (IVF/HNSW) is unnecessary complexity at this scale. Embedding the whole corpus
on CPU with `nomic-embed-text` is the real cost: budget roughly 20–40 ms per chunk, i.e. a
one-off ingestion run of a few minutes.
