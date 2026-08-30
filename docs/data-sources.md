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

**Source: [India Code](https://indiacode.gov.in/)**, the official GoI legislation repository
(Legislative Department, Ministry of Law and Justice).

> ### Migration, August 2026
>
> India Code moved from `www.indiacode.nic.in` to `indiacode.gov.in` and rebuilt as a DSpace 9.1
> Angular application. This broke the original ingester outright, and the failure is worth
> recording because it is the ordinary condition of scraping a government site:
>
> * every `/handle/` path on the old host now returns 404 — the host serves only a migration
>   notice;
> * on the new host the act page is an empty SPA shell, so there is no section index to parse;
> * the custom `/SectionPageContent` endpoint returns 502.
>
> The replacement is **better than what it replaced**. DSpace exposes a documented REST API, and
> India Code models each section as its own item:
>
> | Field | Carries |
> |---|---|
> | `dc.identifier.collection` | `ACT` or `SECTION` |
> | `dc.identifier.section_number` | the section number |
> | `dc.title` | the section heading |
> | `dc.identifier.section_page_note` | the provision text |
> | `dc.identifier.act_id` | links a section to its act |
> | `dc.identifier.state_name` | `CENTRAL` or the adopting state |
> | `dc.identifier.repealed` | publisher-supplied repeal flag |
> | `dc.date.enforcement_date` | real commencement date |
>
> The last two were **not available at all** through the old route. Repeal status had to be
> inferred from heading text, and commencement did not exist — which is what made temporal
> reasoning impossible to build honestly. Both are now recorded per provision.
>
> Two traps found while implementing `ingestion/india_code_api.py`:
>
> 1. **State adoptions outrank the central copy.** India Code holds dozens of state adoptions of
>    each central Act and relevance does not prefer the central one — the central *Right to
>    Information Act* ranked **eighth**. `CENTRAL` must be filtered server-side, not by
>    post-filtering a page of results, or acts are silently reported as missing.
> 2. **Without `embed=indexableObject` the search returns links only**, so every section costs a
>    separate request: 268 extra round trips for one Act. With it, the Contract Act fetches in
>    8 seconds.

Status: Indian statutory text is government-authored legal text. Under §52(1)(q) of the Copyright
Act, 1957 reproduction of Acts and of government-published material is permitted, and India Code
is the official publication channel. National Data Sharing policy material is issued under the
Government Open Data Licence – India. We nonetheless treat every item as **attribution-required**
and record the source URL, retrieval timestamp and authority on every chunk.

**Not vendored into git.** The Acts are fetched by `scripts/fetch_legislation.py` into
`data/knowledge_sources/india_code/`, which is gitignored. The repo ships a **manifest** describing
exactly which Acts, which version, and which URL — so the corpus is reproducible without
redistributing bulk PDFs.

### Corpus as ingested (15 Acts, 1,783 provisions)

Every `act_id` was verified against the live API before being committed to the manifest
(`scripts/check_indiacode_api.py`). A wrong identifier silently ingests the wrong statute, which is
worse than ingesting nothing.

| Act | Year | Provisions | Commencement (as India Code reports it) | Why included |
|---|---|---|---|---|
| Indian Contract Act | 1872 | 192 | 1872-09-01 | Formation, breach, compensation (ss.73–75), indemnity, guarantee, bailment, agency |
| Bharatiya Nyaya Sanhita | 2023 | 358 | — | Current criminal code; cheating and criminal breach of trust shadow commercial disputes |
| Bharatiya Sakshya Adhiniyam | 2023 | 170 | — | Current evidence law; how an agreement or notice is actually proved |
| Code of Civil Procedure | 1908 | 158 | — | Jurisdiction, res judicata, interest, costs, s.89 ADR |
| Negotiable Instruments Act | 1881 | 154 | 1882-03-01 | s.138 cheque dishonour, one of India's highest-volume proceedings |
| Transfer of Property Act | 1882 | 135 | 1882-07-01 | Sale, mortgage, lease, part performance |
| Information Technology Act | 2000 | 109 | — | s.10A validates electronic contracts; ss.65–66, 43A |
| Consumer Protection Act | 2019 | 107 | staged from 24 Jul 2020 by S.O. 2421(E) | Product liability; a cheaper forum than a civil suit |
| Arbitration and Conciliation Act | 1996 | 106 | 22-08-1996 | Most commercial contracts arbitrate |
| Indian Partnership Act | 1932 | 73 | 1932-10-01 (except s.69), 1933-10-01 | s.69 bar on suits by unregistered firms |
| Sale of Goods Act | 1930 | 66 | 1930-07-01 | Passing of property, conditions and warranties |
| Bharatiya Nagarik Suraksha Sanhita | 2023 | 61 | — | Current criminal procedure; **filtered** to what a commercial matter reaches |
| Specific Relief Act | 1963 | 46 | 01-03-1964 | Specific performance, injunctions, rescission |
| Right to Information Act | 2005 | 31 | ss.4(1), 5(1)–(2), 12, 13, 15, 16, 24, 27, 28 on 15-06-2005; rest on the 120th day | Obtaining documents from a public authority |
| Limitation Act | 1963 | 30 | 01-01-1964 | Whether a claim is time-barred is usually asked first |

Section filters are used **only** for BNSS, where 531 sections of criminal procedure are mostly
outside the application's scope. Everything else is held in full: filtering a statute means a
question about an unfiltered section retrieves nothing, and "the corpus does not contain it" is a
far less useful answer than the provision itself.

Two commencement entries are worth noting because they show why a single date field would have
been wrong. The Consumer Protection Act 2019 commenced **section by section** via notification,
and the RTI Act commenced some sections immediately and the rest on the 120th day. India Code
records these as prose, and they are stored as the publisher's own string rather than coerced
into a date.

Deliberately excluded: Companies Act 2013 and the Insolvency and Bankruptcy Code (both resolve
correctly through the API but would each add several hundred provisions of largely
out-of-scope regulatory detail); the Constitution (available, but no evaluation task reaches it).

### Preserved per provision

`act_name`, `act_year`, `chapter`, `section_number`, `section_heading`, `subsection`,
`provision_text`, `source_url`, `source_authority`, `retrieved_at`, `version`, `amendment_note`.

`version` is a **sha256 over whitespace-normalised text**, not a length comparison. It previously
held `f"{len(text)}-{len(footnotes)}"`, described in its own docstring as "cheap and stable" — but
any amendment preserving both lengths was invisible to it, and substituting one word for another
of equal length is an entirely ordinary amendment. Calling that versioning overstated what it did.
The hash also makes re-ingestion properly idempotent.

### Reconciliation

Never trust a single headline count. `scripts/corpus_stats.py` reports per act: provisions,
chunks, how many carry a source URL, and how many carry a real content hash. Current state:

| | Count |
|---|---|
| Provisions | 1,783 |
| Judgments | 47 |
| Chunks | 2,575 |
| Vectors | 2,575 |
| Provisions with `source_url` | 1,783 (100%) |
| Provisions with content hash | 1,783 (100%) |

Growth from the original MVP: 263 → 1,783 provisions, 651 → 2,575 chunks.

### Data-quality findings

`scripts/analyse_corpus.py` runs exact, near-duplicate (SimHash, 64-bit, Hamming ≤ 3) and
structural duplicate detection. It earned its place immediately:

**13 provisions were being indexed with no text at all.** India Code carries the Arbitration
Council sections (43A–43M) with headings and empty bodies. Indexed, they were retrievable and
citable while saying nothing — worse than absent, because the system would offer a lawyer a
provision reference carrying no law. Now filtered at ingestion and purged retroactively
(`scripts/purge_empty_provisions.py`).

| Metric | Before | After |
|---|---|---|
| Provision duplicate rate | 0.84% | **0.17%** |
| Chunk duplicate rate | 0.00% | 0.00% |

The 2 remaining exact duplicates are **genuine statutory parallels**: Arbitration Act s.46 and
s.55 are word-for-word identical across the New York and Geneva Convention chapters. The single
near-duplicate is Consumer Protection Act s.50 vs s.60, parallel provisions for different
commission tiers. Neither is a data error, and neither is removed.

### Citation graph

`ingestion/citations.py` extracts six Indian reporter formats, expands section references across
lists and ranges, and classifies **treatment** from the language around each citation.

Treatment is deliberately separate from `CITES`. A judgment that distinguishes or overrules
another cites it just as much as one that follows it, and conflating those would let the system
present overruled authority as support.

On the current corpus: 45 citations, 38 section references, and treatment breakdown
`distinguished` 7, `relied_on` 5, `followed` 2, `reversed` 2, `referred` 14, `cited` 15.

**All 45 citations resolve outside the corpus**, and that is honest rather than broken: the 47
judgments are a 2023 slice and cite earlier authority the corpus does not hold. Unresolved
citations are reported, not dropped.

Nothing here infers whether a precedent is still good law. That needs subsequent-treatment
analysis across a far larger corpus.

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
| Commercial legal databases | Proprietary | **Rejected on principle** — see below |

### Sources deliberately NOT ingested

Knowing what not to ingest is part of the answer, so the reasoning is recorded rather than left
implicit.

| Source | Scale / claim | Why not |
|---|---|---|
| **SCC Online** | Extensive, editorially enriched | Terms limit the standard licence to research and study and state that a **separate licence agreement is required to commercially exploit the database**. Copying it into a RAG corpus would need that licence. |
| **Manupatra** | Extensive | Terms permit reproduction for private legal research under stated conditions but **restrict machine-readable reproduction** and prohibit commercial distribution without permission. |
| **Taxmann / LexisNexis / Westlaw / AIR** | Proprietary | Product-access licences, not data licences. Subscription access is not the same thing as corpus rights. |
| **Publisher headnotes, digests, case summaries** | — | *Eastern Book Company v. D.B. Modak* is the clear warning: the underlying judgment may be reproducible under §52(1)(q)(iv) while the publisher's editorial layer — copy-editing, arrangement, headnotes — is separately protected. Those are different things and must not be conflated. |
| **High Court bulk dataset** | ~17.8 M judgments, ~1.25 TiB | Licence-compatible (CC-BY-4.0) but the wrong scale for this project. The ingestion pipeline would accept it without code changes. |
| **District / subordinate courts** | Reported in the tens of millions | No convenient national bulk archive; would require a per-court ingestion programme. |
| **Any dataset claiming CC-BY without demonstrable upstream provenance** | — | A README asserting a licence is not evidence of the right to grant it. |

The load-bearing distinction throughout:

```
RAW JUDGMENT  ≠  PUBLISHER HEADNOTE  ≠  CASE SUMMARY  ≠  EDITORIAL COMMENTARY
```

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
