"""Legislation ingestion: India Code -> LegalProvision -> Chunk -> embedding -> FAISS.

Each statutory section becomes one `LegalProvision` row carrying its full provenance (act,
section number, heading, source URL, amendment footnotes, retrieval timestamp) and is then
chunked with the statute-aware chunker. Chunks inherit that provenance, which is what makes a
citation traceable from a generated answer back to an official source URL.

Ingestion is resumable and idempotent: a section already stored at the same version is skipped,
so an interrupted run can simply be restarted. Statutory text is never synthesised - if a
section cannot be fetched it is reported as a failure and left absent.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from legalaid.capabilities import require
from legalaid.core.logging import get_logger
from legalaid.db.base import session_scope, utcnow
from legalaid.db.models import (
    Chunk,
    CitationRecord,
    EmbeddingMetadata,
    KnowledgeSource,
    LegalProvision,
    RetrievalResult,
)
from legalaid.domain.enums import Collection
from legalaid.ingestion import india_code, india_code_api
from legalaid.ingestion.chunking import chunk_statute
from legalaid.ingestion.corpus import ActSpec, all_acts, get_act, wants_section
from legalaid.services.embedding.service import embed_texts_raw
from legalaid.services.vector.store import get_store

logger = get_logger(__name__)

INDIA_CODE_SOURCE_KEY = "india_code"


@dataclass(slots=True)
class ActIngestReport:
    act_key: str
    act_title: str
    sections_fetched: int = 0
    sections_stored: int = 0
    sections_skipped: int = 0
    sections_repealed: int = 0
    sections_purged: int = 0
    sections_empty: int = 0
    chunks_created: int = 0
    vectors_indexed: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)
    fetch_seconds: float = 0.0
    embed_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.act_title}: {self.sections_stored} provisions, "
            f"{self.chunks_created} chunks, {self.vectors_indexed} vectors "
            f"({self.sections_skipped} skipped incl. {self.sections_repealed} repealed"
            + (f", {self.sections_purged} purged" if self.sections_purged else "")
            + f", {len(self.failures)} failures)"
        )


def _ensure_source(retrieved_at: datetime) -> str:
    """Return the India Code knowledge-source id, creating the row if the seed has not run."""
    with session_scope() as session:
        source = session.scalar(
            select(KnowledgeSource).where(KnowledgeSource.key == INDIA_CODE_SOURCE_KEY)
        )
        if source is None:
            source = KnowledgeSource(
                key=INDIA_CODE_SOURCE_KEY,
                name="India Code - Central Acts",
                authority=india_code_api.SOURCE_AUTHORITY,
                source_url=india_code_api.HOST,
                licence="Government of India open content; reproduced with attribution",
                attribution=(
                    "Statutory text reproduced from India Code, the official legislation "
                    "repository of the Government of India."
                ),
            )
            session.add(source)
            session.flush()
        source.retrieved_at = retrieved_at
        return source.id


_REPEALED_MARKERS = ("repealed", "omitted", "rep. by", "[repealed", "[omitted")


def is_repealed(section: india_code.FetchedSection) -> bool:
    """Whether a section is repealed, omitted or spent.

    These must not be indexed as live law. Observed in real retrieval output: a query naming
    "Section 73" ranked *"Section 259 - Repealed ... Rep. by s. 73"* third, purely because the
    repeal note mentions s.73. Citing a dead provision as applicable law is exactly the failure
    mode a legal research tool must not have, so they are excluded at ingestion.

    The check is deliberately conservative - the heading must say so, or a very short body must
    be essentially nothing but a repeal note - so a live section that merely *discusses* repeals
    is kept.
    """
    # The DSpace API reports repeal status directly. Where the publisher says so, that is
    # authoritative and no inference is needed.
    if section.repealed is True:
        return True

    heading = (section.heading or "").strip().lower()
    if any(heading.startswith(marker) or heading == marker.strip("[") for marker in _REPEALED_MARKERS):
        return True
    if heading in {"repealed", "omitted", "repealed.", "omitted."}:
        return True

    body = section.text.strip().lower()
    if len(body) <= 220 and ("rep. by" in body or "omitted by" in body):
        return True
    return False


def _provision_version(section: india_code.FetchedSection) -> str:
    """Version marker for change detection.

    A content hash, not a length comparison. The previous marker was
    ``f"{len(text)}-{len(footnotes)}"``, described in its own docstring as "cheap and stable" -
    but any amendment that preserved both lengths was invisible to it, and substituting one word
    for another of the same length is an entirely ordinary amendment. Calling that "versioning"
    overstated what it did.

    Whitespace is normalised first so a reformatting change upstream does not present as a
    substantive amendment and force a pointless re-embed.
    """
    if section.content_sha256:
        return section.content_sha256
    return content_hash(section.text, section.footnotes)


def content_hash(*parts: str) -> str:
    """Stable sha256 over whitespace-normalised text."""
    normalised = "\x1f".join(" ".join((part or "").split()) for part in parts)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:32]


async def ingest_act(
    spec: ActSpec,
    *,
    delay: float = 0.4,
    max_sections: int | None = None,
    reembed: bool = False,
) -> ActIngestReport:
    """Fetch, store, chunk, embed and index one act."""
    require("knowledge_ingestion")
    report = ActIngestReport(act_key=spec.key, act_title=spec.title)

    started = asyncio.get_running_loop().time()
    # The fetcher is blocking (urllib + polite pacing); keep the event loop free.
    #
    # Resolution prefers the pinned act id and falls back to a title search. The id is exact,
    # but India Code holds state adoptions of every central act, so a title search must be
    # constrained to the CENTRAL copy - which `find_act` does.
    def _fetch():
        ref = None
        if spec.act_id:
            ref = india_code_api.fetch_act_metadata(spec.act_id)
        if ref is None:
            ref = india_code_api.find_act(spec.resolve_title)
        return india_code_api.fetch_act(ref, delay=delay, progress=True)

    act = await asyncio.to_thread(_fetch)
    if max_sections:
        act.sections = act.sections[:max_sections]
    report.fetch_seconds = asyncio.get_running_loop().time() - started
    report.sections_fetched = act.section_count
    report.failures.extend(act.failures)

    if spec.expected_sections and act.section_count == 0:
        report.failures.append({"act": spec.key, "error": "no sections fetched"})
        return report

    source_id = _ensure_source(act.retrieved_at)
    in_scope = [s for s in act.sections if wants_section(spec, s.section_number)]

    # Provisions with a heading but no body are dropped. India Code carries several - the
    # Arbitration Council sections 43F, 43G and 43L among them - with a title and no text at
    # all. Indexed, they are retrievable and citable while saying nothing, which is worse than
    # absent: the system would offer a lawyer a provision reference that carries no law.
    # Found by scripts/analyse_corpus.py, which flagged 13 of them as exact duplicates of each
    # other because their text was uniformly empty.
    empty = [s for s in in_scope if s.is_empty]
    if empty:
        report.sections_empty = len(empty)
        logger.info(
            "india_code_empty_sections_skipped",
            act=spec.key,
            count=len(empty),
            sections=[s.section_number for s in empty][:12],
        )
        in_scope = [s for s in in_scope if not s.is_empty]

    # Repealed and omitted provisions are excluded so they can never be retrieved as live law.
    live = [s for s in in_scope if not is_repealed(s)]
    repealed_numbers = {s.section_number for s in in_scope if is_repealed(s)}
    report.sections_repealed = len(repealed_numbers)
    in_scope = live
    # act.section_count already reflects only-in-scope fetching, so anything missing here was
    # dropped as out-of-scope or repealed.
    report.sections_skipped = max(0, act.section_count - len(in_scope))

    # A provision stored by an earlier run before the repealed filter existed would otherwise
    # linger in the index forever, since it is no longer in `in_scope` to be re-evaluated.
    # Purge them so the filter is retroactive rather than only affecting fresh ingests.
    if repealed_numbers:
        report.sections_purged = _purge_provisions(spec, repealed_numbers)

    # ------------------------------------------------------------ provisions
    new_chunk_ids: list[str] = []
    with session_scope() as session:
        existing = {
            (row.section_number or ""): row
            for row in session.scalars(
                select(LegalProvision).where(
                    LegalProvision.act_name == spec.title,
                    LegalProvision.collection == spec.collection.value,
                )
            ).all()
        }

        for section in in_scope:
            version = _provision_version(section)
            current = existing.get(section.section_number)
            if current is not None and current.version == version:
                continue  # unchanged, already ingested

            if current is not None:
                # Text changed upstream: drop stale chunks so the index cannot serve both.
                stale_ids = [
                    c.id
                    for c in session.scalars(
                        select(Chunk).where(Chunk.provision_id == current.id)
                    ).all()
                ]
                _delete_chunks(session, stale_ids, spec.collection)
                session.delete(current)
                session.flush()

            provision = LegalProvision(
                source_id=source_id,
                collection=spec.collection.value,
                act_name=spec.title,
                act_year=spec.year,
                section_number=section.section_number,
                section_heading=section.heading or None,
                text=section.text,
                source_url=section.source_url,
                source_authority=india_code.SOURCE_AUTHORITY,
                document_path=act.pdf_url,
                version=version,
                amendment_note=section.footnotes or None,
                is_synthetic=False,
                retrieved_at=act.retrieved_at,
            )
            session.add(provision)
            session.flush()
            report.sections_stored += 1

            # Feed the chunker text shaped the way it expects to see a statute section.
            section_text = f"{section.section_number}. {section.heading}\n\n{section.text}"
            candidates = chunk_statute(
                text=section_text,
                act_name=spec.title,
                collection=spec.collection,
                source_url=section.source_url,
                act_year=spec.year,
                extra_metadata={
                    "act_key": spec.key,
                    "act_id": act.actid,
                    "act_handle": act.handle,
                    "provision_id": provision.id,
                    "section_number": section.section_number,
                    "section_heading": section.heading,
                    "citation": f"Section {section.section_number}, {spec.citation_name}",
                    "source_authority": india_code.SOURCE_AUTHORITY,
                    "has_amendments": bool(section.footnotes),
                },
            )
            if not candidates:
                report.failures.append(
                    {"section": section.section_number, "error": "produced no chunks"}
                )
                continue

            for candidate in candidates:
                chunk = Chunk(
                    provision_id=provision.id,
                    collection=candidate.collection.value,
                    source_type=candidate.source_type,
                    source_title=candidate.source_title,
                    section=f"Section {section.section_number}",
                    chapter=candidate.chapter,
                    heading=section.heading or candidate.heading,
                    text=candidate.text,
                    char_count=candidate.char_count,
                    token_count=candidate.token_count,
                    ordinal=candidate.ordinal,
                    chunk_strategy=candidate.chunk_strategy.value,
                    source_url=section.source_url,
                    chunk_metadata=candidate.metadata,
                    version=version,
                )
                session.add(chunk)
                session.flush()
                new_chunk_ids.append(chunk.id)
                report.chunks_created += 1

    # --------------------------------------------------------------- embed
    target_ids = new_chunk_ids
    if reembed:
        with session_scope() as session:
            target_ids = [
                row.id
                for row in session.scalars(
                    select(Chunk).where(Chunk.collection == spec.collection.value)
                ).all()
            ]

    if target_ids:
        embed_started = asyncio.get_running_loop().time()
        report.vectors_indexed = await _embed_and_index(target_ids, spec.collection)
        report.embed_seconds = asyncio.get_running_loop().time() - embed_started

    _update_source_counts()
    logger.info("act_ingested", act=spec.key, **{
        "provisions": report.sections_stored,
        "chunks": report.chunks_created,
        "vectors": report.vectors_indexed,
    })
    return report


def _delete_chunks(session: Session, chunk_ids: list[str], collection: Collection) -> None:
    """Delete chunks and everything referencing them, leaves first.

    `retrieval_results` and `citations` hold foreign keys to `chunks`, so deleting chunks
    first raises a constraint error. The vector must go too, otherwise search would return a
    chunk id that no longer resolves.
    """
    if not chunk_ids:
        return
    get_store().index(collection).delete(chunk_ids)
    session.execute(delete(CitationRecord).where(CitationRecord.chunk_id.in_(chunk_ids)))
    session.execute(delete(RetrievalResult).where(RetrievalResult.chunk_id.in_(chunk_ids)))
    session.execute(delete(EmbeddingMetadata).where(EmbeddingMetadata.chunk_id.in_(chunk_ids)))
    session.execute(delete(Chunk).where(Chunk.id.in_(chunk_ids)))


def _purge_provisions(spec: ActSpec, section_numbers: set[str]) -> int:
    """Remove previously stored provisions that should no longer be indexed."""
    with session_scope() as session:
        rows = session.scalars(
            select(LegalProvision).where(
                LegalProvision.act_name == spec.title,
                LegalProvision.section_number.in_(section_numbers),
            )
        ).all()
        if not rows:
            return 0
        for provision in rows:
            chunk_ids = [
                c.id
                for c in session.scalars(
                    select(Chunk).where(Chunk.provision_id == provision.id)
                ).all()
            ]
            _delete_chunks(session, chunk_ids, spec.collection)
            session.delete(provision)
        logger.info("provisions_purged", act=spec.key, count=len(rows))
        return len(rows)


async def _embed_and_index(chunk_ids: list[str], collection: Collection) -> int:
    """Embed chunks in batches and upsert them into the collection index."""
    require("embeddings")
    require("vector_index")

    with session_scope() as session:
        rows = session.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids))).all()
        pending = [(r.id, r.text, dict(r.chunk_metadata or {})) for r in rows]
        done = {
            r.chunk_id
            for r in session.scalars(
                select(EmbeddingMetadata).where(EmbeddingMetadata.chunk_id.in_(chunk_ids))
            ).all()
        }
    todo = [item for item in pending if item[0] not in done]
    if not todo:
        return 0

    vectors, elapsed_ms, model_name = await embed_texts_raw([text for _, text, _ in todo])
    per_item_ms = elapsed_ms / len(vectors) if vectors else 0.0

    index = get_store().index(collection)
    vector_ids = index.upsert(
        [(cid, vec, meta) for (cid, _t, meta), vec in zip(todo, vectors, strict=True)],
        embedding_model=model_name,
    )

    from legalaid.config import get_settings
    from legalaid.core.tokens import estimate_tokens
    from legalaid.services.embedding.service import _norm, _preview

    settings = get_settings()
    with session_scope() as session:
        for (chunk_id, text, _meta), vector, vector_id in zip(
            todo, vectors, vector_ids, strict=True
        ):
            session.add(
                EmbeddingMetadata(
                    chunk_id=chunk_id,
                    vector_id=vector_id,
                    collection=collection.value,
                    embedding_provider=settings.embedding_provider,
                    embedding_model=model_name,
                    dimensions=len(vector),
                    norm=_norm(vector),
                    latency_ms=per_item_ms,
                    input_tokens=estimate_tokens(text),
                    vector_preview=_preview(vector),
                    created_at=utcnow(),
                )
            )
    return len(vector_ids)


def _update_source_counts() -> None:
    with session_scope() as session:
        source = session.scalar(
            select(KnowledgeSource).where(KnowledgeSource.key == INDIA_CODE_SOURCE_KEY)
        )
        if source is None:
            return
        source.item_count = len(
            session.scalars(
                select(LegalProvision).where(LegalProvision.source_id == source.id)
            ).all()
        )


async def ingest_corpus(
    *,
    act_keys: list[str] | None = None,
    delay: float = 0.4,
    max_sections: int | None = None,
) -> list[ActIngestReport]:
    """Ingest the whole manifest, or a named subset."""
    specs = [get_act(k) for k in act_keys] if act_keys else list(all_acts())
    reports: list[ActIngestReport] = []
    for spec in specs:
        try:
            reports.append(
                await ingest_act(spec, delay=delay, max_sections=max_sections)
            )
        except Exception as exc:  # noqa: BLE001 - one bad act must not abort the run
            logger.error("act_ingest_failed", act=spec.key, error=str(exc))
            report = ActIngestReport(act_key=spec.key, act_title=spec.title)
            report.failures.append({"act": spec.key, "error": str(exc)})
            reports.append(report)
    return reports


def knowledge_stats() -> dict[str, object]:
    """Counts rendered by the Dashboard and Knowledge Base pages."""
    store = get_store()
    with session_scope() as session:
        provisions = session.scalars(select(LegalProvision)).all()
        chunks = session.scalars(select(Chunk)).all()
        by_act: dict[str, int] = {}
        for provision in provisions:
            by_act[provision.act_name] = by_act.get(provision.act_name, 0) + 1
    return {
        "provisions": len(provisions),
        "chunks": len(chunks),
        "provisions_by_act": by_act,
        "vectors_by_collection": {
            collection.value: index.count for collection, index in store.all_indexes().items()
        },
        "total_vectors": store.total_vectors(),
        "checked_at": datetime.now(UTC).isoformat(),
    }
