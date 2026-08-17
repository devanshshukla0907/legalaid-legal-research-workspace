"""Case-law ingestion: e-SCR judgments -> Case rows -> chunks -> embeddings -> FAISS.

Two-stage curation, because relevance cannot be judged cheaply:

1. **Screen on title** (free - metadata is already local). Commercial parties and obvious
   subject matter get a judgment into the candidate pool; clearly unrelated matter is dropped.
2. **Verify on real text** after fetching the PDF. The headnote states the issues and the
   statutes in play, so scoring it is what actually decides inclusion. A judgment that fails
   verification is discarded rather than indexed.

Stage 2 is the honest but expensive step, so the candidate pool is capped and judgments are
fetched with a delay. Nothing about a case is synthesised: title, citation, judges, date and
disposal come from the official metadata, text from the official PDF.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from sqlalchemy import delete, select

from legalaid.capabilities import require
from legalaid.core.logging import get_logger
from legalaid.db.base import session_scope
from legalaid.db.models import Case, Chunk, CitationRecord, EmbeddingMetadata, KnowledgeSource, RetrievalResult
from legalaid.domain.enums import Collection
from legalaid.ingestion import case_law
from legalaid.ingestion.case_curation import ScoredCandidate, diversify, is_excluded, score_candidate
from legalaid.ingestion.chunking import chunk_judgment, segment_judgment_text
from legalaid.services.knowledge.ingest import _embed_and_index
from legalaid.services.vector.store import get_store

logger = get_logger(__name__)

CASE_SOURCE_KEY = "supreme_court_judgments"

# A judgment must score at least this well on its real headnote text to be indexed.
MIN_VERIFIED_SCORE = 12
# Titles are weak evidence, so the screen is permissive; verification does the real filtering.
MIN_TITLE_SCORE = 0


@dataclass(slots=True)
class CaseIngestReport:
    years: list[int] = field(default_factory=list)
    candidates: int = 0
    fetched: int = 0
    stored: int = 0
    rejected: int = 0
    chunks_created: int = 0
    vectors_indexed: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)
    fetch_seconds: float = 0.0
    embed_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"Case law: {self.stored} judgments stored, {self.chunks_created} chunks, "
            f"{self.vectors_indexed} vectors "
            f"({self.fetched} fetched, {self.rejected} rejected on verification, "
            f"{len(self.failures)} failures)"
        )


def _ensure_source() -> str:
    with session_scope() as session:
        source = session.scalar(
            select(KnowledgeSource).where(KnowledgeSource.key == CASE_SOURCE_KEY)
        )
        if source is None:
            source = KnowledgeSource(
                key=CASE_SOURCE_KEY,
                name="Indian Supreme Court Judgments (open dataset)",
                authority=case_law.SOURCE_AUTHORITY,
                source_url="https://registry.opendata.aws/indian-supreme-court-judgments/",
                licence=case_law.LICENCE,
                attribution=case_law.ATTRIBUTION,
            )
            session.add(source)
            session.flush()
        return source.id


# A judgment's full text is mostly not useful for precedent matching, and indexing all of it
# actively hurts. Measured on a 6-case sample: 341 chunks, one constitutional-bench judgment
# alone producing 151. That single case would then dominate every similarity search, and the
# page-by-page procedural narrative it contributes is noise for "find me a similar case".
#
# e-SCR judgments carry a court-authored HEADNOTE stating the facts, the issues for
# consideration and what was held. That is precisely the material a Similar Case Agent needs,
# it is authoritative rather than our summarisation, and it is a few thousand characters rather
# than hundreds of thousands. So the headnote and the recognised legal segments are indexed;
# unsegmented full text is capped instead of split wholesale.
MAX_FULLTEXT_CHARS = 12_000
MAX_SEGMENT_CHARS = 9_000


def _segments_for_index(
    record: case_law.CaseRecord, segments: dict[str, str]
) -> dict[str, str]:
    """Choose what to index for a judgment, preferring the court's own headnote."""
    chosen: dict[str, str] = {}

    if record.headnote:
        chosen["headnote"] = record.headnote[:MAX_SEGMENT_CHARS]
    if record.issues:
        chosen["issues"] = record.issues[:2000]

    # Keep genuinely recognised legal segments, trimmed.
    for key in ("facts", "reasoning", "ratio", "decision"):
        value = (segments.get(key) or "").strip()
        if value:
            chosen[key] = value[:MAX_SEGMENT_CHARS]

    # `judgment` is the segmenter's fallback: it means no headings were recognised, so the
    # whole opinion arrived as one blob. Cap it rather than exploding it into 150 chunks.
    if not chosen:
        fallback = (segments.get("judgment") or record.text).strip()
        chosen["judgment_extract"] = fallback[:MAX_FULLTEXT_CHARS]

    return chosen


def _screen(rows: list[dict[str, object]], *, pool: int) -> list[ScoredCandidate]:
    """Cheap title-based screen to build a candidate pool worth fetching."""
    scored: list[ScoredCandidate] = []
    for row in rows:
        title = str(row.get("title") or "")
        if not title or is_excluded(title):
            continue
        candidate = score_candidate(row, "")
        # Keep everything not explicitly excluded; titles rarely reveal the legal issue.
        scored.append(candidate)

    # Prefer judgments whose titles already look commercial, then fill with the rest so the
    # pool is not biased to only the obvious cases.
    scored.sort(key=lambda c: -c.score)
    return scored[:pool]


async def ingest_cases(
    *,
    years: list[int] | None = None,
    limit: int = 120,
    pool_per_year: int = 130,
    delay: float = 0.25,
) -> CaseIngestReport:
    """Fetch, curate, store, chunk, embed and index a subset of judgments."""
    require("knowledge_ingestion")
    years = years or [2023]
    report = CaseIngestReport(years=years)
    source_id = _ensure_source()

    candidates: list[ScoredCandidate] = []
    for year in years:
        rows = await asyncio.to_thread(case_law.load_year_metadata, year)
        candidates.extend(_screen(rows, pool=pool_per_year))
    report.candidates = len(candidates)

    with session_scope() as session:
        already = {row.case_ref for row in session.scalars(select(Case)).all()}

    loop = asyncio.get_running_loop()
    fetch_started = loop.time()
    new_chunk_ids: list[str] = []
    accepted: list[tuple[case_law.CaseRecord, int, list[str]]] = []

    for candidate in candidates:
        if len(accepted) >= limit:
            break
        case_id = str(candidate.row.get("case_id") or candidate.row.get("path") or "")
        if case_id in already:
            continue

        try:
            record = await asyncio.to_thread(case_law.fetch_judgment, candidate.row)
        except case_law.CaseLawError as exc:
            report.failures.append({"case": case_id, "error": str(exc)[:160]})
            continue
        report.fetched += 1

        # Stage 2: verify against real judgment text, not the title.
        probe = record.headnote or record.text[:6000]
        verified = score_candidate(candidate.row, probe)
        if verified.score < MIN_VERIFIED_SCORE:
            report.rejected += 1
            logger.debug(
                "case_rejected",
                case=record.short_title[:60],
                score=verified.score,
                reason=verified.reason,
            )
            continue

        accepted.append((record, verified.score, verified.topics))
        if delay:
            await asyncio.sleep(delay)

    report.fetch_seconds = loop.time() - fetch_started

    # Keep topic balance so one area cannot dominate the corpus.
    accepted.sort(key=lambda item: -item[1])
    balanced: list[tuple[case_law.CaseRecord, int, list[str]]] = []
    per_topic: dict[str, int] = {}
    cap = max(6, limit // 3)
    for record, score, topics in accepted:
        primary = topics[0] if topics else "other"
        if per_topic.get(primary, 0) >= cap:
            continue
        per_topic[primary] = per_topic.get(primary, 0) + 1
        balanced.append((record, score, topics))

    with session_scope() as session:
        for record, score, topics in balanced:
            segments_for_row = segment_judgment_text(record.text)
            case_row = Case(
                source_id=source_id,
                case_ref=record.case_id,
                case_title=record.short_title,
                court=record.court,
                citation=record.citation,
                neutral_citation=record.neutral_citation,
                decision_date=record.decision_date,
                year=record.year,
                judges=record.judges,
                legal_provisions=record.legal_provisions,
                keywords=record.keywords or topics,
                facts_summary=(record.headnote or "")[:4000],
                issues=record.issues or "",
                reasoning=(segments_for_row.get("reasoning") or "")[:8000],
                ratio=(segments_for_row.get("ratio") or "")[:4000],
                decision=(segments_for_row.get("decision") or "")[:4000],
                disposal_nature=record.disposal_nature,
                source_url=record.source_url,
                document_path=record.source_url,
                licence=case_law.LICENCE,
                dataset="indian-supreme-court-judgments",
                is_synthetic=False,
                retrieved_at=record.retrieved_at,
            )
            session.add(case_row)
            session.flush()
            report.stored += 1

            segments = _segments_for_index(record, segments_for_row)
            metadata_text = "\n".join(
                part
                for part in (
                    f"Case: {record.short_title}",
                    f"Court: {record.court}",
                    f"Citation: {record.citation}",
                    f"Neutral citation: {record.neutral_citation}" if record.neutral_citation else "",
                    f"Decided: {record.decision_date}" if record.decision_date else "",
                    f"Bench: {', '.join(record.judges)}" if record.judges else "",
                    f"Outcome: {record.disposal_nature}" if record.disposal_nature else "",
                    f"Provisions considered: {', '.join(record.legal_provisions)}"
                    if record.legal_provisions else "",
                    f"Issues: {record.issues}" if record.issues else "",
                )
                if part
            )

            candidates_chunks = chunk_judgment(
                metadata_text=metadata_text,
                segments=segments,
                case_title=record.short_title,
                source_url=record.source_url,
                extra_metadata={
                    "case_id": record.case_id,
                    "case_db_id": case_row.id,
                    "court": record.court,
                    "citation": record.citation,
                    "neutral_citation": record.neutral_citation,
                    "decision_date": record.decision_date,
                    "year": record.year,
                    "outcome": record.disposal_nature,
                    "legal_provisions": record.legal_provisions,
                    "topics": topics,
                    "relevance_score": score,
                    "source_authority": case_law.SOURCE_AUTHORITY,
                    "licence": case_law.LICENCE,
                },
            )

            for chunk_candidate in candidates_chunks:
                chunk = Chunk(
                    case_id=case_row.id,
                    collection=Collection.CASE_LAW.value,
                    source_type=chunk_candidate.source_type,
                    source_title=record.short_title,
                    section=chunk_candidate.section,
                    heading=chunk_candidate.heading,
                    text=chunk_candidate.text,
                    char_count=chunk_candidate.char_count,
                    token_count=chunk_candidate.token_count,
                    ordinal=chunk_candidate.ordinal,
                    chunk_strategy=chunk_candidate.chunk_strategy.value,
                    source_url=record.source_url,
                    chunk_metadata=chunk_candidate.metadata,
                )
                session.add(chunk)
                session.flush()
                new_chunk_ids.append(chunk.id)
                report.chunks_created += 1

    if new_chunk_ids:
        embed_started = loop.time()
        report.vectors_indexed = await _embed_and_index(new_chunk_ids, Collection.CASE_LAW)
        report.embed_seconds = loop.time() - embed_started

    with session_scope() as session:
        source = session.scalar(
            select(KnowledgeSource).where(KnowledgeSource.key == CASE_SOURCE_KEY)
        )
        if source is not None:
            source.item_count = len(session.scalars(select(Case)).all())

    logger.info(
        "cases_ingested",
        stored=report.stored,
        chunks=report.chunks_created,
        vectors=report.vectors_indexed,
        rejected=report.rejected,
    )
    return report


def reset_cases() -> int:
    """Remove all case law, leaves-inward so foreign keys hold."""
    with session_scope() as session:
        chunk_ids = [
            row.id
            for row in session.scalars(
                select(Chunk).where(Chunk.collection == Collection.CASE_LAW.value)
            ).all()
        ]
        if chunk_ids:
            get_store().index(Collection.CASE_LAW).delete(chunk_ids)
            session.execute(delete(CitationRecord).where(CitationRecord.chunk_id.in_(chunk_ids)))
            session.execute(delete(RetrievalResult).where(RetrievalResult.chunk_id.in_(chunk_ids)))
            session.execute(delete(EmbeddingMetadata).where(EmbeddingMetadata.chunk_id.in_(chunk_ids)))
            session.execute(delete(Chunk).where(Chunk.id.in_(chunk_ids)))
        removed = len(session.scalars(select(Case)).all())
        session.execute(delete(Case))
    get_store().index(Collection.CASE_LAW).clear()
    return removed
