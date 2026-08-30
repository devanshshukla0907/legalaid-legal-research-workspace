"""Corpus reconciliation: what is actually stored, per act, with provenance completeness.

This is the reconciliation table the data-acquisition research called for - never trust a single
headline count. It reports per act: provisions, chunks, vectors, how many carry a source URL,
and how many carry a real content hash.
"""

from __future__ import annotations

from sqlalchemy import case, func, select

from legalaid.db.base import session_scope
from legalaid.db.models import Case, Chunk, EmbeddingMetadata, LegalProvision
from legalaid.ingestion.corpus import all_acts


def main() -> None:
    with session_scope() as session:
        total_provisions = session.scalar(select(func.count()).select_from(LegalProvision)) or 0
        total_chunks = session.scalar(select(func.count()).select_from(Chunk)) or 0
        total_vectors = session.scalar(select(func.count()).select_from(EmbeddingMetadata)) or 0
        total_cases = session.scalar(select(func.count()).select_from(Case)) or 0

        print(f"provisions : {total_provisions}")
        print(f"cases      : {total_cases}")
        print(f"chunks     : {total_chunks}")
        print(f"vectors    : {total_vectors}")

        print(f"\n{'act':<44}{'prov':>6}{'chunks':>8}{'url':>6}{'hash':>6}")
        print("-" * 72)
        rows = session.execute(
            select(
                LegalProvision.act_name,
                func.count(LegalProvision.id),
                func.sum(case((LegalProvision.source_url.isnot(None), 1), else_=0)),
                # A real content hash is 32 hex chars; the old length-pair marker was short.
                func.sum(case((func.length(LegalProvision.version) >= 16, 1), else_=0)),
            ).group_by(LegalProvision.act_name)
        ).all()

        by_act = {name: (count, urls or 0, hashed or 0) for name, count, urls, hashed in rows}
        chunk_rows = session.execute(
            select(Chunk.source_title, func.count(Chunk.id))
            .where(Chunk.provision_id.isnot(None))
            .group_by(Chunk.source_title)
        ).all()
        chunks_by_act = {title: count for title, count in chunk_rows}

        for name in sorted(by_act):
            count, urls, hashed = by_act[name]
            print(
                f"{name[:43]:<44}{count:>6}{chunks_by_act.get(name, 0):>8}"
                f"{urls:>6}{hashed:>6}"
            )

        print("\nmanifest acts not yet ingested:")
        stored = set(by_act)
        missing = [s.title for s in all_acts() if s.title not in stored]
        for title in missing:
            print(f"  - {title}")
        if not missing:
            print("  (none)")


if __name__ == "__main__":
    main()
