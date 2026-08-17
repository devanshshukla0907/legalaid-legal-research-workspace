"""Clear ingested knowledge (provisions, chunks, embeddings, vectors).

Leaves users, workspaces, matters and uploaded documents untouched, so a re-ingest does not
destroy the workspace. Use before re-running ingestion after a chunking change.

Usage: python scripts/reset_knowledge.py --yes
"""

from __future__ import annotations

import argparse

from sqlalchemy import delete, select

from legalaid.db.base import create_all, session_scope
from legalaid.db.models import (
    Chunk,
    CitationRecord,
    EmbeddingMetadata,
    LegalProvision,
    RetrievalResult,
    RetrievalRun,
)
from legalaid.domain.enums import Collection
from legalaid.services.vector.store import get_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="confirm the reset")
    args = parser.parse_args()

    create_all()
    if not args.yes:
        with session_scope() as session:
            provisions = len(session.scalars(select(LegalProvision)).all())
            chunks = len(session.scalars(select(Chunk)).all())
        print(f"would delete {provisions} provisions and {chunks} knowledge chunks")
        print("re-run with --yes to proceed")
        return

    with session_scope() as session:
        knowledge_chunk_ids = [
            row.id
            for row in session.scalars(
                select(Chunk).where(Chunk.collection != Collection.USER_DOCUMENTS.value)
            ).all()
        ]
        if knowledge_chunk_ids:
            # Order matters. retrieval_results and citations reference chunks, so deleting
            # chunks first fails the foreign key. Clear dependents from the leaves inward.
            session.execute(
                delete(CitationRecord).where(CitationRecord.chunk_id.in_(knowledge_chunk_ids))
            )
            session.execute(
                delete(RetrievalResult).where(
                    RetrievalResult.chunk_id.in_(knowledge_chunk_ids)
                )
            )
            session.execute(
                delete(EmbeddingMetadata).where(
                    EmbeddingMetadata.chunk_id.in_(knowledge_chunk_ids)
                )
            )
            session.execute(delete(Chunk).where(Chunk.id.in_(knowledge_chunk_ids)))
        # Retrieval runs are only meaningful alongside their results.
        session.execute(delete(RetrievalRun))
        session.execute(delete(LegalProvision))

    store = get_store()
    for collection in Collection:
        if collection is Collection.USER_DOCUMENTS:
            continue
        store.index(collection).clear()

    print(f"cleared {len(knowledge_chunk_ids)} knowledge chunks, provisions and their vectors")


if __name__ == "__main__":
    main()
