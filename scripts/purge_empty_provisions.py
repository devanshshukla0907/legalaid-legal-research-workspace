"""Remove provisions stored with no text, and their chunks and vectors.

Retroactive counterpart to the ingestion-time filter. A provision with a heading and no body is
retrievable and citable while carrying no law, so it is worse than absent - the system would
hand a lawyer a reference that says nothing. Found by scripts/analyse_corpus.py.
"""

from __future__ import annotations

from sqlalchemy import func, or_, select

from legalaid.db.base import session_scope
from legalaid.db.models import Chunk, EmbeddingMetadata, LegalProvision
from legalaid.services.retrieval.graph import reset_graph
from legalaid.services.retrieval.keyword import reset_index


def main() -> None:
    with session_scope() as session:
        empty = session.scalars(
            select(LegalProvision).where(
                or_(
                    LegalProvision.text.is_(None),
                    func.trim(LegalProvision.text) == "",
                )
            )
        ).all()

        if not empty:
            print("No empty provisions found.")
            return

        print(f"Found {len(empty)} provisions with no text:")
        by_act: dict[str, list[str]] = {}
        for row in empty:
            by_act.setdefault(row.act_name, []).append(row.section_number or "?")
        for act, sections in sorted(by_act.items()):
            print(f"  {act}: {', '.join(sorted(sections))}")

        provision_ids = [row.id for row in empty]
        chunks = session.scalars(
            select(Chunk).where(Chunk.provision_id.in_(provision_ids))
        ).all()
        chunk_ids = [c.id for c in chunks]

        # Deleted deepest-first with an explicit flush between each level. Without the flushes
        # SQLAlchemy is free to order the DELETE statements as it likes within one flush, and
        # SQLite rejects the provision delete while a chunk still references it.
        embeddings = 0
        if chunk_ids:
            rows = session.scalars(
                select(EmbeddingMetadata).where(EmbeddingMetadata.chunk_id.in_(chunk_ids))
            ).all()
            embeddings = len(rows)
            for row in rows:
                session.delete(row)
            session.flush()

        for chunk in chunks:
            session.delete(chunk)
        session.flush()

        for provision in empty:
            session.delete(provision)
        session.flush()

        print(
            f"\nDeleted {len(empty)} provisions, {len(chunk_ids)} chunks, "
            f"{embeddings} embedding records."
        )

    # The FAISS index still holds the vectors; the lexical index and graph are rebuilt lazily.
    # A full re-embed run rewrites the vector store, so the stale vectors are cleared by
    # scripts/ingest_legislation.py --all --reembed if exactness matters.
    reset_index()
    reset_graph()
    print("Lexical index and graph reset.")


if __name__ == "__main__":
    main()
