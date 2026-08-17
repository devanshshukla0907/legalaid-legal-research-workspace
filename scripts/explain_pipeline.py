"""Trace one real provision from source URL to stored vector, printing what exists at each step.

Answers, with actual data rather than description:
  * do we download PDFs, or text?
  * where did this text come from?
  * what does a chunk look like?
  * what exactly was embedded, by which model, into which index?

Usage: python scripts/explain_pipeline.py [section_number]
"""

from __future__ import annotations

import sys

from sqlalchemy import select

from legalaid.db.base import session_scope
from legalaid.db.models import Chunk, EmbeddingMetadata, LegalProvision
from legalaid.domain.enums import Collection
from legalaid.services.vector.store import get_store

SECTION = sys.argv[1] if len(sys.argv) > 1 else "73"


def main() -> None:
    with session_scope() as session:
        provision = session.scalar(
            select(LegalProvision).where(
                LegalProvision.section_number == SECTION,
                LegalProvision.act_name.like("%Contract%"),
            )
        )
        if provision is None:
            print(f"No provision found for section {SECTION}")
            return

        print("=" * 78)
        print("STEP 1 — WHERE THE TEXT CAME FROM (stored in legal_provisions)")
        print("=" * 78)
        print(f"  act            : {provision.act_name}")
        print(f"  section        : {provision.section_number} — {provision.section_heading}")
        print(f"  authority      : {provision.source_authority}")
        print(f"  source_url     : {provision.source_url}")
        print(f"  document_path  : {provision.document_path}")
        print(f"  retrieved_at   : {provision.retrieved_at}")
        print(f"  version        : {provision.version}")
        print(f"  is_synthetic   : {provision.is_synthetic}")
        print(f"  text length    : {len(provision.text)} characters")
        print(f"\n  first 300 chars of the ACTUAL statutory text:")
        print(f"  {provision.text[:300]!r}")
        if provision.amendment_note:
            print(f"\n  amendment footnotes (from the same API response):")
            print(f"  {provision.amendment_note[:200]!r}")

        chunks = session.scalars(
            select(Chunk).where(Chunk.provision_id == provision.id).order_by(Chunk.ordinal)
        ).all()

        print()
        print("=" * 78)
        print(f"STEP 2 — CHUNKING: this provision became {len(chunks)} chunk(s)")
        print("=" * 78)
        for chunk in chunks:
            print(f"  [{chunk.ordinal}] {chunk.id}")
            print(f"       strategy   : {chunk.chunk_strategy}")
            print(f"       chars/tokens: {chunk.char_count} / {chunk.token_count}")
            print(f"       collection : {chunk.collection}")
            print(f"       citation   : {(chunk.chunk_metadata or {}).get('citation')}")
            print(f"       text starts: {chunk.text[:110]!r}")

        print()
        print("=" * 78)
        print("STEP 3 — EMBEDDING: what was turned into a vector, and by what")
        print("=" * 78)
        store = get_store()
        for chunk in chunks:
            embedding = session.scalar(
                select(EmbeddingMetadata).where(EmbeddingMetadata.chunk_id == chunk.id)
            )
            if embedding is None:
                print(f"  {chunk.id}: NOT embedded")
                continue
            vector = store.index(Collection(chunk.collection)).get_vector(chunk.id)
            print(f"  chunk        : {chunk.id}")
            print(f"  model        : {embedding.embedding_model} ({embedding.embedding_provider})")
            print(f"  dimensions   : {embedding.dimensions}")
            print(f"  L2 norm      : {embedding.norm:.6f}  (1.0 => normalised, so dot = cosine)")
            print(f"  embed latency: {embedding.latency_ms:.0f} ms")
            print(f"  vector_id    : {embedding.vector_id}   <- position in the FAISS index")
            print(f"  index file   : data/indexes/{chunk.collection.lower()}.faiss")
            if vector:
                shown = ", ".join(f"{v:.5f}" for v in vector[:6])
                print(f"  first 6 of {len(vector)} floats: [{shown}, ...]")
            print()

        print("=" * 78)
        print("STEP 4 — WHAT IS IN EACH FAISS INDEX")
        print("=" * 78)
        for collection, index in store.all_indexes().items():
            if index.count:
                print(
                    f"  {collection.value:22} {index.count:>5} vectors  "
                    f"model={index.embedding_model}  file={index.index_path.name}"
                )


if __name__ == "__main__":
    main()
