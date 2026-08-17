"""Audit the ingested corpus for quality problems that would undermine citations.

Checks:
  * repealed/omitted provisions that slipped into the index
  * provisions with suspiciously short text
  * chunks missing provenance (source URL / citation)
  * chunk size distribution
  * every chunk has an embedding and a vector
"""

from __future__ import annotations

import re

from sqlalchemy import select

from legalaid.db.base import session_scope
from legalaid.db.models import Chunk, EmbeddingMetadata, LegalProvision
from legalaid.domain.enums import Collection
from legalaid.services.vector.store import get_store

REPEALED = re.compile(r"^\s*(\[?\s*(repealed|omitted)|rep\. by)", re.I)


def main() -> int:
    problems = 0
    with session_scope() as session:
        provisions = session.scalars(select(LegalProvision)).all()
        chunks = session.scalars(select(Chunk)).all()
        embeddings = session.scalars(select(EmbeddingMetadata)).all()

        print(f"provisions : {len(provisions)}")
        print(f"chunks     : {len(chunks)}")
        print(f"embeddings : {len(embeddings)}")

        repealed = [
            p for p in provisions
            if REPEALED.match(p.section_heading or "")
            or (len(p.text.strip()) <= 220 and "rep. by" in p.text.lower())
        ]
        print(f"\nrepealed/omitted still indexed : {len(repealed)}")
        for p in repealed[:8]:
            print(f"  ! s.{p.section_number}: {(p.section_heading or '')[:60]}")
        problems += len(repealed)

        short = [p for p in provisions if len(p.text.strip()) < 60]
        print(f"provisions with < 60 chars     : {len(short)}")
        for p in short[:8]:
            print(f"  ? s.{p.section_number}: {p.text.strip()[:60]!r}")

        no_url = [c for c in chunks if not c.source_url]
        no_cite = [c for c in chunks if not (c.chunk_metadata or {}).get("citation")]
        print(f"chunks missing source_url      : {len(no_url)}")
        print(f"chunks missing citation        : {len(no_cite)}")
        problems += len(no_url) + len(no_cite)

        sizes = sorted(c.char_count for c in chunks)
        if sizes:
            print(
                f"\nchunk chars: min={sizes[0]} p50={sizes[len(sizes)//2]} "
                f"p95={sizes[int(len(sizes)*0.95)]} max={sizes[-1]}"
            )
            tiny = [s for s in sizes if s < 300]
            print(f"chunks under 300 chars         : {len(tiny)}")

        missing_embedding = len(chunks) - len(embeddings)
        print(f"chunks without an embedding     : {missing_embedding}")
        problems += max(0, missing_embedding)

        models = {e.embedding_model for e in embeddings}
        dims = {e.dimensions for e in embeddings}
        print(f"embedding models used           : {models or '-'}")
        print(f"embedding dimensions            : {dims or '-'}")
        if len(dims) > 1:
            print("  ! mixed dimensions in one index")
            problems += 1

    vectors = get_store().index(Collection.LEGAL_ACTS).count
    print(f"vectors in LEGAL_ACTS           : {vectors}")
    if vectors != len(embeddings):
        print(f"  ! vector/embedding mismatch ({vectors} vs {len(embeddings)})")
        problems += 1

    print(f"\nPROBLEMS: {problems}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
