"""Diagnose case-law chunking: how many chunks per judgment, and from which segments."""

from __future__ import annotations

from sqlalchemy import select

from legalaid.db.base import session_scope
from legalaid.db.models import Case, Chunk
from legalaid.domain.enums import Collection


def main() -> None:
    with session_scope() as session:
        cases = session.scalars(select(Case)).all()
        print(f"{'chunks':>7} {'chars':>9}  case")
        print("-" * 84)
        for case in cases:
            chunks = session.scalars(
                select(Chunk).where(Chunk.case_id == case.id).order_by(Chunk.ordinal)
            ).all()
            total = sum(c.char_count for c in chunks)
            print(f"{len(chunks):>7} {total:>9}  {case.case_title[:58]}")
            segments: dict[str, int] = {}
            for chunk in chunks:
                key = str((chunk.chunk_metadata or {}).get("segment") or chunk.section or "-")
                segments[key] = segments.get(key, 0) + 1
            for key, count in sorted(segments.items(), key=lambda kv: -kv[1]):
                print(f"          {count:>4}  {key}")

        all_chunks = session.scalars(
            select(Chunk).where(Chunk.collection == Collection.CASE_LAW.value)
        ).all()
        if all_chunks:
            sizes = sorted(c.char_count for c in all_chunks)
            print(
                f"\ncase chunks={len(sizes)} min={sizes[0]} "
                f"p50={sizes[len(sizes)//2]} p95={sizes[int(len(sizes)*0.95)]} max={sizes[-1]}"
            )
            print(f"total chars stored: {sum(sizes):,}")


if __name__ == "__main__":
    main()
