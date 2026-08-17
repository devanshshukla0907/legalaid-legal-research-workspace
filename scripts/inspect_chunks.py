"""Inspect stored chunks to sanity-check chunking quality.

Usage:
  python scripts/inspect_chunks.py                # summary
  python scripts/inspect_chunks.py --section 43   # show chunks for one section
"""

from __future__ import annotations

import argparse

from sqlalchemy import select

from legalaid.db.base import session_scope
from legalaid.db.models import Chunk, LegalProvision


def summary() -> None:
    with session_scope() as session:
        provisions = session.scalars(select(LegalProvision)).all()
        print(f"{'section':<10} {'prov chars':>10} {'chunks':>7} {'chunk chars':>28}  heading")
        print("-" * 100)
        for provision in sorted(provisions, key=lambda p: (p.act_name, p.section_number or "")):
            chunks = session.scalars(
                select(Chunk).where(Chunk.provision_id == provision.id).order_by(Chunk.ordinal)
            ).all()
            sizes = [c.char_count for c in chunks]
            shown = ",".join(str(s) for s in sizes[:6])
            if len(sizes) > 6:
                shown += ",..."
            print(
                f"{provision.section_number or '-':<10} {len(provision.text):>10} "
                f"{len(chunks):>7} {shown:>28}  {(provision.section_heading or '')[:34]}"
            )

        total_chunks = session.scalars(select(Chunk)).all()
        if total_chunks:
            sizes = [c.char_count for c in total_chunks]
            tiny = [s for s in sizes if s < 300]
            print(
                f"\nchunks={len(sizes)}  min={min(sizes)}  median={sorted(sizes)[len(sizes)//2]}  "
                f"max={max(sizes)}  under_300_chars={len(tiny)}"
            )


def show_section(section: str) -> None:
    with session_scope() as session:
        provision = session.scalar(
            select(LegalProvision).where(LegalProvision.section_number == section)
        )
        if provision is None:
            print(f"no provision for section {section}")
            return
        print(f"Section {provision.section_number} — {provision.section_heading}")
        print(f"provision chars: {len(provision.text)}")
        print(f"source: {provision.source_url}\n")
        chunks = session.scalars(
            select(Chunk).where(Chunk.provision_id == provision.id).order_by(Chunk.ordinal)
        ).all()
        for chunk in chunks:
            print("=" * 90)
            print(
                f"[{chunk.ordinal}] {chunk.char_count} chars | strategy={chunk.chunk_strategy} "
                f"| subsection={chunk.chunk_metadata.get('subsection')}"
            )
            print("-" * 90)
            print(chunk.text[:600])
            print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section")
    args = parser.parse_args()
    if args.section:
        show_section(args.section)
    else:
        summary()


if __name__ == "__main__":
    main()
