"""Ingest a curated subset of Supreme Court judgments into the CASE_LAW collection.

Usage:
  python scripts/ingest_cases.py --limit 8 --pool 20     # quick smoke run
  python scripts/ingest_cases.py --limit 120             # full MVP corpus
  python scripts/ingest_cases.py --reset
  python scripts/ingest_cases.py --stats
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from legalaid.db.base import create_all, session_scope
from legalaid.db.models import Case, Chunk
from legalaid.domain.enums import Collection
from legalaid.services.knowledge.ingest_cases import ingest_cases, reset_cases
from legalaid.services.vector.store import get_store


def print_stats() -> None:
    with session_scope() as session:
        cases = session.scalars(select(Case)).all()
        chunks = session.scalars(
            select(Chunk).where(Chunk.collection == Collection.CASE_LAW.value)
        ).all()
    vectors = get_store().index(Collection.CASE_LAW).count
    print(f"cases stored     : {len(cases)}")
    print(f"case chunks      : {len(chunks)}")
    print(f"case vectors     : {vectors}")
    if cases:
        years: dict[int, int] = {}
        for case in cases:
            years[case.year or 0] = years.get(case.year or 0, 0) + 1
        print(f"years            : {dict(sorted(years.items()))}")
        print("\nsample:")
        for case in cases[:8]:
            print(f"  {case.citation or '-':<24} {case.case_title[:58]}")
            if case.legal_provisions:
                print(f"      provisions: {', '.join(case.legal_provisions[:3])}")


async def run(args: argparse.Namespace) -> None:
    report = await ingest_cases(
        years=args.years,
        limit=args.limit,
        pool_per_year=args.pool,
        delay=args.delay,
    )
    print()
    print(report.summary())
    print(f"    fetch {report.fetch_seconds:.1f}s | embed {report.embed_seconds:.1f}s")
    for failure in report.failures[:5]:
        print(f"    ! {failure}")
    print()
    print_stats()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=120, help="max judgments to store")
    parser.add_argument("--pool", type=int, default=130, help="candidate pool per year")
    parser.add_argument(
        "--years", type=int, nargs="*", default=[2023], help="years to draw from"
    )
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--reset", action="store_true", help="delete all case law")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    create_all()
    if args.reset:
        print(f"removed {reset_cases()} cases")
        return 0
    if args.stats:
        print_stats()
        return 0
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
