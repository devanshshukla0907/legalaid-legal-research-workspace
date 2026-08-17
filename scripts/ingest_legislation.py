"""Ingest Indian legislation from India Code into the knowledge base.

Fetches genuine statutory text, stores provisions with full provenance, chunks them with the
statute-aware chunker, embeds them and indexes the vectors in FAISS.

Usage:
  python scripts/ingest_legislation.py --list
  python scripts/ingest_legislation.py --act it_act_2000
  python scripts/ingest_legislation.py --act it_act_2000 --max-sections 5
  python scripts/ingest_legislation.py --all
  python scripts/ingest_legislation.py --stats
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from legalaid.db.base import create_all
from legalaid.ingestion.corpus import all_acts, estimated_provisions
from legalaid.services.knowledge.ingest import ingest_corpus, knowledge_stats


def print_manifest() -> None:
    print(f"{'key':<22} {'handle':<8} {'sections':>8} {'scope':>6}  title")
    print("-" * 88)
    for spec in all_acts():
        scope = len(spec.section_filter) or (spec.expected_sections or 0)
        print(
            f"{spec.key:<22} {spec.handle:<8} {spec.expected_sections or 0:>8} "
            f"{scope:>6}  {spec.title}"
        )
    print(f"\nestimated provisions in scope: {estimated_provisions()}")


def print_stats() -> None:
    stats = knowledge_stats()
    print(f"provisions stored : {stats['provisions']}")
    print(f"chunks created    : {stats['chunks']}")
    print(f"total vectors     : {stats['total_vectors']}")
    by_act = stats["provisions_by_act"]
    if by_act:
        print("\nprovisions by act:")
        for act, count in sorted(by_act.items(), key=lambda kv: -kv[1]):  # type: ignore[union-attr]
            print(f"  {count:>4}  {act}")
    print("\nvectors by collection:")
    for collection, count in stats["vectors_by_collection"].items():  # type: ignore[union-attr]
        if count:
            print(f"  {count:>4}  {collection}")


async def run(args: argparse.Namespace) -> int:
    create_all()
    act_keys = [args.act] if args.act else None

    reports = await ingest_corpus(
        act_keys=act_keys,
        delay=args.delay,
        max_sections=args.max_sections,
    )

    print()
    failed = 0
    for report in reports:
        print(report.summary())
        print(
            f"    fetch {report.fetch_seconds:.1f}s | embed {report.embed_seconds:.1f}s"
        )
        if report.failures:
            failed += len(report.failures)
            for failure in report.failures[:5]:
                print(f"    ! {failure}")
            if len(report.failures) > 5:
                print(f"    ... {len(report.failures) - 5} more")

    print()
    print_stats()
    # Empty/repealed sections are expected failures, so a non-zero count is not fatal.
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act", help="ingest a single act by manifest key")
    parser.add_argument("--all", action="store_true", help="ingest the whole manifest")
    parser.add_argument("--list", action="store_true", help="show the corpus manifest")
    parser.add_argument("--stats", action="store_true", help="show knowledge base counts")
    parser.add_argument(
        "--max-sections",
        type=int,
        default=None,
        help="cap sections fetched per act (useful for a quick smoke run)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="seconds between section requests; India Code throttles bursts",
    )
    args = parser.parse_args()

    if args.list:
        print_manifest()
        return 0
    if args.stats:
        print_stats()
        return 0
    if not args.act and not args.all:
        parser.print_help()
        return 1
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
