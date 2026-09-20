"""Ingest this repository's own source code into the CODE collection.

Closes the Exercise 6 gap directly: previously the legal RAG system scored 0/61 relevant files
retrieved for codebase questions because no code was indexed anywhere. This walks
backend/src/legalaid and frontend/src, chunks every file by function/class boundary, embeds the
chunks, and indexes them into their own FAISS collection - separate from every legal collection,
so a codebase question is never answered by statutory text scoring highest only by default.

Usage:
  python scripts/ingest_codebase.py --all
  python scripts/ingest_codebase.py --all --max-files 50
  python scripts/ingest_codebase.py --reset
  python scripts/ingest_codebase.py --stats
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from legalaid.db.base import create_all
from legalaid.services.knowledge.ingest_codebase import (
    codebase_stats,
    ingest_codebase,
    reset_code,
)


def print_stats() -> None:
    stats = codebase_stats()
    print(f"files indexed : {stats['files']}")
    print(f"chunks        : {stats['chunks']}")
    print(f"vectors       : {stats['vectors']}")


async def run(max_files: int | None) -> int:
    create_all()
    report = await ingest_codebase(max_files=max_files)
    print(report.summary())
    if report.failures:
        for failure in report.failures[:10]:
            print(f"  ! {failure}")
    print()
    print_stats()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="ingest backend/src and frontend/src")
    parser.add_argument("--max-files", type=int, default=None, help="cap total files ingested")
    parser.add_argument("--reset", action="store_true", help="remove all indexed source code")
    parser.add_argument("--stats", action="store_true", help="show current CODE collection counts")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return 0
    if args.reset:
        removed = reset_code()
        print(f"removed {removed} code chunks")
        return 0
    if not args.all:
        parser.print_help()
        return 1
    return asyncio.run(run(args.max_files))


if __name__ == "__main__":
    sys.exit(main())
