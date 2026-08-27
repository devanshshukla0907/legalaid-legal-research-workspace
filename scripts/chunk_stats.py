"""Chunk size distribution per collection, from live data."""

from __future__ import annotations

from sqlalchemy import select

from legalaid.db.base import session_scope
from legalaid.db.models import Chunk

with session_scope() as session:
    rows = session.scalars(select(Chunk)).all()

groups: dict[str, list[Chunk]] = {}
for row in rows:
    groups.setdefault(row.collection, []).append(row)

print(f"{'collection':<16}{'n':>5}{'min':>7}{'p50':>7}{'p95':>7}{'max':>7}{'avg tok':>9}  strategies")
print("-" * 86)
for name, items in sorted(groups.items()):
    sizes = sorted(c.char_count for c in items)
    tokens = [c.token_count for c in items]
    p50 = sizes[len(sizes) // 2]
    p95 = sizes[int(len(sizes) * 0.95)]
    strategies = ", ".join(sorted({c.chunk_strategy for c in items}))
    print(
        f"{name:<16}{len(sizes):>5}{sizes[0]:>7}{p50:>7}{p95:>7}{sizes[-1]:>7}"
        f"{sum(tokens) // len(tokens):>9}  {strategies}"
    )

allsizes = sorted(c.char_count for c in rows)
print(f"\ntotal chunks: {len(allsizes)}  |  total chars: {sum(allsizes):,}")
print(f"overall: min={allsizes[0]} p50={allsizes[len(allsizes)//2]} max={allsizes[-1]}")
