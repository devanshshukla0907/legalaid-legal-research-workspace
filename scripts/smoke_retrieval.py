"""Smoke-test retrieval: does a natural-language legal question find the right provision?

This is the real test of the RAG layer. The headline case is the question Code Llama answered
incorrectly without retrieval ("what does Section 73 of the Indian Contract Act deal with?" ->
it said "sale of goods"). Retrieval must surface the actual provision.

Usage: python scripts/smoke_retrieval.py
"""

from __future__ import annotations

import asyncio

from legalaid.domain.contracts import RetrievalRequest
from legalaid.domain.enums import RetrievalMode
from legalaid.services.retrieval.service import search

QUESTIONS: list[tuple[str, str]] = [
    # (question, section number we expect to see near the top)
    ("What compensation can I claim for loss caused by breach of contract?", "73"),
    ("Is a penalty clause in a contract enforceable, and how much can be recovered?", "74"),
    ("What does Section 73 of the Indian Contract Act deal with?", "73"),
    ("Are electronic records legally recognised as valid?", None),
    ("What makes an agreement void for want of consideration?", None),
]


async def main() -> int:
    failures = 0
    for question, expected_section in QUESTIONS:
        response = await search(
            RetrievalRequest(question=question, mode=RetrievalMode.LAW_SEARCH)
        )
        print("=" * 92)
        print(f"Q: {question}")
        emb = response.query_embedding
        if emb:
            print(
                f"   embedding: {emb.model} {emb.dimensions}d "
                f"in {emb.latency_ms:.0f} ms | preview {emb.preview[:3]}"
            )
        for trace in response.searches:
            if trace.candidate_count:
                print(
                    f"   {trace.collection.value}: {trace.candidate_count} candidates, "
                    f"top_k={trace.top_k}, returned={trace.returned}, "
                    f"search {trace.search_ms:.0f} ms"
                )
        print(f"   total {response.total_ms:.0f} ms, context {response.context_tokens} tokens")

        if not response.chunks:
            print("   !! no results")
            failures += 1
            continue

        for chunk in response.chunks[:4]:
            label = chunk.section or chunk.heading or "-"
            print(f"\n   [{chunk.rank}] {chunk.similarity:.4f}  {label} — {chunk.source_title}")
            print(f"        {chunk.text[:150].replace(chr(10), ' ')}...")
            for reason in chunk.match_reasons[:3]:
                print(f"        · {reason}")

        if expected_section:
            top_sections = [
                (c.section or "") for c in response.chunks[:5]
            ]
            hit = any(expected_section in s for s in top_sections)
            print(f"\n   expected Section {expected_section} in top 5: {'YES' if hit else 'NO'}")
            if not hit:
                failures += 1
        print()

    print("=" * 92)
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
