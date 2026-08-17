"""Verify CASE_LAW retrieval finds relevant precedents for matter-style questions."""

from __future__ import annotations

import asyncio

from legalaid.domain.contracts import RetrievalRequest
from legalaid.domain.enums import RetrievalMode
from legalaid.services.retrieval.service import search

QUESTIONS = [
    "Which cases deal with rejection of a plaint and specific performance of an agreement?",
    "Precedents on arbitration agreements and appointment of an arbitrator",
    "Cases about compensation for breach of a commercial contract",
]


async def main() -> None:
    for question in QUESTIONS:
        response = await search(
            RetrievalRequest(question=question, mode=RetrievalMode.CASE_SEARCH, top_k=4)
        )
        print("=" * 96)
        print(f"Q: {question}")
        for trace in response.searches:
            print(
                f"   {trace.collection.value}: {trace.candidate_count} candidates -> "
                f"{trace.returned} returned in {trace.search_ms:.0f} ms"
            )
        if not response.chunks:
            print("   !! no results")
            continue
        for chunk in response.chunks:
            meta = chunk.metadata
            print(f"\n   [{chunk.rank}] sim={chunk.similarity:.4f} via={','.join(meta.get('retrievers') or [])}")
            print(f"        {chunk.source_title[:70]}")
            print(f"        citation: {meta.get('citation')} | outcome: {meta.get('outcome')}")
            print(f"        segment : {chunk.section or chunk.heading}")
            print(f"        text    : {chunk.text[:160].replace(chr(10), ' ')}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
