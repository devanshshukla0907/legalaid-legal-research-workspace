"""Compare dense-only, keyword-only and hybrid+graph retrieval on the same questions.

This is the evidence that hybrid retrieval is worth its complexity. It prints which retrievers
found each hit so the contribution of each is visible rather than asserted.

Usage: python scripts/smoke_hybrid.py
"""

from __future__ import annotations

import asyncio

from legalaid.config import get_settings
from legalaid.domain.contracts import RetrievalRequest
from legalaid.domain.enums import RetrievalMode
from legalaid.services.retrieval.graph import graph_stats
from legalaid.services.retrieval.keyword import get_index
from legalaid.services.retrieval.service import search

QUESTIONS = [
    "What does Section 73 of the Indian Contract Act deal with?",
    "What compensation can I claim for loss caused by breach of contract?",
    "Is a penalty clause enforceable under Section 74?",
    "Order VII Rule 11 rejection of plaint",
]


async def run_mode(question: str, *, keyword: bool, graph: bool) -> None:
    settings = get_settings()
    settings.retrieval_enable_keyword = keyword
    settings.retrieval_enable_graph = graph
    label = (
        "hybrid+graph" if keyword and graph
        else "hybrid" if keyword
        else "dense only"
    )
    response = await search(
        RetrievalRequest(question=question, mode=RetrievalMode.LAW_SEARCH, top_k=5)
    )
    print(f"\n  --- {label} ---")
    for chunk in response.chunks[:5]:
        retrievers = ",".join(chunk.metadata.get("retrievers") or [])
        print(
            f"    [{chunk.rank}] sim={chunk.similarity:.4f} "
            f"fused={chunk.metadata.get('fused_score')} via={retrievers:<20} "
            f"{(chunk.section or chunk.heading or '-')[:26]} | {chunk.source_title[:34]}"
        )


async def main() -> None:
    index = get_index()
    print(f"BM25 index: {len(index.docs)} docs, {len(index.df)} terms")
    print(f"graph: {graph_stats()}")

    for question in QUESTIONS:
        print("\n" + "=" * 96)
        print(f"Q: {question}")
        await run_mode(question, keyword=False, graph=False)
        await run_mode(question, keyword=True, graph=False)
        await run_mode(question, keyword=True, graph=True)


if __name__ == "__main__":
    asyncio.run(main())
