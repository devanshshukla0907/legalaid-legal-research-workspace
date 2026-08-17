"""Why is BM25 not discriminating? Inspect tokenisation and per-term scores."""

from __future__ import annotations

from legalaid.domain.enums import Collection
from legalaid.services.retrieval.keyword import get_index, tokenize

QUERIES = [
    "What does Section 73 of the Indian Contract Act deal with?",
    "Order VII Rule 11 rejection of plaint",
]


def main() -> None:
    index = get_index()
    print(f"docs={len(index.docs)} vocab={len(index.df)} avg_len={index.avg_length:.1f}")

    for query in QUERIES:
        terms = tokenize(query)
        print(f"\nQ: {query}")
        print(f"  tokens: {terms}")
        print("  document frequency of each token:")
        for term in terms:
            df = index.df.get(term, 0)
            print(f"    {term:<20} df={df:<5} ({df / max(1, len(index.docs)):.1%} of corpus)")

        hits, total = index.search(query, collection=Collection.LEGAL_ACTS, top_k=6)
        print(f"  BM25 top hits (of {total} candidates):")
        for rank, hit in enumerate(hits, 1):
            doc = next(d for d in index.docs if d.chunk_id == hit.chunk_id)
            print(
                f"    [{rank}] score={hit.score:6.3f} len={doc.length:<5} "
                f"matched={hit.matched_terms}"
            )


if __name__ == "__main__":
    main()
