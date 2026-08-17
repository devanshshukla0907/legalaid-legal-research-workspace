"""Reciprocal Rank Fusion for hybrid retrieval.

Dense and BM25 scores are not comparable: cosine sits in [-1, 1] while BM25 is unbounded and
corpus-dependent. Normalising them into a weighted sum requires picking a scale that shifts
whenever the corpus changes, which is fragile.

Reciprocal Rank Fusion sidesteps that by combining *ranks* rather than scores:

    RRF(d) = sum over retrievers of  weight / (k + rank(d))

It needs no score calibration, is robust when one retriever returns nothing useful, and
rewards documents that both retrievers agree on. `k` damps the influence of top ranks so a
single retriever cannot dominate on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Standard damping constant from the original RRF paper. Larger k flattens the curve, so
# agreement between retrievers matters more than one retriever's confidence.
RRF_K = 60


@dataclass(slots=True)
class FusedHit:
    chunk_id: str
    fused_score: float
    dense_rank: int | None = None
    dense_similarity: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    matched_terms: list[str] = field(default_factory=list)

    @property
    def retrievers(self) -> list[str]:
        found: list[str] = []
        if self.dense_rank is not None:
            found.append("dense")
        if self.keyword_rank is not None:
            found.append("keyword")
        return found

    @property
    def agreed(self) -> bool:
        """Both retrievers found it - the strongest signal available."""
        return self.dense_rank is not None and self.keyword_rank is not None


def reciprocal_rank_fusion(
    dense: list[tuple[str, float]],
    keyword: list[tuple[str, float, list[str]]],
    *,
    dense_weight: float = 1.0,
    keyword_weight: float = 0.6,
    k: int = RRF_K,
    limit: int | None = None,
) -> list[FusedHit]:
    """Fuse dense and keyword result lists, both ordered best-first.

    `keyword_weight` is below 1.0 deliberately: semantic matching is the primary signal for
    natural-language legal questions, with BM25 acting as a precision correction for exact
    identifiers like "Section 73". Both are configurable.
    """
    hits: dict[str, FusedHit] = {}

    for rank, (chunk_id, similarity) in enumerate(dense, start=1):
        hit = hits.setdefault(chunk_id, FusedHit(chunk_id=chunk_id, fused_score=0.0))
        hit.dense_rank = rank
        hit.dense_similarity = similarity
        hit.fused_score += dense_weight / (k + rank)

    for rank, (chunk_id, score, terms) in enumerate(keyword, start=1):
        hit = hits.setdefault(chunk_id, FusedHit(chunk_id=chunk_id, fused_score=0.0))
        hit.keyword_rank = rank
        hit.keyword_score = score
        hit.matched_terms = terms
        hit.fused_score += keyword_weight / (k + rank)

    ordered = sorted(
        hits.values(),
        # Ties broken toward documents both retrievers agree on, then dense similarity.
        key=lambda h: (-h.fused_score, not h.agreed, -(h.dense_similarity or 0.0)),
    )
    return ordered[:limit] if limit else ordered


def describe_fusion(hit: FusedHit) -> list[str]:
    """Human-readable explanation of how a hit was ranked, for the Retrieval Inspector."""
    reasons: list[str] = []
    if hit.agreed:
        reasons.append(
            f"Found by BOTH retrievers - dense rank {hit.dense_rank}, "
            f"keyword rank {hit.keyword_rank}"
        )
    elif hit.dense_rank is not None:
        reasons.append(f"Semantic match only (dense rank {hit.dense_rank})")
    elif hit.keyword_rank is not None:
        reasons.append(f"Exact-term match only (keyword rank {hit.keyword_rank})")

    if hit.matched_terms:
        pretty = [t.replace("section_", "Section ").replace("_", " ") for t in hit.matched_terms]
        reasons.append(f"Keyword terms: {', '.join(pretty[:8])}")
    return reasons
