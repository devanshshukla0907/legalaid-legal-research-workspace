"""Keyword (BM25) retrieval — the lexical half of hybrid search.

Dense vectors are good at meaning and bad at exact identifiers. Legal research needs both:
"Section 73" and "Order VII Rule 11" are precise tokens that must match exactly, and an
embedding will happily return a semantically similar but *legally different* provision.

Observed on this corpus before hybrid search existed: the query "What does Section 73 of the
Indian Contract Act deal with?" put three different s.73 chunks on top, but a repealed s.259
whose repeal note merely *mentioned* s.73 also scored 0.7975 — dense similarity could not tell
the difference between a provision and a reference to it. BM25 scores exact section tokens far
more sharply, so fusing the two rankings is measurably better than either alone.

BM25 is implemented directly (about 40 lines) rather than adding a dependency, and the index is
built from the chunk rows already in the database.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import select

from legalaid.core.logging import get_logger
from legalaid.db.base import session_scope
from legalaid.db.models import Chunk
from legalaid.domain.enums import Collection

logger = get_logger(__name__)

# Okapi BM25 defaults; k1 controls term-frequency saturation, b length normalisation.
K1 = 1.5
B = 0.75

# Underscore MUST be a word character here. Citation normalisation rewrites "Section 73" to
# the single token "section_73"; a token pattern that excluded "_" split it straight back into
# ["section", "73"], which destroyed the whole point - "section" appears in 71% of the corpus
# and "73" in 1.4%, so the precise part of the citation was swamped by the generic part.
_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:[a-z0-9_\-']*[a-z0-9_])?")

# Citation tokens produced by normalisation below. A document containing the exact provision
# the user named is qualitatively more relevant than one merely sharing generic legal words,
# but plain BM25 cannot express that: it rewards short documents, so a 38-token chunk matching
# only {indian, contract, act, deal} outscored chunks that actually contained section_73.
# Citation matches therefore get an explicit boost.
_CITATION_PREFIXES = ("section_", "order_", "rule_", "article_")
CITATION_BOOST = 3.0


def is_citation_token(term: str) -> bool:
    return term.startswith(_CITATION_PREFIXES)

_STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to for with by from as is are was were be been being
    this that these those it its which who whom whose what when where how why not no nor
    any all some such other than then there here shall may can will would should could
    have has had do does did done under over between within without upon into about
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with legal citation forms normalised.

    "Section 73", "S. 73" and "sec 73" must all produce the same token, otherwise exact-match
    scoring depends on how the draftsman happened to abbreviate.
    """
    lowered = text.lower()

    # "Order VII Rule 11" is one citation; the order and rule numbers mean nothing apart, so
    # emit a combined token as well as the parts.
    lowered = re.sub(
        r"\border\s+([ivxlcdm]+)\s+rule\s+(\d+[a-z]{0,2})\b",
        r" order_\1_rule_\2 order_\1 rule_\2 ",
        lowered,
    )
    # Normalise section references to a single canonical token so "Section 73", "S. 73" and
    # "sec 73" are the same term.
    lowered = re.sub(
        r"\b(?:sections?|sec\.?|s\.)\s*(\d+[a-z]{0,2})\b", r" section_\1 ", lowered
    )
    lowered = re.sub(r"\border\s+([ivxlcdm]+)\b", r" order_\1 ", lowered)
    lowered = re.sub(r"\brule\s*(\d+[a-z]{0,2})\b", r" rule_\1 ", lowered)
    lowered = re.sub(r"\b(?:articles?|art\.)\s*(\d+[a-z]{0,2})\b", r" article_\1 ", lowered)

    tokens = _TOKEN_RE.findall(lowered)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


@dataclass(slots=True)
class KeywordHit:
    chunk_id: str
    score: float
    matched_terms: list[str]


@dataclass(slots=True)
class _Doc:
    chunk_id: str
    collection: str
    length: int
    freqs: Counter[str]
    matter_id: str | None = None


@dataclass(slots=True)
class BM25Index:
    """In-memory BM25 index over stored chunks, one logical index per collection."""

    docs: list[_Doc] = field(default_factory=list)
    df: Counter[str] = field(default_factory=Counter)
    avg_length: float = 0.0
    built_for: int = 0

    def search(
        self,
        query: str,
        *,
        collection: Collection,
        top_k: int,
        matter_id: str | None = None,
    ) -> tuple[list[KeywordHit], int]:
        """Return (hits, candidate_count) for one collection."""
        terms = tokenize(query)
        if not terms:
            return [], 0

        pool = [d for d in self.docs if d.collection == collection.value]
        if matter_id is not None:
            pool = [d for d in pool if d.matter_id == matter_id]
        if not pool:
            return [], 0

        total = len(pool)
        scored: list[KeywordHit] = []
        for doc in pool:
            score = 0.0
            matched: list[str] = []
            for term in terms:
                freq = doc.freqs.get(term, 0)
                if not freq:
                    continue
                # IDF with the standard +0.5 smoothing, floored so common terms cannot go
                # negative and subtract from a genuine match.
                idf = max(
                    0.0,
                    math.log((total - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5) + 1.0),
                )
                denom = freq + K1 * (1 - B + B * (doc.length / (self.avg_length or 1.0)))
                contribution = idf * (freq * (K1 + 1)) / (denom or 1.0)
                # An exact statutory citation is the strongest lexical evidence there is.
                if is_citation_token(term):
                    contribution *= CITATION_BOOST
                score += contribution
                matched.append(term)
            if score > 0:
                scored.append(KeywordHit(doc.chunk_id, score, matched))

        scored.sort(key=lambda h: -h.score)
        return scored[:top_k], total


_index: BM25Index | None = None
_lock = threading.Lock()


def build_index(force: bool = False) -> BM25Index:
    """Build (or rebuild) the BM25 index from the chunks table.

    The corpus is small enough (<1k chunks) that a full in-memory rebuild is instant, so the
    index is invalidated by chunk count rather than maintained incrementally.
    """
    global _index
    with _lock:
        with session_scope() as session:
            rows = session.scalars(select(Chunk)).all()
            payload = [
                (r.id, r.collection, r.text, (r.chunk_metadata or {}).get("matter_id"))
                for r in rows
            ]

        if (
            _index is not None
            and not force
            and _index.built_for == len(payload)
        ):
            return _index

        index = BM25Index()
        for chunk_id, collection, text, matter_id in payload:
            tokens = tokenize(text)
            if not tokens:
                continue
            freqs = Counter(tokens)
            index.docs.append(
                _Doc(
                    chunk_id=chunk_id,
                    collection=collection,
                    length=len(tokens),
                    freqs=freqs,
                    matter_id=str(matter_id) if matter_id else None,
                )
            )
            for term in freqs:
                index.df[term] += 1

        index.avg_length = (
            sum(d.length for d in index.docs) / len(index.docs) if index.docs else 0.0
        )
        index.built_for = len(payload)
        _index = index
        logger.info(
            "bm25_index_built", documents=len(index.docs), vocabulary=len(index.df)
        )
        return index


def get_index() -> BM25Index:
    return build_index()


def reset_index() -> None:
    global _index
    with _lock:
        _index = None
