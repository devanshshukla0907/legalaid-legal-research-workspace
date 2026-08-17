"""Why-was-this-retrieved explanations.

Similarity alone is not an explanation. A lawyer inspecting a result needs to see which legal
concepts and terms actually connect the question to the retrieved provision, and the brief
requires the UI to answer "Why was this retrieved?" for every major result.

These explanations are derived from the question and the retrieved text - they are lexical
evidence surfaced alongside the vector score, never a model-generated rationalisation. That
distinction matters: an LLM-written justification could sound convincing for a bad match.
"""

from __future__ import annotations

import re

# Legal concept vocabulary. Grouping surface terms under a concept lets the UI say
# "matched legal concepts: breach, compensation" rather than listing raw word overlaps.
LEGAL_CONCEPTS: dict[str, tuple[str, ...]] = {
    "breach": ("breach", "broken", "violate", "violation", "non-performance", "default"),
    "compensation": ("compensation", "damages", "loss", "indemnity", "reparation", "recompense"),
    "contract": ("contract", "agreement", "covenant", "promise", "consideration", "bargain"),
    "penalty": ("penalty", "liquidated", "stipulated", "forfeiture", "penal"),
    "termination": ("termination", "rescind", "rescission", "repudiation", "cancel", "revoke"),
    "notice": ("notice", "notification", "intimation", "summons", "communication"),
    "evidence": ("evidence", "proof", "admissible", "document", "witness", "testimony"),
    "electronic_records": (
        "electronic", "digital", "computer", "data", "record", "signature", "e-mail",
    ),
    "jurisdiction": ("jurisdiction", "court", "forum", "venue", "cognizance", "tribunal"),
    "limitation": ("limitation", "time-barred", "prescribed period", "delay", "laches"),
    "specific_performance": ("specific performance", "injunction", "mandatory", "restrain"),
    "fraud": ("fraud", "misrepresentation", "cheating", "deceit", "dishonest", "mislead"),
    "liability": ("liable", "liability", "responsible", "obligation", "duty"),
    "remedy": ("remedy", "relief", "restitution", "recovery", "claim"),
    "interest": ("interest", "rate", "accrued", "payable"),
    "arbitration": ("arbitration", "arbitral", "conciliation", "award", "tribunal"),
    "consumer": ("consumer", "goods", "service", "deficiency", "unfair trade"),
    "criminal_liability": (
        "punish", "punishment", "imprisonment", "fine", "offence", "criminal", "abetment",
    ),
}

_STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to for with by from as is are was were be been being
    this that these those it its which who whom whose what when where how why not no nor
    any all some such other than then there here shall may can will would should could
    have has had do does did done under over between within without upon into about
    i me my we our you your he him his she her they them their
    """.split()
)

_WORD_RE = re.compile(r"[a-z][a-z\-']+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def concepts_in(text: str) -> set[str]:
    """Legal concepts present in a piece of text."""
    lowered = text.lower()
    found: set[str] = set()
    for concept, terms in LEGAL_CONCEPTS.items():
        if any(term in lowered for term in terms):
            found.add(concept)
    return found


def significant_terms(text: str, *, limit: int = 40) -> set[str]:
    """Content words, stopwords and very short tokens removed."""
    return {w for w in _words(text) if len(w) > 3 and w not in _STOPWORDS}


def explain_match(
    *,
    question: str,
    chunk_text: str,
    similarity: float,
    section: str | None = None,
    source_title: str | None = None,
) -> tuple[list[str], list[str]]:
    """Return (reasons, matched_terms) explaining why a chunk was retrieved."""
    shared_concepts = sorted(concepts_in(question) & concepts_in(chunk_text))
    question_terms = significant_terms(question)
    chunk_terms = significant_terms(chunk_text)
    overlap = sorted(question_terms & chunk_terms)

    reasons: list[str] = []

    band = (
        "Very high" if similarity >= 0.85
        else "High" if similarity >= 0.7
        else "Moderate" if similarity >= 0.55
        else "Weak"
    )
    reasons.append(f"{band} semantic similarity to the question ({similarity:.4f})")

    if shared_concepts:
        pretty = ", ".join(c.replace("_", " ") for c in shared_concepts)
        reasons.append(f"Shared legal concepts: {pretty}")

    if overlap:
        reasons.append(f"Overlapping terms: {', '.join(overlap[:8])}")

    # An explicit section reference in the question is strong, non-semantic evidence.
    if section:
        for match in re.finditer(r"section\s+(\d+[A-Za-z]{0,2})", question, re.I):
            if match.group(1).lower() in section.lower():
                reasons.append(f"Question explicitly names {section}")
                break

    if source_title and source_title.lower() in question.lower():
        reasons.append(f"Question names the source: {source_title}")

    if not shared_concepts and not overlap:
        reasons.append(
            "Matched on overall meaning rather than shared wording - verify relevance"
        )

    return reasons, overlap[:12]
