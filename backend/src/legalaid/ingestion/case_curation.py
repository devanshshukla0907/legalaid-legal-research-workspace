"""Curation for the case-law corpus.

The brief asks for 50-200 carefully selected judgments, not a full case-law engine. Selection
has to be deliberate: the Similar Case Agent is only useful if the corpus actually contains
precedents connected to the kinds of matters the demo covers.

Titles alone are not enough - the dataset's `description` column is empty and only 3 of 856
judgments in 2023 look contract-related by title. So candidates are scored on their headnote
text, which is where e-SCR records the issues and the statutes in play.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Topic weights. Contract and commercial disputes score highest because the demo matter is a
# breach-of-contract claim; procedure and evidence matter because those questions arise in any
# civil matter.
TOPIC_WEIGHTS: dict[str, tuple[int, tuple[str, ...]]] = {
    "contract": (
        10,
        (
            "breach of contract", "contractual", "contract act", "agreement",
            "memorandum of understanding", "mou", "consideration", "specific performance",
            "novation", "rescission",
        ),
    ),
    "compensation": (
        9,
        ("compensation", "damages", "liquidated damages", "penalty", "loss or damage",
         "restitution", "mesne profits"),
    ),
    "commercial": (
        7,
        ("commercial", "supply", "tender", "bid", "sale of goods", "payment", "invoice",
         "recovery of money", "guarantee", "surety"),
    ),
    "arbitration": (
        7,
        ("arbitration", "arbitral", "conciliation", "award", "section 11", "section 34",
         "arbitration agreement"),
    ),
    "civil_procedure": (
        6,
        ("civil procedure", "order vii", "order ii", "plaint", "res judicata", "cause of action",
         "limitation", "decree", "suit was barred", "jurisdiction"),
    ),
    "evidence": (
        5,
        ("evidence act", "admissibility", "electronic record", "burden of proof",
         "documentary evidence", "secondary evidence"),
    ),
    "consumer": (
        5,
        ("consumer protection", "consumer forum", "deficiency in service", "unfair trade practice"),
    ),
    "it_act": (
        4,
        ("information technology act", "electronic signature", "digital signature",
         "intermediary", "cyber"),
    ),
    "fraud": (
        4,
        ("fraud", "misrepresentation", "cheating", "criminal breach of trust", "dishonest"),
    ),
}

# Subject matter deliberately excluded: it is genuinely irrelevant to a commercial practice
# and would dilute both the corpus and the similarity search.
EXCLUDE_PATTERNS = re.compile(
    r"bail application|narcotic|ndps|murder|rape|dowry death|abduction|"
    r"election petition|service tax|excise duty|customs duty|land acquisition|"
    r"tenancy|rent control|matrimonial|divorce|maintenance under section 125|"
    r"disciplinary proceedings|promotion|seniority|pension",
    re.I,
)


@dataclass(slots=True)
class ScoredCandidate:
    row: dict[str, object]
    score: int
    topics: list[str]
    reason: str


def score_text(text: str) -> tuple[int, list[str]]:
    """Score a judgment's relevance to the commercial/contract practice area."""
    lowered = text.lower()
    total = 0
    topics: list[str] = []
    for topic, (weight, terms) in TOPIC_WEIGHTS.items():
        hits = sum(1 for term in terms if term in lowered)
        if hits:
            # Diminishing returns: breadth of topics beats one term repeated.
            total += weight * min(hits, 3)
            topics.append(topic)
    return total, topics


def is_excluded(text: str) -> bool:
    return bool(EXCLUDE_PATTERNS.search(text))


def score_candidate(row: dict[str, object], probe_text: str) -> ScoredCandidate:
    """Score a judgment.

    `probe_text` must be real judgment text - the headnote, or the first pages of the PDF.

    Do **not** pass the dataset's ``raw_html`` column: it is page furniture (language selector,
    disclaimer modal) that is identical across judgments, so every record scores as a match.
    Screening on it reported 450 of 856 judgments as contract-related, which is nonsense.
    """
    title = str(row.get("title") or "")
    haystack = f"{title}\n{probe_text}"
    score, topics = score_text(haystack)

    if is_excluded(haystack):
        score = 0
        reason = "excluded: unrelated subject matter"
    elif score == 0:
        reason = "no matching legal topics"
    else:
        reason = "matched topics: " + ", ".join(topics)

    return ScoredCandidate(row=row, score=score, topics=topics, reason=reason)


def diversify(
    candidates: list[ScoredCandidate], *, limit: int, per_topic_cap: int = 24
) -> list[ScoredCandidate]:
    """Take the best candidates while preventing one topic from dominating.

    Without a cap the corpus fills with near-identical arbitration matters, which makes the
    Similar Case Agent look good on one question and useless on every other.
    """
    ordered = sorted(candidates, key=lambda c: -c.score)
    counts: dict[str, int] = {}
    chosen: list[ScoredCandidate] = []

    for candidate in ordered:
        if len(chosen) >= limit:
            break
        if candidate.score <= 0:
            continue
        primary = candidate.topics[0] if candidate.topics else "other"
        if counts.get(primary, 0) >= per_topic_cap:
            continue
        counts[primary] = counts.get(primary, 0) + 1
        chosen.append(candidate)

    return chosen
