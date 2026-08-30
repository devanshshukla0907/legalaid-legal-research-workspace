"""Citation extraction and the precedent graph.

The graph retriever currently traverses statutory structure - act contains section, judgment
considers act. It has no notion of one judgment citing another, so "what did the Supreme Court
say after this decision?" is unanswerable by traversal and falls back to embedding similarity,
which does not model precedent at all.

This module extracts citations from judgment text and builds the missing edges.

Two things it deliberately does **not** do:

* It does not treat `CITES` as approval. A judgment that distinguishes or overrules another
  cites it just as much as one that follows it, and conflating those would be worse than having
  no edges - it would let the system present overruled authority as support. Treatment is
  classified separately, from the language around the citation, and reported with the evidence
  so a reader can check it.
* It does not assert that a precedent is good law. That needs subsequent-treatment analysis
  across the whole corpus, and the corpus here is a 2023 slice. `treatment` records what one
  judgment did to another; nothing infers current standing from it.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

# ------------------------------------------------------------------- citation formats
#
# Indian reporter citations are regular enough to extract reliably. Each pattern is anchored on
# the reporter abbreviation so that ordinary numbers in judgment text are not swept up.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # (2014) 7 SCC 123  /  (2014) 7 S.C.C. 123
    ("scc_bracketed", re.compile(r"\((?P<year>1[89]\d{2}|20\d{2})\)\s*(?P<vol>\d{1,3})\s*S\.?C\.?C\.?\s*(?P<page>\d{1,4})", re.I)),
    # AIR 1973 SC 1461
    ("air", re.compile(r"\bA\.?I\.?R\.?\s*(?P<year>1[89]\d{2}|20\d{2})\s*(?P<court>SC|S\.C\.|Bom|Del|Cal|Mad|Kar|Ker|All|P&H|Guj|Raj|MP|AP)\s*(?P<page>\d{1,5})", re.I)),
    # 2023 SCC OnLine SC 1066
    ("scc_online", re.compile(r"(?P<year>20\d{2})\s*S\.?C\.?C\.?\s*OnLine\s*(?P<court>SC|Bom|Del|Cal|Mad|Kar|Ker|All)\s*(?P<page>\d{1,5})", re.I)),
    # (2014) 7 SCR 123
    ("scr", re.compile(r"\((?P<year>1[89]\d{2}|20\d{2})\)\s*(?P<vol>\d{1,3})\s*S\.?C\.?R\.?\s*(?P<page>\d{1,4})", re.I)),
    # 2019 (3) SCALE 45
    ("scale", re.compile(r"(?P<year>1[89]\d{2}|20\d{2})\s*\((?P<vol>\d{1,2})\)\s*SCALE\s*(?P<page>\d{1,4})", re.I)),
    # Civil Appeal No. 1234 of 2019 - not a reporter citation, but identifies a case
    ("case_number", re.compile(r"\b(?P<kind>Civil|Criminal)\s+Appeal\s+No\.?\s*(?P<number>\d{1,6})\s+of\s+(?P<year>1[89]\d{2}|20\d{2})", re.I)),
)

# Statutory references, so a judgment can be linked to the provisions it interprets.
_SECTION_REF = re.compile(
    r"\b(?:section|sections|s\.|ss\.)\s*(?P<numbers>\d+[A-Za-z]{0,2}(?:\s*(?:,|and|to|-)\s*\d+[A-Za-z]{0,2})*)",
    re.I,
)


class Treatment(str, Enum):
    """What the citing judgment did with the cited one."""

    FOLLOWED = "followed"
    RELIED_ON = "relied_on"
    DISTINGUISHED = "distinguished"
    OVERRULED = "overruled"
    REVERSED = "reversed"
    AFFIRMED = "affirmed"
    DOUBTED = "doubted"
    REFERRED = "referred"  # mentioned without a clear stance
    CITED = "cited"  # default: no signal either way


# Ordered by strength. When several cues appear near one citation the strongest wins, because
# "we follow X, distinguishing Y" should not mark X as distinguished.
_TREATMENT_CUES: tuple[tuple[Treatment, tuple[str, ...]], ...] = (
    (Treatment.OVERRULED, ("overrul", "no longer good law", "cannot be considered good law")),
    (Treatment.REVERSED, ("revers", "set aside the judgment", "set aside the impugned")),
    (Treatment.AFFIRMED, ("affirm", "upheld the judgment", "uphold the judgment")),
    (Treatment.DISTINGUISHED, ("distinguish", "inapplicable", "not applicable to the facts",
                               "turned on its own facts")),
    (Treatment.DOUBTED, ("doubt", "with respect, we are unable to agree", "per incuriam")),
    (Treatment.FOLLOWED, ("follow", "bound by", "squarely covered", "settled by")),
    (Treatment.RELIED_ON, ("relied upon", "relied on", "reliance was placed", "placed reliance")),
    (Treatment.REFERRED, ("referred to", "reference was made", "see also", "cf.")),
)

# How much text either side of a citation is inspected for treatment language. A citation's
# treatment is normally stated in the same sentence or the one before it.
CUE_WINDOW = 260


@dataclass(frozen=True, slots=True)
class Citation:
    """One extracted citation, normalised so two spellings of it match."""

    raw: str
    kind: str
    normalised: str
    year: int | None
    position: int

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "kind": self.kind,
            "normalised": self.normalised,
            "year": self.year,
            "position": self.position,
        }


@dataclass
class CitationEdge:
    source_case_id: str
    target_citation: str
    target_case_id: str | None
    treatment: Treatment
    evidence: str
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "source_case_id": self.source_case_id,
            "target_citation": self.target_citation,
            "target_case_id": self.target_case_id,
            "treatment": self.treatment.value,
            "evidence": self.evidence,
            "resolved": self.resolved,
        }


def normalise_citation(kind: str, match: re.Match[str]) -> str:
    """Canonical form, so "(2014) 7 SCC 123" and "(2014) 7 S.C.C. 123" are one key."""
    groups = match.groupdict()
    year = groups.get("year", "")
    if kind == "case_number":
        return f"{groups.get('kind', '').lower()}-appeal:{groups.get('number')}:{year}"
    court = (groups.get("court") or "SC").upper().replace(".", "")
    volume = groups.get("vol") or ""
    page = groups.get("page") or ""
    reporter = {
        "scc_bracketed": "SCC",
        "scc_online": "SCCONLINE",
        "scr": "SCR",
        "scale": "SCALE",
        "air": "AIR",
    }[kind]
    if reporter == "AIR":
        return f"air:{year}:{court}:{page}"
    return f"{reporter.lower()}:{year}:{volume}:{page}"


def extract_citations(text: str) -> list[Citation]:
    """Every reporter citation in a judgment, de-duplicated by canonical form."""
    found: dict[str, Citation] = {}
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text or ""):
            year_raw = match.groupdict().get("year")
            citation = Citation(
                raw=match.group(0).strip(),
                kind=kind,
                normalised=normalise_citation(kind, match),
                year=int(year_raw) if year_raw and year_raw.isdigit() else None,
                position=match.start(),
            )
            # Keep the first occurrence; its surrounding text is usually where the court
            # explains what it is doing with the authority.
            found.setdefault(citation.normalised, citation)
    return sorted(found.values(), key=lambda c: c.position)


def classify_treatment(text: str, position: int, *, window: int = CUE_WINDOW) -> tuple[Treatment, str]:
    """Infer treatment from the language around a citation, and return the evidence."""
    start = max(0, position - window)
    end = min(len(text), position + window)
    context = text[start:end]
    lowered = context.lower()
    for treatment, cues in _TREATMENT_CUES:
        for cue in cues:
            index = lowered.find(cue)
            if index != -1:
                snippet_start = max(0, index - 60)
                return treatment, context[snippet_start : index + len(cue) + 90].strip()
    return Treatment.CITED, context[:150].strip()


def extract_section_references(text: str) -> list[str]:
    """Section numbers a judgment discusses, expanded across ranges and lists."""
    out: list[str] = []
    for match in _SECTION_REF.finditer(text or ""):
        numbers = match.group("numbers")
        for part in re.split(r"\s*(?:,|and)\s*", numbers):
            part = part.strip()
            if not part:
                continue
            # "73 to 75" and "73-75" both expand, but only for plain numeric endpoints.
            span = re.match(r"^(\d+)\s*(?:to|-)\s*(\d+)$", part)
            if span:
                low, high = int(span.group(1)), int(span.group(2))
                if 0 < high - low <= 30:
                    out.extend(str(n) for n in range(low, high + 1))
                    continue
            token = re.match(r"^\d+[A-Za-z]{0,2}$", part)
            if token:
                out.append(part.upper())
    # Preserve first-seen order while de-duplicating.
    seen: set[str] = set()
    ordered: list[str] = []
    for number in out:
        if number not in seen:
            seen.add(number)
            ordered.append(number)
    return ordered


def build_edges(
    cases: list[tuple[str, str, list[str]]],
) -> tuple[list[CitationEdge], dict[str, int]]:
    """Build citation edges across the corpus.

    `cases` is (case_id, full_text, known_citations). `known_citations` are the citations the
    case is *published under*, which is how a citation found in one judgment is resolved to
    another case in the corpus.

    Returns the edges plus a stats dict, including how many citations could not be resolved -
    an unresolved citation is not a failure, it simply points outside a 47-judgment corpus, and
    reporting it honestly matters more than silently dropping it.
    """
    # Reverse index from every canonical citation form to the case it identifies.
    by_citation: dict[str, str] = {}
    for case_id, _text, citations in cases:
        for citation in citations:
            for kind, pattern in _PATTERNS:
                match = pattern.search(citation or "")
                if match:
                    by_citation[normalise_citation(kind, match)] = case_id

    edges: list[CitationEdge] = []
    stats: dict[str, int] = defaultdict(int)
    for case_id, text, _citations in cases:
        for citation in extract_citations(text):
            target = by_citation.get(citation.normalised)
            if target == case_id:
                continue  # a judgment citing its own reported citation
            treatment, evidence = classify_treatment(text, citation.position)
            edges.append(
                CitationEdge(
                    source_case_id=case_id,
                    target_citation=citation.raw,
                    target_case_id=target,
                    treatment=treatment,
                    evidence=evidence,
                    resolved=target is not None,
                )
            )
            stats["citations_found"] += 1
            stats["resolved" if target else "unresolved"] += 1
            stats[f"treatment_{treatment.value}"] += 1
    return edges, dict(stats)
