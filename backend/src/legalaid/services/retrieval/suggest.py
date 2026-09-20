"""Prompt suggestions: static, corpus-grounded prefix matching, no LLM call per keystroke.

The Naman/Karthikey pattern praised in the Week 3/4 feedback was "as the user types, suggest
4-5 relevant prompts". The obvious implementation - re-run the Intake Router on every keystroke
- does not work here: the router is an LLM call that takes seconds (see
`agents/intake_router.py`), so it cannot run per-keystroke without either debouncing so
aggressively the feature feels laggy, or burning a generation slot on every partial word typed.

Instead this builds a small, in-memory suggestion vocabulary directly from the corpus - section
headings, act names, and case titles already stored in SQLite - and does plain prefix/substring
matching against it. That is a few milliseconds, entirely LLM-free, and every suggestion is
*grounded*: it corresponds to something the knowledge base can actually answer, unlike a
hardcoded example list which may or may not still be true as the corpus grows.

Same invalidate-by-count pattern as the BM25 index (`services/retrieval/keyword.py`): the corpus
is small enough that a full rebuild is instant, so there is no incremental maintenance.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from sqlalchemy import select

from legalaid.db.base import session_scope
from legalaid.db.models import Case, LegalProvision


@dataclass(slots=True)
class Suggestion:
    text: str
    kind: str  # "statute" | "case"
    detail: str


@dataclass(slots=True)
class SuggestionIndex:
    entries: list[Suggestion] = field(default_factory=list)
    built_for: tuple[int, int] = (0, 0)

    def search(self, prefix: str, limit: int = 6) -> list[Suggestion]:
        needle = prefix.strip().lower()
        if len(needle) < 2:
            return []
        # Prefix matches (the word the user is currently typing starts a suggestion phrase)
        # rank above substring matches, so typing "leave" surfaces "Leave and licence..." before
        # a suggestion that merely contains "leave" mid-sentence.
        starts: list[Suggestion] = []
        contains: list[Suggestion] = []
        for entry in self.entries:
            haystack = entry.text.lower()
            if haystack.startswith(needle):
                starts.append(entry)
            elif needle in haystack:
                contains.append(entry)
            if len(starts) >= limit:
                break
        combined = (starts + contains)[:limit]
        return combined


_index: SuggestionIndex | None = None
_lock = threading.Lock()


def _question_phrasing(kind: str, act_name: str, section_number: str | None, heading: str | None) -> str:
    """Turn a stored provision into a phrasing a lawyer would actually type as a question."""
    clean_heading = (heading or "").strip().rstrip(".").lower()
    if section_number and clean_heading:
        return f"What does Section {section_number} of the {act_name} provide about {clean_heading}?"
    if section_number:
        return f"What does Section {section_number} of the {act_name} provide for?"
    return f"What does the {act_name} say about {clean_heading or 'this'}?"


def build_index(force: bool = False) -> SuggestionIndex:
    global _index
    with _lock:
        with session_scope() as session:
            provisions = session.scalars(
                select(LegalProvision).where(LegalProvision.section_heading.is_not(None))
            ).all()
            cases = session.scalars(select(Case)).all()

        signature = (len(provisions), len(cases))
        if _index is not None and not force and _index.built_for == signature:
            return _index

        entries: list[Suggestion] = []
        seen_headings: set[str] = set()
        for p in provisions:
            heading = (p.section_heading or "").strip()
            if not heading or heading.lower() in seen_headings:
                continue
            seen_headings.add(heading.lower())
            entries.append(
                Suggestion(
                    text=_question_phrasing("statute", p.act_name, p.section_number, heading),
                    kind="statute",
                    detail=f"Section {p.section_number}, {p.act_name}",
                )
            )

        for c in cases:
            title = (c.case_title or "").strip()
            if not title:
                continue
            entries.append(
                Suggestion(
                    text=f"Is there Supreme Court authority similar to {title}?",
                    kind="case",
                    detail=c.citation or c.case_ref or title,
                )
            )
            if c.keywords:
                topic = str(c.keywords[0]).replace("_", " ")
                entries.append(
                    Suggestion(
                        text=f"Find Supreme Court judgments on {topic}",
                        kind="case",
                        detail=f"seeded from {title}",
                    )
                )

        index = SuggestionIndex(entries=entries, built_for=signature)
        _index = index
        return index


def get_index() -> SuggestionIndex:
    return build_index()


def suggest(prefix: str, limit: int = 6) -> list[Suggestion]:
    return get_index().search(prefix, limit=limit)
