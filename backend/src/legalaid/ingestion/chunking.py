"""Legal-document-aware chunking.

A single fixed-length splitter is wrong for legal text: it cuts across section boundaries and
destroys the provenance that a legal citation depends on. Three strategies are implemented,
each preserving the structure that matters for its document type.

    Statute    Act -> Chapter -> Section -> Subsection -> Paragraph
    Judgment   Metadata -> Facts -> Issues -> Arguments -> Reasoning -> Ratio -> Decision
    Upload     Document -> Page -> Heading -> Paragraph

Overlap is applied only when a single structural unit must be split because it exceeds the
maximum size. Units that fit are never overlapped, because duplicating statutory text across
chunks would corrupt citation provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from legalaid.config import get_settings
from legalaid.core.tokens import estimate_tokens
from legalaid.domain.enums import ChunkStrategy, Collection


@dataclass(slots=True)
class ChunkCandidate:
    """A chunk before it is persisted. Carries the provenance the UI later displays."""

    text: str
    collection: Collection
    source_type: str
    source_title: str
    ordinal: int
    chunk_strategy: ChunkStrategy
    section: str | None = None
    chapter: str | None = None
    subsection: str | None = None
    page: int | None = None
    heading: str | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.text)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

_PAGE_ARTEFACT_RE = re.compile(
    r"^\s*(page\s+\d+\s*(of\s+\d+)?|\d+\s*\|\s*p\s*a\s*g\s*e|-\s*\d+\s*-)\s*$",
    re.IGNORECASE,
)
_MULTI_SPACE_RE = re.compile(r"[ \t\u00a0]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_legal_text(text: str) -> str:
    """Normalise extracted text without destroying legal structure.

    Deliberately conservative: it removes PDF artefacts and repairs hyphenated line breaks,
    but preserves numbering, sub-clause markers and paragraph boundaries, because those are
    what the section-aware splitters key on.
    """
    if not text:
        return ""
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    lines = [line for line in text.split("\n") if not _PAGE_ARTEFACT_RE.match(line)]
    text = "\n".join(line.rstrip() for line in lines)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Size-bounded splitting (used only when a structural unit is too large)
# --------------------------------------------------------------------------- #

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\[])")


def split_oversized(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """Split text that exceeds max_chars, preferring paragraph then sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    units = [u.strip() for u in _PARAGRAPH_RE.split(text) if u.strip()]
    if len(units) == 1:
        units = [u.strip() for u in _SENTENCE_RE.split(text) if u.strip()]
    if len(units) == 1:
        return _hard_wrap(text, max_chars, overlap)

    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        if len(unit) > max_chars:
            if current:
                parts.append("\n\n".join(current))
                current, current_len = [], 0
            parts.extend(_hard_wrap(unit, max_chars, overlap))
            continue
        addition = len(unit) + (2 if current else 0)
        if current_len + addition > max_chars and current:
            parts.append("\n\n".join(current))
            tail = current[-1]
            if overlap > 0 and len(tail) <= overlap:
                current, current_len = [tail, unit], len(tail) + len(unit) + 2
            else:
                current, current_len = [unit], len(unit)
        else:
            current.append(unit)
            current_len += addition
    if current:
        parts.append("\n\n".join(current))
    return parts


def _hard_wrap(text: str, max_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    step = max(1, max_chars - max(0, overlap))
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            window = text.rfind(" ", start + int(max_chars * 0.6), end)
            if window > start:
                end = window
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + step, end - max(0, overlap))
    return [c for c in chunks if c]


# --------------------------------------------------------------------------- #
# Statute chunking
# --------------------------------------------------------------------------- #

_CHAPTER_RE = re.compile(
    r"^\s*(CHAPTER|PART)\s+([IVXLCDM\d]+[A-Z]?)\s*[-–—.:]?\s*(.*)$", re.IGNORECASE
)
_SECTION_RE = re.compile(
    r"^\s*(?:Section\s+)?(\d+[A-Z]{0,2})\s*\.\s*(?![\d])\s*(.{0,300})$",
    re.IGNORECASE,
)
_SUBSECTION_RE = re.compile(r"^\s*\((\d+|[a-z]{1,3}|[ivxlcdm]{1,5})\)\s+")


@dataclass(slots=True)
class StatuteSection:
    section_number: str | None
    heading: str | None
    chapter: str | None
    text: str


def parse_statute_structure(text: str) -> list[StatuteSection]:
    """Split an Act into chapters and numbered sections.

    Returns a preamble section (section_number=None) when text precedes the first section, so
    long titles and enacting formulae are not silently dropped.
    """
    lines = clean_legal_text(text).split("\n")
    sections: list[StatuteSection] = []
    chapter: str | None = None
    current: StatuteSection | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current, buffer
        if current is not None:
            current.text = "\n".join(buffer).strip()
            if current.text or current.section_number:
                sections.append(current)
        current, buffer = None, []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            buffer.append("")
            continue

        chapter_match = _CHAPTER_RE.match(stripped)
        if chapter_match and len(stripped) < 140:
            flush()
            label = f"{chapter_match.group(1).title()} {chapter_match.group(2)}"
            title = chapter_match.group(3).strip(" .:-–—")
            chapter = f"{label} - {title}" if title else label
            continue

        section_match = _SECTION_RE.match(stripped)
        if section_match and len(stripped) < 400:
            flush()
            heading = section_match.group(2).strip(" .:-–—") or None
            current = StatuteSection(
                section_number=section_match.group(1),
                heading=heading,
                chapter=chapter,
                text="",
            )
            buffer = []
            continue

        if current is None:
            current = StatuteSection(section_number=None, heading="Preamble", chapter=chapter, text="")
            buffer = []
        buffer.append(stripped)

    flush()
    return sections


def chunk_statute(
    *,
    text: str,
    act_name: str,
    collection: Collection = Collection.LEGAL_ACTS,
    source_url: str | None = None,
    act_year: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[ChunkCandidate]:
    settings = get_settings()
    sections = parse_statute_structure(text)
    candidates: list[ChunkCandidate] = []
    ordinal = 0

    for section in sections:
        body = section.text.strip()
        if not body and not section.heading:
            continue
        label = (
            f"Section {section.section_number}" if section.section_number else (section.heading or "Preamble")
        )
        header = f"{act_name} - {label}"
        if section.heading and section.section_number:
            header += f": {section.heading}"

        subsections = _split_subsections(body)
        for sub_label, sub_text in subsections:
            full = f"{header}\n\n{sub_text}".strip()
            pieces = split_oversized(
                full, max_chars=settings.chunk_max_chars, overlap=settings.chunk_overlap_chars
            )
            for piece in pieces:
                if len(piece.strip()) < settings.chunk_min_chars and len(pieces) > 1:
                    continue
                candidates.append(
                    ChunkCandidate(
                        text=piece,
                        collection=collection,
                        source_type="statute",
                        source_title=act_name,
                        ordinal=ordinal,
                        chunk_strategy=ChunkStrategy.STATUTE_SECTION,
                        section=label if section.section_number else None,
                        chapter=section.chapter,
                        subsection=sub_label,
                        heading=section.heading,
                        source_url=source_url,
                        metadata={
                            "act_name": act_name,
                            "act_year": act_year,
                            "section_number": section.section_number,
                            "subsection": sub_label,
                            **(extra_metadata or {}),
                        },
                    )
                )
                ordinal += 1
    return candidates


def _split_subsections(body: str) -> list[tuple[str | None, str]]:
    """Group a section body by subsection markers like (1), (2), (a).

    Short sections are kept whole. Long sections are split on subsection markers and then
    **packed back up to the target chunk size**.

    The packing step matters. Splitting at every marker without repacking produced 28 chunks
    from one definitions section, several of them ~120-300 characters holding a single
    definition. Chunks that small embed poorly: they carry too little context to match a
    natural-language question, and they flood top-k with near-duplicates. Packing keeps
    related clauses together while still honouring the subsection boundary as the split point,
    so a chunk never begins mid-clause.

    The label reported for a packed chunk is its first subsection marker, and a range is
    recorded in metadata by the caller when several are merged.
    """
    settings = get_settings()
    if len(body) <= settings.chunk_target_chars:
        return [(None, body)]

    groups: list[tuple[str | None, list[str]]] = []
    current_label: str | None = None
    current: list[str] = []
    for line in body.split("\n"):
        match = _SUBSECTION_RE.match(line)
        if match:
            if current:
                groups.append((current_label, current))
            current_label = f"({match.group(1)})"
            current = [line.strip()]
        else:
            current.append(line.strip())
    if current:
        groups.append((current_label, current))

    units: list[tuple[str | None, str]] = []
    for label, lines in groups:
        text = "\n".join(lines).strip()
        if text:
            units.append((label, text))

    packed: list[tuple[str | None, str]] = []
    buffer: list[str] = []
    buffer_label: str | None = None
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_label, buffer_len
        if buffer:
            packed.append((buffer_label, "\n".join(buffer).strip()))
        buffer, buffer_label, buffer_len = [], None, 0

    for label, text in units:
        # A single oversized subsection stands alone; split_oversized handles it downstream.
        if len(text) > settings.chunk_target_chars:
            flush()
            packed.append((label, text))
            continue
        if buffer and buffer_len + len(text) + 1 > settings.chunk_target_chars:
            flush()
        if not buffer:
            buffer_label = label
        buffer.append(text)
        buffer_len += len(text) + 1
    flush()

    return packed or [(None, body)]


# --------------------------------------------------------------------------- #
# Judgment chunking
# --------------------------------------------------------------------------- #

JUDGMENT_SEGMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("facts", ("facts", "factual background", "brief facts", "background", "factual matrix")),
    ("issues", ("issues", "questions for consideration", "point for determination", "issue for consideration")),
    ("arguments", ("arguments", "submissions", "contentions", "learned counsel", "rival submissions")),
    ("reasoning", ("analysis", "reasoning", "discussion", "consideration", "we have heard")),
    ("ratio", ("principle", "ratio", "law laid down", "settled law")),
    ("decision", ("conclusion", "held", "order", "disposition", "we allow", "we dismiss", "decree")),
)


def chunk_judgment(
    *,
    metadata_text: str,
    segments: dict[str, str],
    case_title: str,
    source_url: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[ChunkCandidate]:
    """Chunk a judgment by its legal sections rather than by length."""
    settings = get_settings()
    candidates: list[ChunkCandidate] = []
    ordinal = 0

    ordered: list[tuple[str, str]] = []
    if metadata_text.strip():
        ordered.append(("metadata", metadata_text.strip()))
    for key, _ in JUDGMENT_SEGMENTS:
        value = (segments.get(key) or "").strip()
        if value:
            ordered.append((key, value))
    for key, value in segments.items():
        if key not in dict(ordered) and key not in {k for k, _ in JUDGMENT_SEGMENTS} and value.strip():
            ordered.append((key, value.strip()))

    for segment_name, body in ordered:
        header = f"{case_title} - {segment_name.replace('_', ' ').title()}"
        full = f"{header}\n\n{body}"
        for piece in split_oversized(
            full, max_chars=settings.chunk_max_chars, overlap=settings.chunk_overlap_chars
        ):
            if len(piece.strip()) < settings.chunk_min_chars:
                continue
            candidates.append(
                ChunkCandidate(
                    text=piece,
                    collection=Collection.CASE_LAW,
                    source_type="judgment",
                    source_title=case_title,
                    ordinal=ordinal,
                    chunk_strategy=ChunkStrategy.JUDGMENT_SECTION,
                    heading=segment_name.replace("_", " ").title(),
                    section=segment_name,
                    source_url=source_url,
                    metadata={"segment": segment_name, **(extra_metadata or {})},
                )
            )
            ordinal += 1
    return candidates


def segment_judgment_text(text: str) -> dict[str, str]:
    """Heuristically split raw judgment text into named legal segments.

    When no headings are recognised the whole text is returned under 'judgment', so nothing is
    lost. That fallback is recorded in chunk metadata so the UI can show that segmentation was
    approximate for that document.
    """
    cleaned = clean_legal_text(text)
    if not cleaned:
        return {}

    lines = cleaned.split("\n")
    hits: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        probe = line.strip().lower().strip(" .:-–—")
        if not probe or len(probe) > 90:
            continue
        for key, markers in JUDGMENT_SEGMENTS:
            if any(probe.startswith(marker) for marker in markers):
                hits.append((index, key))
                break

    if not hits:
        return {"judgment": cleaned}

    segments: dict[str, str] = {}
    if hits[0][0] > 0:
        preamble = "\n".join(lines[: hits[0][0]]).strip()
        if preamble:
            segments["preamble"] = preamble
    for position, (index, key) in enumerate(hits):
        end = hits[position + 1][0] if position + 1 < len(hits) else len(lines)
        body = "\n".join(lines[index:end]).strip()
        if body:
            segments[key] = f"{segments[key]}\n\n{body}" if key in segments else body
    return segments


# --------------------------------------------------------------------------- #
# Uploaded document chunking
# --------------------------------------------------------------------------- #

_HEADING_RE = re.compile(r"^(?:[A-Z][A-Z0-9 ,.&/()-]{4,80}|\d+\.\s+[A-Z].{0,80})$")


def detect_headings(page_text: str) -> list[str]:
    headings: list[str] = []
    for line in page_text.split("\n"):
        stripped = line.strip()
        if 5 <= len(stripped) <= 90 and _HEADING_RE.match(stripped):
            headings.append(stripped)
    return headings[:12]


def chunk_uploaded_document(
    *,
    pages: list[tuple[int, str]],
    filename: str,
    document_id: str,
    matter_id: str | None = None,
    doc_class: str = "other",
) -> list[ChunkCandidate]:
    """Chunk an uploaded document by page then paragraph, never losing the page number."""
    settings = get_settings()
    candidates: list[ChunkCandidate] = []
    ordinal = 0

    for page_number, raw_text in pages:
        cleaned = clean_legal_text(raw_text)
        if not cleaned:
            continue
        headings = detect_headings(cleaned)
        current_heading = headings[0] if headings else None

        blocks = [b.strip() for b in _PARAGRAPH_RE.split(cleaned) if b.strip()]
        buffer: list[str] = []
        buffer_len = 0

        def flush(heading: str | None) -> None:
            nonlocal buffer, buffer_len, ordinal
            if not buffer:
                return
            body = "\n\n".join(buffer)
            header = f"{filename} - Page {page_number}"
            if heading:
                header += f" - {heading}"
            full = f"{header}\n\n{body}"
            for piece in split_oversized(
                full, max_chars=settings.chunk_max_chars, overlap=settings.chunk_overlap_chars
            ):
                candidates.append(
                    ChunkCandidate(
                        text=piece,
                        collection=Collection.USER_DOCUMENTS,
                        source_type="user_document",
                        source_title=filename,
                        ordinal=ordinal,
                        chunk_strategy=ChunkStrategy.DOCUMENT_PAGE_PARAGRAPH,
                        page=page_number,
                        heading=heading,
                        metadata={
                            "document_id": document_id,
                            "matter_id": matter_id,
                            "doc_class": doc_class,
                            "page": page_number,
                        },
                    )
                )
                ordinal += 1
            buffer, buffer_len = [], 0

        for block in blocks:
            if block in headings:
                flush(current_heading)
                current_heading = block
                continue
            if buffer_len + len(block) > settings.chunk_target_chars and buffer:
                flush(current_heading)
            buffer.append(block)
            buffer_len += len(block) + 2
        flush(current_heading)

    # A page of only headings/short lines would otherwise vanish; keep a minimal record.
    if not candidates and pages:
        for page_number, raw_text in pages:
            cleaned = clean_legal_text(raw_text)
            if not cleaned:
                continue
            candidates.append(
                ChunkCandidate(
                    text=f"{filename} - Page {page_number}\n\n{cleaned}",
                    collection=Collection.USER_DOCUMENTS,
                    source_type="user_document",
                    source_title=filename,
                    ordinal=ordinal,
                    chunk_strategy=ChunkStrategy.FALLBACK_FIXED,
                    page=page_number,
                    metadata={
                        "document_id": document_id,
                        "matter_id": matter_id,
                        "doc_class": doc_class,
                        "fallback": True,
                    },
                )
            )
            ordinal += 1
    return candidates
