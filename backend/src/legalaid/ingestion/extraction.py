"""Document text extraction with page-level provenance.

Provenance is the point: a legal citation that says "page 14" must be verifiable, so every
extracted page keeps its page number and heading candidates. PDF pages that yield no text are
reported as empty rather than being silently skipped, because a scanned page is a real finding
the lawyer needs to see.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from legalaid.core.errors import UploadRejectedError
from legalaid.core.logging import get_logger
from legalaid.domain.enums import DocumentClass
from legalaid.ingestion.chunking import clean_legal_text, detect_headings

logger = get_logger(__name__)


@dataclass(slots=True)
class ExtractedPage:
    page_number: int
    text: str
    headings: list[str] = field(default_factory=list)
    char_count: int = 0
    is_empty: bool = False


@dataclass(slots=True)
class ExtractionResult:
    pages: list[ExtractedPage]
    method: str
    page_count: int
    total_chars: int
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 128), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_document(path: Path, *, content_type: str | None = None) -> ExtractionResult:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix in {".txt", ".md"}:
        return _extract_text(path)
    raise UploadRejectedError(
        f"Unsupported file type '{suffix}'",
        {"supported": [".pdf", ".txt", ".docx", ".md"], "content_type": content_type},
    )


def _extract_pdf(path: Path) -> ExtractionResult:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise UploadRejectedError("PDF support requires PyMuPDF", {"error": str(exc)}) from exc

    pages: list[ExtractedPage] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}

    with pymupdf.open(path) as document:
        metadata = {
            "pdf_title": (document.metadata or {}).get("title"),
            "pdf_author": (document.metadata or {}).get("author"),
            "pdf_page_count": document.page_count,
            "pdf_producer": (document.metadata or {}).get("producer"),
        }
        for index in range(document.page_count):
            page = document.load_page(index)
            raw = page.get_text("text") or ""
            cleaned = clean_legal_text(raw)
            is_empty = not cleaned.strip()
            if is_empty:
                warnings.append(
                    f"Page {index + 1} contains no extractable text "
                    "(likely scanned; OCR would be required)."
                )
            pages.append(
                ExtractedPage(
                    page_number=index + 1,
                    text=cleaned,
                    headings=detect_headings(cleaned),
                    char_count=len(cleaned),
                    is_empty=is_empty,
                )
            )

    total = sum(page.char_count for page in pages)
    if pages and total == 0:
        warnings.append(
            "No text could be extracted from this PDF. It appears to be a scanned document."
        )

    return ExtractionResult(
        pages=pages,
        method="pymupdf",
        page_count=len(pages),
        total_chars=total,
        metadata=metadata,
        warnings=warnings,
    )


def _extract_docx(path: Path) -> ExtractionResult:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise UploadRejectedError("DOCX support requires python-docx", {"error": str(exc)}) from exc

    document = docx.Document(str(path))
    blocks: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            style = (paragraph.style.name or "").lower() if paragraph.style else ""
            blocks.append(text.upper() if "heading" in style and text.islower() else text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))

    # DOCX has no fixed pagination; group into pseudo-pages so page provenance still exists
    # and is labelled as approximate in metadata.
    pages: list[ExtractedPage] = []
    buffer: list[str] = []
    size = 0
    page_number = 1
    for block in blocks:
        buffer.append(block)
        size += len(block)
        if size >= 3000:
            text = clean_legal_text("\n\n".join(buffer))
            pages.append(
                ExtractedPage(page_number, text, detect_headings(text), len(text), not text)
            )
            buffer, size = [], 0
            page_number += 1
    if buffer:
        text = clean_legal_text("\n\n".join(buffer))
        pages.append(ExtractedPage(page_number, text, detect_headings(text), len(text), not text))

    if not pages:
        pages = [ExtractedPage(1, "", [], 0, True)]

    return ExtractionResult(
        pages=pages,
        method="python-docx",
        page_count=len(pages),
        total_chars=sum(p.char_count for p in pages),
        metadata={"pagination": "approximate", "paragraph_count": len(blocks)},
        warnings=["DOCX has no fixed pagination; page numbers are approximate groupings."],
    )


def _extract_text(path: Path) -> ExtractionResult:
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_legal_text(raw)
    pages: list[ExtractedPage] = []
    if len(cleaned) <= 6000:
        pages.append(ExtractedPage(1, cleaned, detect_headings(cleaned), len(cleaned), not cleaned))
    else:
        blocks = cleaned.split("\n\n")
        buffer: list[str] = []
        size = 0
        page_number = 1
        for block in blocks:
            buffer.append(block)
            size += len(block)
            if size >= 4000:
                text = "\n\n".join(buffer)
                pages.append(
                    ExtractedPage(page_number, text, detect_headings(text), len(text), not text)
                )
                buffer, size = [], 0
                page_number += 1
        if buffer:
            text = "\n\n".join(buffer)
            pages.append(ExtractedPage(page_number, text, detect_headings(text), len(text), not text))

    return ExtractionResult(
        pages=pages,
        method="plaintext",
        page_count=len(pages),
        total_chars=sum(p.char_count for p in pages),
        metadata={"pagination": "approximate" if len(pages) > 1 else "single"},
        warnings=[],
    )


def classify_document(filename: str, text: str) -> DocumentClass:
    """Lightweight rule-based classification.

    Deliberately rule-based rather than an LLM call: it is deterministic, instant and its
    reasoning is inspectable. The class only steers chunk metadata and UI grouping, so a
    heuristic is proportionate; the LLM handles genuine document understanding separately.
    """
    haystack = f"{filename}\n{text[:4000]}".lower()

    rules: list[tuple[DocumentClass, tuple[str, ...]]] = [
        (DocumentClass.LEGAL_NOTICE, ("legal notice", "notice under section", "hereby serve", "advocate notice")),
        (DocumentClass.COURT_ORDER, ("in the court of", "order sheet", "it is ordered", "coram", "hon'ble")),
        (DocumentClass.JUDGMENT, ("judgment", "judgement", "appellant", "respondent", "civil appeal")),
        (DocumentClass.AGREEMENT, ("agreement", "this deed", "parties hereto", "witnesseth", "terms and conditions")),
        (DocumentClass.COMPLAINT, ("complaint", "complainant", "fir no", "police station")),
        (DocumentClass.REPLY, ("reply to", "written statement", "counter affidavit", "rejoinder")),
        (DocumentClass.CLIENT_STATEMENT, ("client statement", "statement of facts", "my client states", "instructions from")),
        (DocumentClass.STATUTE, ("act, 1", "act, 2", "be it enacted", "chapter i", "short title and extent")),
    ]

    best: tuple[DocumentClass, int] = (DocumentClass.OTHER, 0)
    for doc_class, markers in rules:
        score = sum(1 for marker in markers if marker in haystack)
        if score > best[1]:
            best = (doc_class, score)
    return best[0]
