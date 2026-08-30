"""Document intake: upload a file or paste text, then extract, chunk, embed and index it.

Page-level provenance is preserved throughout, so a fact extracted from a document can be
traced back to the page it came from. Pasted text is supported as a first-class path because in
practice a lawyer will often paste an FIR or a notice rather than attach a PDF.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from legalaid.config import get_settings
from legalaid.core.logging import get_logger
from legalaid.core.security import Principal, assert_matter_access, enforce_rate_limit
from legalaid.db.base import session_scope
from legalaid.db.models import Chunk, Document, DocumentPage, Matter, Workspace
from legalaid.domain.enums import Collection, DocumentClass, DocumentStatus
from legalaid.ingestion.chunking import chunk_uploaded_document
from legalaid.ingestion.extraction import extract_document
from legalaid.services.knowledge.ingest import _embed_and_index
from legalaid.services.retrieval.graph import reset_graph
from legalaid.services.retrieval.keyword import reset_index

logger = get_logger(__name__)

router = APIRouter()


class PasteRequest(BaseModel):
    text: str = Field(min_length=20)
    filename: str = "pasted-text.txt"
    matter_id: str | None = None
    doc_class: DocumentClass = DocumentClass.OTHER


def _classify(filename: str, text: str) -> DocumentClass:
    """Cheap heuristic document classification from filename and opening text.

    Deliberately keyword-based rather than an LLM call: it runs on every upload, the signals are
    strong and explicit in Indian legal documents, and it costs nothing. The Intake Router
    refines this later with the model when it matters.
    """
    haystack = f"{filename}\n{text[:3000]}".lower()
    rules: list[tuple[DocumentClass, tuple[str, ...]]] = [
        (DocumentClass.LEGAL_NOTICE, ("legal notice", "notice under section", "hereby serve")),
        (DocumentClass.COMPLAINT, ("first information report", "f.i.r", "fir no", "complaint")),
        (DocumentClass.AGREEMENT, ("agreement", "memorandum of understanding", "this deed")),
        (DocumentClass.COURT_ORDER, ("order", "in the court of", "hon'ble", "judgment")),
        (DocumentClass.REPLY, ("reply to", "written statement", "rejoinder")),
        (DocumentClass.CLIENT_STATEMENT, ("statement of", "client statement", "affidavit")),
    ]
    for doc_class, markers in rules:
        if any(marker in haystack for marker in markers):
            return doc_class
    return DocumentClass.OTHER


async def _process(
    *,
    filename: str,
    raw: bytes | None,
    text: str | None,
    matter_id: str | None,
    doc_class: DocumentClass | None,
    is_synthetic: bool = False,
) -> dict[str, Any]:
    """Shared pipeline for uploaded files and pasted text."""
    settings = get_settings()
    settings.ensure_directories()

    with session_scope() as session:
        if matter_id:
            if session.get(Matter, matter_id) is None:
                raise HTTPException(404, f"Matter {matter_id} not found")
            workspace_id = session.get(Matter, matter_id).workspace_id
        else:
            workspace = session.scalar(select(Workspace).order_by(Workspace.created_at))
            workspace_id = workspace.id if workspace else None

    # ---------------------------------------------------------------- extract
    if raw is not None:
        suffix = Path(filename).suffix.lower()
        if suffix not in settings.allowed_upload_extensions:
            raise HTTPException(
                400,
                f"Unsupported file type '{suffix}'. Allowed: "
                f"{', '.join(settings.allowed_upload_extensions)}",
            )
        if len(raw) > settings.max_upload_bytes:
            raise HTTPException(
                400,
                f"File is {len(raw) / 1e6:.1f} MB; the limit is "
                f"{settings.max_upload_bytes / 1e6:.0f} MB",
            )
        digest = hashlib.sha256(raw).hexdigest()
        stored = settings.upload_dir / f"{digest[:16]}{suffix}"
        stored.write_bytes(raw)
        extraction = extract_document(stored)
        pages = extraction.pages
        method = extraction.method
        size_bytes = len(raw)
        stored_path = str(stored)
    else:
        body = (text or "").strip()
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        stored = settings.upload_dir / f"{digest[:16]}.txt"
        stored.write_text(body, encoding="utf-8")
        # Pasted text has no real pages; treat it as a single page so provenance still resolves.
        pages = [(1, body)]
        method = "pasted_text"
        size_bytes = len(body.encode("utf-8"))
        stored_path = str(stored)

    joined = "\n".join(t for _n, t in pages)
    if not joined.strip():
        raise HTTPException(
            422,
            "No text could be extracted. If this is a scanned PDF it needs OCR, which is not "
            "enabled for this upload path.",
        )

    resolved_class = doc_class or _classify(filename, joined)

    # ------------------------------------------------------------------- dedup
    # The digest already decided the stored filename, so re-uploading a file silently
    # overwrote it on disk while still inserting a fresh Document, Pages, Chunks and
    # embeddings. The duplicate chunks then competed with the originals for top-k, quietly
    # degrading retrieval - the worst kind of bug, because nothing appears to fail.
    with session_scope() as session:
        existing = session.scalar(
            select(Document)
            .where(Document.sha256 == digest, Document.matter_id == matter_id)
            .order_by(Document.uploaded_at)
        )
        if existing is not None:
            existing_chunks = (
                session.scalar(
                    select(func.count()).select_from(Chunk).where(Chunk.document_id == existing.id)
                )
                or 0
            )
            logger.info(
                "document_duplicate_ignored",
                document_id=existing.id,
                sha256=digest[:16],
                filename=filename,
            )
            return {
                "document_id": existing.id,
                "filename": existing.filename,
                "doc_class": existing.doc_class,
                "pages": existing.page_count,
                "chunks": existing_chunks,
                "vectors_indexed": 0,
                "extraction_method": existing.extraction_method,
                "chars": len(joined),
                "status": existing.status,
                "sha256": digest,
                "duplicate": True,
                "detail": (
                    "This file is byte-identical to a document already ingested "
                    f"({existing.filename}), so the existing record was reused. Re-indexing it "
                    "would put duplicate chunks into retrieval."
                ),
            }

    # ------------------------------------------------------------------ store
    with session_scope() as session:
        document = Document(
            matter_id=matter_id,
            workspace_id=workspace_id,
            filename=filename,
            stored_path=stored_path,
            content_type="application/pdf" if filename.lower().endswith(".pdf") else "text/plain",
            size_bytes=size_bytes,
            sha256=digest,
            doc_class=resolved_class.value,
            page_count=len(pages),
            status=DocumentStatus.EXTRACTED.value,
            extraction_method=method,
            is_synthetic=is_synthetic,
        )
        session.add(document)
        session.flush()
        document_id = document.id

        for number, page_text in pages:
            session.add(
                DocumentPage(
                    document_id=document_id,
                    page_number=number,
                    text=page_text,
                    char_count=len(page_text),
                )
            )

        candidates = chunk_uploaded_document(
            pages=pages,
            filename=filename,
            document_id=document_id,
            matter_id=matter_id,
            doc_class=resolved_class.value,
        )
        chunk_ids: list[str] = []
        for candidate in candidates:
            chunk = Chunk(
                document_id=document_id,
                matter_id=matter_id,
                collection=Collection.USER_DOCUMENTS.value,
                source_type=candidate.source_type,
                source_title=filename,
                section=candidate.section,
                heading=candidate.heading,
                page=candidate.page,
                text=candidate.text,
                char_count=candidate.char_count,
                token_count=candidate.token_count,
                ordinal=candidate.ordinal,
                chunk_strategy=candidate.chunk_strategy.value,
                chunk_metadata=candidate.metadata,
            )
            session.add(chunk)
            session.flush()
            chunk_ids.append(chunk.id)
        document.status = DocumentStatus.CHUNKED.value

    # ------------------------------------------------------- embed and index
    vectors = 0
    if chunk_ids:
        vectors = await _embed_and_index(chunk_ids, Collection.USER_DOCUMENTS)
        with session_scope() as session:
            doc = session.get(Document, document_id)
            if doc is not None:
                doc.status = DocumentStatus.INDEXED.value

    # New chunks invalidate the lexical index and the graph.
    reset_index()
    reset_graph()

    logger.info(
        "document_ingested",
        document_id=document_id,
        pages=len(pages),
        chunks=len(chunk_ids),
        vectors=vectors,
        doc_class=resolved_class.value,
    )
    return {
        "document_id": document_id,
        "filename": filename,
        "doc_class": resolved_class.value,
        "pages": len(pages),
        "chunks": len(chunk_ids),
        "vectors_indexed": vectors,
        "extraction_method": method,
        "chars": len(joined),
        "status": DocumentStatus.INDEXED.value if vectors else DocumentStatus.CHUNKED.value,
        "sha256": digest,
        "duplicate": False,
    }


@router.post("/api/documents/upload", tags=["documents"])
async def upload_document(
    file: UploadFile = File(...),
    matter_id: str | None = Form(None),
    doc_class: str | None = Form(None),
    principal: Principal = Depends(enforce_rate_limit),
) -> dict[str, Any]:
    """Upload a PDF/DOCX/TXT, extract with page provenance, chunk, embed and index it.

    Rate limited because ingestion embeds every chunk, and authorised against the target
    matter so a caller cannot attach documents to someone else's matter.
    """
    assert_matter_access(matter_id, principal)
    raw = await file.read()
    chosen = None
    if doc_class:
        try:
            chosen = DocumentClass(doc_class)
        except ValueError as exc:
            raise HTTPException(400, f"Unknown doc_class '{doc_class}'") from exc
    return await _process(
        filename=file.filename or "upload.bin",
        raw=raw,
        text=None,
        matter_id=matter_id,
        doc_class=chosen,
    )


@router.post("/api/documents/paste", tags=["documents"])
async def paste_document(
    payload: PasteRequest, principal: Principal = Depends(enforce_rate_limit)
) -> dict[str, Any]:
    """Ingest pasted text (an FIR, a notice) exactly like an uploaded file."""
    assert_matter_access(payload.matter_id, principal)
    return await _process(
        filename=payload.filename,
        raw=None,
        text=payload.text,
        matter_id=payload.matter_id,
        doc_class=payload.doc_class,
    )
