"""Codebase ingestion: repository source files -> Chunk -> embedding -> FAISS.

Exercise 6 measured 0/61 relevant files retrieved when the legal RAG system was asked questions
about its own source code, because the corpus held only law and cases. The corpus-distance
guardrail then made that failure safe (refuse rather than answer confidently with statutory
text), but it did not close the retrieval gap itself. This module closes it: it walks the
repository's own source trees, chunks each file by function/class boundary
(`ingestion.chunking.chunk_source_code`), embeds the chunks with the same embedding model used
for everything else, and indexes them into the dedicated `Collection.CODE` FAISS index.

Deliberately excludes generated/vendored/test-fixture directories (node_modules, __pycache__,
.venv, dist, build) and caps total files, so a full ingest run stays a few minutes rather than
indexing the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import delete, select

import legalaid
from legalaid.capabilities import require
from legalaid.config import RepoRoot, get_settings
from legalaid.core.logging import get_logger
from legalaid.db.base import session_scope
from legalaid.db.models import Chunk, EmbeddingMetadata
from legalaid.domain.enums import Collection
from legalaid.ingestion.chunking import chunk_source_code
from legalaid.services.knowledge.ingest import _embed_and_index
from legalaid.services.vector.store import get_store

logger = get_logger(__name__)

# Directories that hold nothing worth indexing: dependencies, build output, caches, generated
# artefacts. Walking these would mean indexing other projects' code, not this repository's own.
_EXCLUDED_DIRS = {
    "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".git", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "coverage", ".next", "site-packages", "eval",
}

_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
}

# Located from the imported `legalaid` package itself rather than RepoRoot, deliberately.
# RepoRoot (config.py) is `Path(__file__).resolve().parents[3]`, a fixed index that assumes one
# specific directory depth. That assumption holds for local dev (.../backend/src/legalaid/
# config.py, four levels down to the git root) but not inside the Docker image, where
# `backend/Dockerfile` does `COPY src /app/src` with `WORKDIR /app` - one level shallower, so
# the same fixed index lands on `/` instead of `/app`. Every other RepoRoot-derived path in this
# app (data_dir, index_dir, ...) is masked from that bug because docker-compose.yml sets them
# via explicit env vars (DATA_DIR=/data etc); this module is the first to need the source tree
# itself, which has no such override, so it locates itself from `legalaid.__file__` instead -
# correct in both layouts by construction, not by coincidence of which directory depth
# `parents[3]` happens to land on.
_PACKAGE_DIR = Path(legalaid.__file__).resolve().parent  # .../legalaid, in either layout

# The parts of the repository that are actually source code, as opposed to data, docs or
# generated files. Kept explicit rather than "everything" so a bad run cannot accidentally
# ingest the SQLite database or a node_modules tree that slipped past the exclusion set.
#
# Resolved relative to the backend package itself (see _PACKAGE_DIR above), plus a
# RepoRoot-relative frontend/src root that only exists in local dev - the Docker image never
# copies the frontend into this container at all, so that root is silently absent there and the
# codebase collection is backend-only in a Docker deployment. That is documented, not hidden.
BACKEND_SOURCE_ROOT: Path = _PACKAGE_DIR
FRONTEND_SOURCE_ROOT: Path = RepoRoot / "frontend" / "src"

MAX_FILE_BYTES = 200_000  # skip anything absurdly large; real source files are well under this


@dataclass(slots=True)
class CodebaseIngestReport:
    files_scanned: int = 0
    files_skipped: int = 0
    files_indexed: int = 0
    chunks_created: int = 0
    vectors_indexed: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Codebase: {self.files_indexed} files, {self.chunks_created} chunks, "
            f"{self.vectors_indexed} vectors ({self.files_skipped} skipped, "
            f"{len(self.failures)} failures)"
        )


def _relative_label(path: Path) -> str:
    """A stable, human-readable path for provenance, independent of which root matched.

    Preferred relative-to RepoRoot when the file happens to be under it (always true for
    frontend/src; true for backend source only in local dev). Falls back to a package-relative
    label (`legalaid/...`) inside Docker, where the backend source is not a RepoRoot subtree at
    all - still stable and still traceable to an exact file, which is what the citation
    provenance needs, even though it is not the git-root-relative path used in local dev.
    """
    try:
        return path.relative_to(RepoRoot).as_posix()
    except ValueError:
        return f"legalaid/{path.relative_to(_PACKAGE_DIR).as_posix()}"


def _iter_source_files(roots: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in _LANGUAGE_BY_SUFFIX:
                continue
            if any(part in _EXCLUDED_DIRS for part in path.parts):
                continue
            files.append(path)
    return sorted(files)


def reset_code() -> int:
    """Remove all indexed source code, leaves-inward so foreign keys hold."""
    with session_scope() as session:
        chunk_ids = [
            row.id
            for row in session.scalars(
                select(Chunk).where(Chunk.collection == Collection.CODE.value)
            ).all()
        ]
        if chunk_ids:
            get_store().index(Collection.CODE).delete(chunk_ids)
            session.execute(delete(EmbeddingMetadata).where(EmbeddingMetadata.chunk_id.in_(chunk_ids)))
            session.execute(delete(Chunk).where(Chunk.id.in_(chunk_ids)))
        removed = len(chunk_ids)
    get_store().index(Collection.CODE).clear()
    return removed


async def ingest_codebase(
    *,
    roots: tuple[Path, ...] = (BACKEND_SOURCE_ROOT, FRONTEND_SOURCE_ROOT),
    max_files: int | None = None,
) -> CodebaseIngestReport:
    """Walk the repository, chunk by function/class, embed, and index into Collection.CODE."""
    require("knowledge_ingestion")
    report = CodebaseIngestReport()

    files = _iter_source_files(roots)
    if max_files:
        files = files[:max_files]
    report.files_scanned = len(files)

    new_chunk_ids: list[str] = []
    with session_scope() as session:
        # Idempotent by relative path: re-running ingestion replaces a file's chunks rather
        # than duplicating them, the same resumability contract the statute ingester gives.
        existing_by_path: dict[str, list[str]] = {}
        for row in session.scalars(
            select(Chunk).where(Chunk.collection == Collection.CODE.value)
        ).all():
            existing_by_path.setdefault((row.chunk_metadata or {}).get("file_path", ""), []).append(row.id)

        for path in files:
            try:
                raw_bytes = path.stat().st_size
                if raw_bytes > MAX_FILE_BYTES:
                    report.files_skipped += 1
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                report.failures.append({"file": str(path), "error": str(exc)})
                continue

            if not text.strip():
                report.files_skipped += 1
                continue

            rel_path = _relative_label(path)
            language = _LANGUAGE_BY_SUFFIX[path.suffix]

            stale_ids = existing_by_path.get(rel_path, [])
            if stale_ids:
                get_store().index(Collection.CODE).delete(stale_ids)
                session.execute(delete(EmbeddingMetadata).where(EmbeddingMetadata.chunk_id.in_(stale_ids)))
                session.execute(delete(Chunk).where(Chunk.id.in_(stale_ids)))

            candidates = chunk_source_code(
                text=text,
                file_path=rel_path,
                language=language,
                extra_metadata={"repo_relative_path": rel_path},
            )
            if not candidates:
                report.files_skipped += 1
                continue

            for candidate in candidates:
                chunk = Chunk(
                    collection=candidate.collection.value,
                    source_type=candidate.source_type,
                    source_title=candidate.source_title,
                    heading=candidate.heading,
                    text=candidate.text,
                    char_count=candidate.char_count,
                    token_count=candidate.token_count,
                    ordinal=candidate.ordinal,
                    chunk_strategy=candidate.chunk_strategy.value,
                    chunk_metadata=candidate.metadata,
                )
                session.add(chunk)
                session.flush()
                new_chunk_ids.append(chunk.id)
                report.chunks_created += 1

            report.files_indexed += 1

    if new_chunk_ids:
        report.vectors_indexed = await _embed_and_index(new_chunk_ids, Collection.CODE)

    logger.info(
        "codebase_ingested",
        files=report.files_indexed,
        chunks=report.chunks_created,
        vectors=report.vectors_indexed,
    )
    return report


def codebase_stats() -> dict[str, object]:
    with session_scope() as session:
        chunks = session.scalars(
            select(Chunk).where(Chunk.collection == Collection.CODE.value)
        ).all()
        by_file: dict[str, int] = {}
        for chunk in chunks:
            path = (chunk.chunk_metadata or {}).get("file_path", "unknown")
            by_file[path] = by_file.get(path, 0) + 1
    store = get_store()
    return {
        "chunks": len(chunks),
        "files": len(by_file),
        "vectors": store.index(Collection.CODE).count,
    }
