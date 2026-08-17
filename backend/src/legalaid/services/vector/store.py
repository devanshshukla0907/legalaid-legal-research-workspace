"""FAISS-backed vector store, one index per knowledge collection.

Design decisions and why:

- **Flat inner-product index on L2-normalised vectors.** With normalised vectors, inner product
  IS cosine similarity, and a flat index gives *exact* search. The MVP corpus is under ~2,000
  vectors, so IVF/HNSW would add approximation error and tuning burden for no measurable gain.
- **One index per collection.** The requirement is explicit that retrieval must know which
  source type it is searching. Separate indexes make that structural rather than a filter.
- **IDMap wrapper.** FAISS assigns sequential positions; we need stable chunk ids. IndexIDMap2
  lets us add and remove by our own integer id, which is what makes delete work properly.
- **Metadata sidecar.** Filters (matter_id, act_name, court, ...) are evaluated against a JSON
  sidecar. Filtering happens *before* top-k selection by over-fetching candidates, so a filter
  can never silently shrink the result set below top_k when matches exist.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from legalaid.config import get_settings
from legalaid.core.errors import VectorStoreError
from legalaid.core.logging import get_logger
from legalaid.db.base import utcnow
from legalaid.domain.enums import Collection

logger = get_logger(__name__)


@dataclass(slots=True)
class StoredVector:
    chunk_id: str
    vector_id: str
    faiss_id: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    vector_id: str
    similarity: float
    metadata: dict[str, Any]


def normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise so that inner product equals cosine similarity."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).astype(np.float32)


class CollectionIndex:
    """A single FAISS index plus its metadata sidecar, persisted to disk."""

    def __init__(self, collection: Collection, dimensions: int, index_dir: Path) -> None:
        self.collection = collection
        self.dimensions = dimensions
        self.index_dir = index_dir
        self.index_path = index_dir / f"{collection.value.lower()}.faiss"
        self.meta_path = index_dir / f"{collection.value.lower()}.meta.json"
        self._lock = threading.RLock()
        self._index: Any = None
        self._by_chunk: dict[str, StoredVector] = {}
        self._by_faiss_id: dict[int, str] = {}
        self._next_faiss_id = 1
        self._embedding_model: str | None = None
        self._last_updated: datetime | None = None
        self._load()

    # ------------------------------------------------------------------ faiss
    def _faiss(self) -> Any:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise VectorStoreError(
                "faiss-cpu is required for the vector store", {"error": str(exc)}
            ) from exc
        return faiss

    def _new_index(self) -> Any:
        faiss = self._faiss()
        base = faiss.IndexFlatIP(self.dimensions)
        return faiss.IndexIDMap2(base)

    def _load(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss = self._faiss()
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self._index = faiss.read_index(str(self.index_path))
                payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if payload.get("dimensions") != self.dimensions:
                    logger.warning(
                        "vector_index_dimension_mismatch",
                        collection=self.collection.value,
                        stored=payload.get("dimensions"),
                        configured=self.dimensions,
                    )
                    self._reset()
                    return
                for entry in payload.get("vectors", []):
                    stored = StoredVector(
                        chunk_id=entry["chunk_id"],
                        vector_id=entry["vector_id"],
                        faiss_id=int(entry["faiss_id"]),
                        metadata=entry.get("metadata", {}),
                    )
                    self._by_chunk[stored.chunk_id] = stored
                    self._by_faiss_id[stored.faiss_id] = stored.chunk_id
                self._next_faiss_id = int(payload.get("next_faiss_id", 1))
                self._embedding_model = payload.get("embedding_model")
                updated = payload.get("last_updated")
                self._last_updated = datetime.fromisoformat(updated) if updated else None
                return
            except Exception as exc:  # noqa: BLE001 - corrupt index must not break startup
                logger.warning(
                    "vector_index_load_failed",
                    collection=self.collection.value,
                    error=str(exc),
                )
        self._reset()

    def _reset(self) -> None:
        self._index = self._new_index()
        self._by_chunk.clear()
        self._by_faiss_id.clear()
        self._next_faiss_id = 1
        self._last_updated = None

    def _persist(self) -> None:
        faiss = self._faiss()
        faiss.write_index(self._index, str(self.index_path))
        payload = {
            "collection": self.collection.value,
            "dimensions": self.dimensions,
            "next_faiss_id": self._next_faiss_id,
            "embedding_model": self._embedding_model,
            "last_updated": (self._last_updated or utcnow()).isoformat(),
            "vectors": [
                {
                    "chunk_id": stored.chunk_id,
                    "vector_id": stored.vector_id,
                    "faiss_id": stored.faiss_id,
                    "metadata": stored.metadata,
                }
                for stored in self._by_chunk.values()
            ],
        }
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.meta_path)

    # ----------------------------------------------------------------- upsert
    def upsert(
        self,
        items: list[tuple[str, list[float], dict[str, Any]]],
        *,
        embedding_model: str | None = None,
    ) -> list[str]:
        if not items:
            return []
        with self._lock:
            vectors = np.zeros((len(items), self.dimensions), dtype=np.float32)
            for row, (chunk_id, vector, _meta) in enumerate(items):
                if len(vector) != self.dimensions:
                    raise VectorStoreError(
                        "Embedding dimension mismatch",
                        {
                            "collection": self.collection.value,
                            "chunk_id": chunk_id,
                            "expected": self.dimensions,
                            "received": len(vector),
                        },
                    )
                vectors[row] = np.asarray(vector, dtype=np.float32)
            vectors = normalise(vectors)

            # Replacing an existing chunk means removing the old vector first, otherwise the
            # same chunk would be returned twice by a search.
            replacing = [c for c, _, _ in items if c in self._by_chunk]
            if replacing:
                self._remove_ids([self._by_chunk[c].faiss_id for c in replacing])

            faiss_ids = np.zeros(len(items), dtype=np.int64)
            vector_ids: list[str] = []
            for row, (chunk_id, _vector, metadata) in enumerate(items):
                faiss_id = self._next_faiss_id
                self._next_faiss_id += 1
                faiss_ids[row] = faiss_id
                vector_id = f"{self.collection.value}#{faiss_id}"
                vector_ids.append(vector_id)
                stored = StoredVector(chunk_id, vector_id, faiss_id, metadata)
                self._by_chunk[chunk_id] = stored
                self._by_faiss_id[faiss_id] = chunk_id

            self._index.add_with_ids(vectors, faiss_ids)
            if embedding_model:
                self._embedding_model = embedding_model
            self._last_updated = utcnow()
            self._persist()
            return vector_ids

    def _remove_ids(self, faiss_ids: list[int]) -> int:
        if not faiss_ids:
            return 0
        faiss = self._faiss()
        selector = faiss.IDSelectorBatch(np.asarray(faiss_ids, dtype=np.int64))
        removed = int(self._index.remove_ids(selector))
        for faiss_id in faiss_ids:
            chunk_id = self._by_faiss_id.pop(faiss_id, None)
            if chunk_id is not None:
                self._by_chunk.pop(chunk_id, None)
        return removed

    def delete(self, chunk_ids: list[str]) -> int:
        with self._lock:
            targets = [self._by_chunk[c].faiss_id for c in chunk_ids if c in self._by_chunk]
            removed = self._remove_ids(targets)
            if removed:
                self._last_updated = utcnow()
                self._persist()
            return removed

    # ----------------------------------------------------------------- search
    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        min_similarity: float | None = None,
    ) -> tuple[list[SearchHit], int, int]:
        """Return (hits, candidate_count, filtered_count).

        candidate_count is the total number of vectors in the collection - the honest
        denominator shown in the Retrieval Inspector, not the number FAISS happened to scan.
        """
        with self._lock:
            total = len(self._by_chunk)
            if total == 0:
                return [], 0, 0
            if len(vector) != self.dimensions:
                raise VectorStoreError(
                    "Query embedding dimension mismatch",
                    {
                        "collection": self.collection.value,
                        "expected": self.dimensions,
                        "received": len(vector),
                    },
                )

            filters = filters or {}
            settings = get_settings()
            # Over-fetch so metadata filtering cannot starve the result set.
            fetch = total if filters else min(total, max(top_k, 1) * settings.retrieval_candidate_multiplier)
            fetch = max(1, min(total, fetch))

            query = normalise(np.asarray(vector, dtype=np.float32))
            scores, ids = self._index.search(query, fetch)

            eligible = 0
            hits: list[SearchHit] = []
            for score, faiss_id in zip(scores[0], ids[0], strict=False):
                if faiss_id == -1:
                    continue
                chunk_id = self._by_faiss_id.get(int(faiss_id))
                if chunk_id is None:
                    continue
                stored = self._by_chunk.get(chunk_id)
                if stored is None:
                    continue
                if not _matches(stored.metadata, filters):
                    continue
                eligible += 1
                similarity = float(score)
                if min_similarity is not None and similarity < min_similarity:
                    continue
                hits.append(
                    SearchHit(
                        chunk_id=chunk_id,
                        vector_id=stored.vector_id,
                        similarity=similarity,
                        metadata=stored.metadata,
                    )
                )

            hits.sort(key=lambda hit: hit.similarity, reverse=True)
            return hits[:top_k], total, eligible

    # ------------------------------------------------------------------ stats
    @property
    def count(self) -> int:
        return len(self._by_chunk)

    @property
    def embedding_model(self) -> str | None:
        return self._embedding_model

    @property
    def last_updated(self) -> datetime | None:
        return self._last_updated

    def size_bytes(self) -> int:
        total = 0
        for path in (self.index_path, self.meta_path):
            if path.exists():
                total += path.stat().st_size
        return total

    def clear(self) -> None:
        with self._lock:
            self._reset()
            self._persist()

    def get_vector(self, chunk_id: str) -> list[float] | None:
        with self._lock:
            stored = self._by_chunk.get(chunk_id)
            if stored is None:
                return None
            try:
                reconstructed = self._index.reconstruct(stored.faiss_id)
            except Exception:  # noqa: BLE001 - reconstruct unsupported after some removals
                return None
            return [float(value) for value in reconstructed]


def _matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Metadata filter supporting scalars, list membership and {"in": [...]} clauses."""
    for key, expected in filters.items():
        if expected is None:
            continue
        actual = metadata.get(key)
        if isinstance(expected, dict) and "in" in expected:
            allowed = expected["in"]
            if isinstance(actual, list):
                if not set(map(str, actual)) & set(map(str, allowed)):
                    return False
            elif str(actual) not in {str(v) for v in allowed}:
                return False
            continue
        if isinstance(expected, (list, tuple, set)):
            allowed = {str(v) for v in expected}
            if isinstance(actual, list):
                if not {str(v) for v in actual} & allowed:
                    return False
            elif str(actual) not in allowed:
                return False
            continue
        if isinstance(actual, list):
            if str(expected) not in {str(v) for v in actual}:
                return False
            continue
        if str(actual) != str(expected):
            return False
    return True


class VectorStore:
    """Holds one CollectionIndex per collection."""

    def __init__(self, index_dir: Path | None = None, dimensions: int | None = None) -> None:
        settings = get_settings()
        self.index_dir = index_dir or settings.index_dir
        self.dimensions = dimensions or settings.embedding_dimensions
        self._indexes: dict[Collection, CollectionIndex] = {}
        self._lock = threading.RLock()

    def index(self, collection: Collection) -> CollectionIndex:
        with self._lock:
            if collection not in self._indexes:
                self._indexes[collection] = CollectionIndex(
                    collection, self.dimensions, self.index_dir
                )
            return self._indexes[collection]

    def all_indexes(self) -> dict[Collection, CollectionIndex]:
        for collection in Collection:
            self.index(collection)
        return dict(self._indexes)

    def total_vectors(self) -> int:
        return sum(index.count for index in self.all_indexes().values())

    def reset(self) -> None:
        for index in self.all_indexes().values():
            index.clear()


_store: VectorStore | None = None
_store_lock = threading.Lock()


def get_store() -> VectorStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = VectorStore()
        return _store


def reset_store() -> None:
    global _store
    with _store_lock:
        _store = None
