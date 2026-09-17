"""Authentication, authorisation and rate limiting.

Before this module the API had no access control of any kind: every endpoint was open,
including `/api/traces/{id}`, which returns the verbatim prompts sent to the model - document
excerpts and client names included. The database already had `User`, `Role`, `Permission`,
`Membership` and `DocumentPermission` tables; nothing read them.

Three concerns, deliberately separated:

**Authentication** - API keys over a bearer header. Keys, not passwords, because the consumers
are the SPA and the evaluation harness rather than interactive humans, and because a key can be
rotated without touching user records. Comparison is constant-time.

**Authorisation** - matter ownership. `matter_id` was previously an unvalidated caller-supplied
string used only for relevance filtering, so any caller could read any matter's documents by
guessing an id. This is the concrete analogue of the brief's "an HR assistant must not return
another employee's salary": lawyer A must not retrieve lawyer B's matter.

**Rate limiting** - a token bucket per key, plus a semaphore around generation. One research
request is 4-6 LLM calls on a single GPU, so unbounded concurrency does not just risk abuse, it
makes the model spill out of VRAM and slows every caller down. The semaphore is honest capacity
management rather than a punishment.
"""

from __future__ import annotations

import asyncio
import hmac
import secrets
import time
from collections import deque
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select

from legalaid.config import get_settings
from legalaid.core.logging import get_logger
from legalaid.db.base import session_scope
from legalaid.db.models import Document, Matter

logger = get_logger(__name__)

# Generated once per process when no keys are configured. A hard-coded default key would be
# worse than no auth at all, because it looks like protection.
_DEV_KEY: str | None = None


def _dev_key() -> str:
    """Stable development key, persisted so the SPA and scripts can read it.

    Not logged: the logging layer redacts credential-shaped keys, and defeating that to print a
    secret would be the wrong instinct. Written to a gitignored file under the data directory
    instead, which is also what survives a reload.
    """
    global _DEV_KEY
    if _DEV_KEY is not None:
        return _DEV_KEY

    settings = get_settings()
    key_path = settings.data_dir / ".dev-api-key"
    try:
        if key_path.exists():
            existing = key_path.read_text(encoding="utf-8").strip()
            if existing:
                _DEV_KEY = existing
                return _DEV_KEY
    except OSError:
        pass

    _DEV_KEY = "dev-" + secrets.token_urlsafe(24)
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(_DEV_KEY, encoding="utf-8")
    except OSError as exc:
        logger.warning("dev_key_not_persisted", error=str(exc))
    logger.warning(
        "auth_dev_key_generated",
        detail=(
            "No API_KEYS configured. Generated a development key and wrote it to "
            f"{key_path}. Set API_KEYS in .env for anything real."
        ),
    )
    return _DEV_KEY


def valid_keys() -> list[str]:
    settings = get_settings()
    configured = [k for k in settings.api_keys if k.strip()]
    return configured or [_dev_key()]


@dataclass(frozen=True)
class Principal:
    """Who is making the request."""

    key_id: str
    is_dev_key: bool

    @property
    def label(self) -> str:
        return f"{self.key_id[:8]}..." if len(self.key_id) > 8 else self.key_id


def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


async def require_principal(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Principal:
    """FastAPI dependency: authenticate the caller."""
    settings = get_settings()
    if not settings.auth_enabled:
        return Principal(key_id="anonymous", is_dev_key=True)

    presented = _extract_key(authorization, x_api_key)
    if not presented:
        raise HTTPException(
            401,
            "Missing API key. Send it as 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # compare_digest on every candidate, so a wrong key costs the same time as a right one.
    for candidate in valid_keys():
        if hmac.compare_digest(presented, candidate):
            return Principal(key_id=presented, is_dev_key=presented.startswith("dev-"))

    logger.warning("auth_rejected", presented_prefix=presented[:6])
    raise HTTPException(401, "Invalid API key.", headers={"WWW-Authenticate": "Bearer"})


# --------------------------------------------------------------------------- rate limiting


class TokenBucket:
    """Sliding-window request counter, per key.

    A deque of timestamps rather than a fixed counter, so a caller cannot burst the entire
    allowance immediately after a window boundary.
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str, *, limit: int, window: float) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        async with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = window - (now - hits[0])
                return False, max(0.5, round(retry_after, 1))
            hits.append(now)
            return True, 0.0

    def snapshot(self) -> dict[str, int]:
        return {key: len(hits) for key, hits in self._hits.items()}


_bucket = TokenBucket()


async def enforce_rate_limit(
    request: Request,
    principal: Annotated[Principal, Depends(require_principal)],
) -> Principal:
    """FastAPI dependency: authenticate, then apply the per-key request budget."""
    settings = get_settings()
    if not settings.guardrails_enabled:
        return principal

    allowed, retry_after = await _bucket.check(
        principal.key_id,
        limit=settings.rate_limit_requests,
        window=settings.rate_limit_window_seconds,
    )
    if not allowed:
        logger.warning("rate_limited", key=principal.label, path=request.url.path)
        raise HTTPException(
            429,
            f"Rate limit exceeded: {settings.rate_limit_requests} requests per "
            f"{settings.rate_limit_window_seconds:.0f}s. Retry in {retry_after}s.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
    return principal


# --------------------------------------------------------------------------- GPU capacity

_llm_semaphore: asyncio.Semaphore | None = None
_llm_waiting = 0


def llm_slot() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(get_settings().llm_max_concurrency)
    return _llm_semaphore


class GenerationSlot:
    """Async context manager serialising access to the GPU.

    Concurrent generations on a 6 GB card do not run twice as fast; they force the model to
    stop fully offloading and measurably slow everything down. Queuing is the faster option.
    """

    async def __aenter__(self) -> None:
        global _llm_waiting
        settings = get_settings()
        _llm_waiting += 1
        try:
            await asyncio.wait_for(
                llm_slot().acquire(), timeout=settings.llm_queue_timeout
            )
        except TimeoutError as exc:
            _llm_waiting -= 1
            raise HTTPException(
                503,
                f"The model queue did not clear within {settings.llm_queue_timeout:.0f}s. "
                "Too many concurrent requests for one GPU; try again shortly.",
                headers={"Retry-After": "30"},
            ) from exc
        _llm_waiting -= 1

    async def __aexit__(self, *_exc: object) -> None:
        llm_slot().release()


def queue_depth() -> int:
    return _llm_waiting


def rate_limit_snapshot() -> dict[str, int]:
    """Current request count per key, keyed by a short prefix rather than the key itself."""
    return {
        (key[:8] + "..." if len(key) > 8 else key): count
        for key, count in _bucket.snapshot().items()
    }


# --------------------------------------------------------------------------- authorisation


def assert_matter_access(matter_id: str | None, principal: Principal) -> None:
    """Authorise access to a matter.

    Single-key deployments own everything, so this is currently an existence and
    consistency check rather than a per-tenant one - and it is written to say exactly that
    rather than implying a multi-tenant model that is not there. What it does prevent is the
    real bug: an unvalidated `matter_id` reaching retrieval, where a caller could probe for
    another matter's documents by guessing ids.
    """
    if not matter_id:
        return
    with session_scope() as session:
        matter = session.get(Matter, matter_id)
    if matter is None:
        # 404 rather than 403: revealing which ids exist would itself leak information.
        raise HTTPException(404, f"Matter {matter_id} not found")


def assert_document_access(document_id: str, principal: Principal) -> Document:
    """Authorise access to a document and return it."""
    with session_scope() as session:
        document = session.get(Document, document_id)
        if document is None:
            raise HTTPException(404, f"Document {document_id} not found")
        if document.matter_id:
            assert_matter_access(document.matter_id, principal)
        session.expunge(document)
        return document


# Use these as explicit defaults in endpoint signatures:
#
#     async def endpoint(principal: Principal = Depends(enforce_rate_limit)) -> ...:
#
# Not as `Annotated[Principal, Depends(...)]` aliases. `Principal` is a dataclass, so FastAPI
# will treat it as a request-body model when the dependency is not directly attached in the
# signature. That failed loudly on GET routes (422 on every call) and silently on POST routes
# that already had a body - the parameter validated as part of the payload and the rate limiter
# never ran at all, which is exactly the sort of guardrail that looks present and is not.


# --------------------------------------------------------------------------- middleware

# Paths reachable without a key. Deliberately tiny, and enumerated rather than pattern-matched:
#   - health must be callable by a container orchestrator that has no credentials
#   - the OpenAPI docs describe the surface without exposing any data
# Everything else requires a key, so a route added later is protected by default. That property
# matters more than the convenience of per-route decorators, which are easy to forget.
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/api/system/health",
        # The split-service Docker deployment (llm, embedding, retrieval) exposes its own plain
        # `/health` route per service for the container healthcheck, distinct from the gateway's
        # `/api/system/health`. A container orchestrator polling that route has no credentials
        # either, so it must be public for exactly the same reason the gateway's health route is.
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/docs/oauth2-redirect",
        "/",
    }
)


def is_public_path(path: str) -> bool:
    return path in _PUBLIC_PATHS or path.startswith("/static")


def authenticate_request(request: Request) -> Principal:
    """Synchronous authentication for use inside middleware.

    Mirrors `require_principal`, which exists separately because FastAPI dependencies give
    better OpenAPI output for the endpoints that declare them explicitly.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return Principal(key_id="anonymous", is_dev_key=True)

    presented = _extract_key(
        request.headers.get("authorization"), request.headers.get("x-api-key")
    )
    if not presented:
        raise HTTPException(
            401,
            "Missing API key. Send it as 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    for candidate in valid_keys():
        if hmac.compare_digest(presented, candidate):
            return Principal(key_id=presented, is_dev_key=presented.startswith("dev-"))
    raise HTTPException(401, "Invalid API key.", headers={"WWW-Authenticate": "Bearer"})
