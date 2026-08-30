"""Authentication, authorisation and rate limiting.

The important assertion here is the one about `/api/traces/{id}`: it returns the verbatim
prompts sent to the model, document excerpts included, and it used to be completely open.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from legalaid.config import get_settings
from legalaid.core.security import TokenBucket, valid_keys


@pytest.fixture
def client(db: None) -> TestClient:
    """Depends on `db` so the schema exists: several of these routes read tables."""
    from legalaid.services.gateway.app import app

    return TestClient(app)


@pytest.fixture
def api_key() -> str:
    return valid_keys()[0]


# --------------------------------------------------------------- authentication


def test_health_is_public(client: TestClient) -> None:
    """A container orchestrator has no credentials, so liveness must not need one."""
    assert client.get("/api/system/health").status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/api/traces",
        "/api/chunks",
        "/api/knowledge",
        "/api/matters",
        "/api/documents",
        "/api/metrics",
        "/api/agents",
        "/api/prompts",
        "/api/models",
    ],
)
def test_endpoints_require_a_key(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 401


def test_trace_detail_requires_a_key(client: TestClient) -> None:
    """This endpoint returns verbatim prompts including document text."""
    assert client.get("/api/traces/trc_anything").status_code == 401


def test_research_requires_a_key(client: TestClient) -> None:
    response = client.post("/api/research/query", json={"question": "Section 73?"})
    assert response.status_code == 401


def test_valid_key_is_accepted(client: TestClient, api_key: str) -> None:
    response = client.get("/api/knowledge", headers={"X-API-Key": api_key})
    assert response.status_code == 200


def test_bearer_scheme_is_accepted(client: TestClient, api_key: str) -> None:
    response = client.get("/api/knowledge", headers={"Authorization": f"Bearer {api_key}"})
    assert response.status_code == 200


def test_wrong_key_is_rejected(client: TestClient) -> None:
    response = client.get("/api/knowledge", headers={"X-API-Key": "not-the-key"})
    assert response.status_code == 401


def test_unauthenticated_response_does_not_leak_data(client: TestClient) -> None:
    body = client.get("/api/traces/trc_x").text.lower()
    for leaked in ("system_prompt", "user_prompt", "raw_response"):
        assert leaked not in body


# ---------------------------------------------------------------- authorisation


def test_unknown_matter_is_not_found(client: TestClient, api_key: str) -> None:
    """An unvalidated matter_id previously reached retrieval as a bare filter."""
    response = client.get(
        "/api/documents", params={"matter_id": "mat_does_not_exist"},
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 404


def test_unknown_document_is_not_found(client: TestClient, api_key: str) -> None:
    response = client.get("/api/documents/doc_nope", headers={"X-API-Key": api_key})
    assert response.status_code == 404


# ----------------------------------------------------------------- rate limiting


def test_rate_limit_dependency_returns_429_over_http() -> None:
    """The dependency must actually fire through the ASGI stack, not just in isolation.

    A mounted probe route is used rather than a real endpoint so this asserts the limiter
    itself without also paying for embedding on every call.
    """
    from fastapi import Depends, FastAPI

    from legalaid.core.security import Principal, enforce_rate_limit

    probe = FastAPI()

    @probe.get("/probe")
    async def _probe(  # pragma: no cover - probe route
        principal: Principal = Depends(enforce_rate_limit),
    ) -> dict[str, str]:
        return {"ok": principal.label}

    settings = get_settings()
    limit = settings.rate_limit_requests
    client = TestClient(probe)
    key = valid_keys()[0]
    first = client.get("/probe", headers={"X-API-Key": key})
    assert first.status_code == 200, (
        f"dependency did not resolve: {first.status_code} {first.text[:200]}"
    )
    codes = [200] + [
        client.get("/probe", headers={"X-API-Key": key}).status_code
        for _ in range(limit + 2)
    ]
    assert codes[:limit] == [200] * limit, f"expected the first {limit} to pass, got {codes}"
    assert codes[limit:] == [429] * 3, f"expected the rest to be limited, got {codes}"
    assert "Retry-After" in client.get("/probe", headers={"X-API-Key": key}).headers


@pytest.mark.anyio
async def test_token_bucket_allows_up_to_the_limit() -> None:
    bucket = TokenBucket()
    for _ in range(3):
        allowed, _retry = await bucket.check("k", limit=3, window=60)
        assert allowed is True
    allowed, retry_after = await bucket.check("k", limit=3, window=60)
    assert allowed is False
    assert retry_after > 0


@pytest.mark.anyio
async def test_token_bucket_is_per_key() -> None:
    bucket = TokenBucket()
    allowed, _ = await bucket.check("a", limit=1, window=60)
    assert allowed is True
    allowed, _ = await bucket.check("a", limit=1, window=60)
    assert allowed is False
    # A different caller must not inherit the first one's exhausted budget.
    allowed, _ = await bucket.check("b", limit=1, window=60)
    assert allowed is True


@pytest.mark.anyio
async def test_token_bucket_window_expires() -> None:
    bucket = TokenBucket()
    allowed, _ = await bucket.check("k", limit=1, window=0.01)
    assert allowed is True
    import asyncio

    await asyncio.sleep(0.05)
    allowed, _ = await bucket.check("k", limit=1, window=0.01)
    assert allowed is True


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
