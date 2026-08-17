"""Embedding service ASGI app."""

from __future__ import annotations

from fastapi import FastAPI

from legalaid.services.common import create_service_app, mount_registered_endpoints
from legalaid.services.embedding import service as embedding_service


def build_app() -> FastAPI:
    app = create_service_app(
        embedding_service.SERVICE_NAME,
        title="LegalAId Embedding Service",
        description=(
            "Document and query embeddings through the configured Ollama embedding model. "
            "Reports model, dimensions, norms, latency and vector ids for every request."
        ),
        init_db=True,
    )
    mount_registered_endpoints(app, embedding_service.SERVICE_NAME)
    return app


app = build_app()
