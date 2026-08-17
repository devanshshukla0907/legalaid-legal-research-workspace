"""Retrieval service ASGI app."""

from __future__ import annotations

from fastapi import FastAPI

from legalaid.services.common import create_service_app, mount_registered_endpoints
from legalaid.services.retrieval import service as retrieval_service


def build_app() -> FastAPI:
    app = create_service_app(
        retrieval_service.SERVICE_NAME,
        title="LegalAId Retrieval Service",
        description=(
            "Query embedding, vector similarity search, metadata filtering, ranking and "
            "context construction, with a recorded explanation for every retrieved chunk."
        ),
        init_db=True,
    )
    mount_registered_endpoints(app, retrieval_service.SERVICE_NAME)
    return app


app = build_app()
