"""LLM service ASGI app. The only service that talks to Ollama."""

from __future__ import annotations

from fastapi import FastAPI

from legalaid.services.common import create_service_app, mount_registered_endpoints
from legalaid.services.llm import service as llm_service


def build_app() -> FastAPI:
    app = create_service_app(
        llm_service.SERVICE_NAME,
        title="LegalAId LLM Service",
        description=(
            "Generation through Ollama using the configured Code Llama model. "
            "Records a ModelRun for every call: prompts, parameters, tokens and latency."
        ),
        init_db=True,
    )
    mount_registered_endpoints(app, llm_service.SERVICE_NAME)
    return app


app = build_app()
