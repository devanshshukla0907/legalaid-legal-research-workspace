"""Shared FastAPI application factory for every LegalAId service.

Each service gets the same middleware stack: trace-id propagation, structured logging,
consistent error handling and OpenAPI documentation. Handlers registered for in-process
dispatch are mounted here too, so one implementation serves both transports.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from legalaid.config import get_settings
from legalaid.core.errors import install_exception_handlers
from legalaid.core.logging import configure_logging, get_logger, set_trace_id
from legalaid.core.tracing import new_trace_id
from legalaid.core.transport import registry
from legalaid.db.base import create_all

TRACE_HEADER = "X-Trace-Id"


def create_service_app(
    service_name: str,
    *,
    title: str,
    description: str,
    init_db: bool = False,
) -> FastAPI:
    configure_logging(service_name)
    logger = get_logger(service_name)
    settings = get_settings()

    app = FastAPI(
        title=title,
        description=description,
        version="0.1.0",
        openapi_tags=[{"name": service_name, "description": description}],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[TRACE_HEADER],
    )

    @app.middleware("http")
    async def _trace_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(TRACE_HEADER)
        trace_id = incoming or new_trace_id()
        set_trace_id(trace_id)
        request.state.trace_id = trace_id
        request.state.trace_id_was_provided = bool(incoming)
        response = await call_next(request)
        response.headers[TRACE_HEADER] = trace_id
        return response

    install_exception_handlers(app)

    if init_db:

        @app.on_event("startup")
        async def _startup() -> None:
            create_all()
            logger.info("service_started", service=service_name, transport=settings.service_transport)

    app.state.service_name = service_name
    return app


def mount_registered_endpoints(app: FastAPI, service_name: str) -> None:
    """Mount every in-process-registered handler of this service as a real HTTP route.

    This guarantees the HTTP and in-process transports expose an identical surface.
    """
    for endpoint in registry.all():
        if endpoint.service != service_name:
            continue
        _mount(app, endpoint.method, endpoint.path, endpoint.handler, endpoint.response_model,
               endpoint.request_model, service_name)


def _mount(
    app: FastAPI,
    method: str,
    path: str,
    handler: Callable[..., Awaitable[Any]],
    response_model: type,
    request_model: type | None,
    service_name: str,
) -> None:
    if request_model is not None:
        async def route(payload: request_model) -> Any:  # type: ignore[valid-type]
            return await handler(payload)
    else:
        async def route() -> Any:  # type: ignore[misc]
            return await handler()

    route.__name__ = f"{method.lower()}_{path.strip('/').replace('/', '_') or 'root'}"
    app.add_api_route(
        path,
        route,
        methods=[method.upper()],
        response_model=response_model,
        tags=[service_name],
        summary=handler.__doc__.strip().splitlines()[0] if handler.__doc__ else None,
    )
