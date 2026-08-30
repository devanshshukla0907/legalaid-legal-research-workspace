"""Shared FastAPI application factory for every LegalAId service.

Each service gets the same middleware stack: trace-id propagation, structured logging,
consistent error handling and OpenAPI documentation. Handlers registered for in-process
dispatch are mounted here too, so one implementation serves both transports.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from legalaid.config import get_settings
from legalaid.core.errors import install_exception_handlers
from legalaid.core.logging import configure_logging, get_logger, set_trace_id
from legalaid.core.security import authenticate_request, is_public_path, valid_keys
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

    @app.middleware("http")
    async def _auth_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Authenticate everything except a small enumerated public allowlist.

        Enforced here rather than per-route so that a route added later is protected by
        default. `/api/traces/{id}` returns the verbatim prompts sent to the model, including
        document excerpts, so leaving a new endpoint open by accident is a real risk.
        """
        # CORS preflight carries no credentials by design.
        if request.method == "OPTIONS" or is_public_path(request.url.path):
            return await call_next(request)
        try:
            request.state.principal = authenticate_request(request)
        except HTTPException as exc:
            return JSONResponse(
                {"error": {"code": exc.status_code, "message": exc.detail}},
                status_code=exc.status_code,
                headers=exc.headers or {},
            )
        return await call_next(request)

    install_exception_handlers(app)

    if init_db:

        @app.on_event("startup")
        async def _startup() -> None:
            create_all()
            if settings.auth_enabled:
                # Touching valid_keys() here makes the generated development key appear in the
                # startup log, rather than on whichever request happens to arrive first.
                keys = valid_keys()
                logger.info(
                    "auth_enabled",
                    configured_keys=len([k for k in settings.api_keys if k.strip()]),
                    using_dev_key=keys[0].startswith("dev-"),
                )
            else:
                logger.warning("auth_disabled", detail="AUTH_ENABLED=false; API is open.")
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
