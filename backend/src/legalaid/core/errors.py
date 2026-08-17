"""Error model shared by every service.

Errors are surfaced to the UI as real, typed failures with the trace id attached, rather
than being swallowed. A failed stage stays visible in the trace.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from legalaid.core.logging import get_logger, get_trace_id

logger = get_logger(__name__)


class LegalAIdError(Exception):
    """Base application error."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "internal_error"

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "detail": self.detail,
                "trace_id": get_trace_id(),
            }
        }


class NotFoundError(LegalAIdError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ValidationFailedError(LegalAIdError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_failed"


class UploadRejectedError(LegalAIdError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "upload_rejected"


class CapabilityUnavailableError(LegalAIdError):
    status_code = status.HTTP_409_CONFLICT
    code = "capability_unavailable"


class UpstreamServiceError(LegalAIdError):
    """A dependent service failed. Includes which service, so the System page can show it."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "upstream_service_error"

    def __init__(
        self, service: str, message: str, detail: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, {**(detail or {}), "service": service})
        self.service = service


class LLMUnavailableError(LegalAIdError):
    """Ollama or the configured model is not reachable.

    This is reported honestly rather than falling back to a different provider, because the
    application is required to demonstrate Ollama + Code Llama specifically.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "llm_unavailable"


class VectorStoreError(LegalAIdError):
    code = "vector_store_error"


class PromptNotFoundError(LegalAIdError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "prompt_not_found"


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LegalAIdError)
    async def _handle_app_error(_request: Request, exc: LegalAIdError) -> JSONResponse:
        logger.warning(
            "application_error", code=exc.code, message=exc.message, detail=exc.detail
        )
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(Exception)
    async def _handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_error", error=str(exc), exc_info=True)
        wrapped = LegalAIdError(f"{type(exc).__name__}: {exc}")
        return JSONResponse(status_code=wrapped.status_code, content=wrapped.to_payload())
