"""Structured logging with trace-id correlation.

Every log line carries the trace_id so that logs and the UI trace view can be reconciled.
Secrets are never logged: payloads are truncated and known-sensitive keys are redacted.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

from legalaid.config import get_settings

_trace_id_var: ContextVar[str | None] = ContextVar("legalaid_trace_id", default=None)
_service_var: ContextVar[str | None] = ContextVar("legalaid_service", default=None)

REDACT_KEYS = {
    "password",
    "password_hash",
    "authorization",
    "api_key",
    "apikey",
    "token",
    "secret",
    "cookie",
}

MAX_LOGGED_STRING = 400


def set_trace_id(trace_id: str | None) -> None:
    _trace_id_var.set(trace_id)


def get_trace_id() -> str | None:
    return _trace_id_var.get()


def set_service_name(service: str) -> None:
    _service_var.set(service)


def redact(value: Any, _depth: int = 0) -> Any:
    """Recursively redact sensitive keys and truncate long strings for logging."""
    if _depth > 6:
        return "<truncated:depth>"
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if str(key).lower() in REDACT_KEYS else redact(val, _depth + 1))
            for key, val in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = [redact(item, _depth + 1) for item in value[:20]]
        if len(value) > 20:
            items.append(f"<+{len(value) - 20} more>")
        return items
    if isinstance(value, str) and len(value) > MAX_LOGGED_STRING:
        return value[:MAX_LOGGED_STRING] + f"<+{len(value) - MAX_LOGGED_STRING} chars>"
    return value


def _add_context(
    _logger: Any, _name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    trace_id = _trace_id_var.get()
    if trace_id:
        event_dict.setdefault("trace_id", trace_id)
    service = _service_var.get()
    if service:
        event_dict.setdefault("service", service)
    return event_dict


def _redact_processor(
    _logger: Any, _name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    for key in list(event_dict):
        if str(key).lower() in REDACT_KEYS:
            event_dict[key] = "<redacted>"
        else:
            event_dict[key] = redact(event_dict[key])
    return event_dict


_configured = False


def configure_logging(service: str | None = None) -> None:
    global _configured
    if service:
        set_service_name(service)
    if _configured:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.environment == "docker"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _add_context,
            _redact_processor,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "legalaid") -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]
