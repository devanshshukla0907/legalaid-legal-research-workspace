"""Pipeline trace recorder.

This module is what makes the UI observability requirement achievable. Every request opens a
trace; every internal step is recorded as an ordered, timed stage with its real input and
output payloads; every retrieval, agent run and model call references the trace id.

Design notes:
- Stages are persisted immediately on start (status=running) and updated on completion, so an
  in-flight request is observable live rather than only after it finishes.
- Payloads are size-capped so that a huge prompt cannot bloat the database, but the cap is
  generous enough to keep prompts and retrieved context fully inspectable.
- A live event bus lets the frontend stream stage transitions over SSE.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from legalaid.capabilities import capability_matrix
from legalaid.config import get_settings
from legalaid.core.logging import get_logger, set_trace_id
from legalaid.db.base import new_id, session_scope, utcnow
from legalaid.db.models import PipelineTrace, TraceStage
from legalaid.domain.enums import STAGE_LABELS, StageKey, StageStatus

logger = get_logger(__name__)

MAX_PAYLOAD_CHARS = 24_000


def _json_safe(value: Any, _depth: int = 0) -> Any:
    """Convert arbitrary values into JSON-serialisable form with size caps."""
    if _depth > 8:
        return "<truncated:depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > MAX_PAYLOAD_CHARS:
            return value[:MAX_PAYLOAD_CHARS] + f"\n<+{len(value) - MAX_PAYLOAD_CHARS} chars>"
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        capped = [_json_safe(item, _depth + 1) for item in seq[:200]]
        if len(seq) > 200:
            capped.append(f"<+{len(seq) - 200} more items>")
        return capped
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"), _depth + 1)
    return str(value)


def new_trace_id() -> str:
    """Human-readable trace ids, e.g. LA-3f9c21ab. Shown directly in the UI."""
    return f"LA-{new_id()[:8]}"


class TraceEventBus:
    """In-memory pub/sub so the UI can stream live pipeline progress via SSE."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def subscribe(self, trace_id: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers[trace_id].append(queue)
        return queue

    async def unsubscribe(self, trace_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            if queue in self._subscribers.get(trace_id, []):
                self._subscribers[trace_id].remove(queue)
            if not self._subscribers.get(trace_id):
                self._subscribers.pop(trace_id, None)

    def publish(self, trace_id: str, event: dict[str, Any]) -> None:
        """Non-blocking publish; safe to call from sync code with no running loop."""
        queues = list(self._subscribers.get(trace_id, []))
        if not queues:
            return
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - slow consumer
                logger.debug("trace_event_dropped", trace_id=trace_id)


event_bus = TraceEventBus()


class StageHandle:
    """Mutable handle for a running stage."""

    def __init__(self, recorder: TraceRecorder, stage_id: str, stage_key: str, ordinal: int):
        self._recorder = recorder
        self.stage_id = stage_id
        self.stage_key = stage_key
        self.ordinal = ordinal
        self._output: dict[str, Any] = {}

    def set_output(self, **payload: Any) -> None:
        self._output.update(payload)

    @property
    def output(self) -> dict[str, Any]:
        return self._output


class TraceRecorder:
    """Records one pipeline trace. Safe to use from sync and async code."""

    def __init__(
        self,
        trace_id: str,
        *,
        question: str,
        mode: str,
        workspace_id: str | None = None,
        matter_id: str | None = None,
        parent_trace_id: str | None = None,
    ) -> None:
        self.trace_id = trace_id
        self.question = question
        self.mode = mode
        self.workspace_id = workspace_id
        self.matter_id = matter_id
        self.parent_trace_id = parent_trace_id
        self._ordinal = 0
        self._started = utcnow()
        set_trace_id(trace_id)

    # ------------------------------------------------------------------ trace
    def open(self) -> None:
        with session_scope() as session:
            session.add(
                PipelineTrace(
                    id=self.trace_id,
                    workspace_id=self.workspace_id,
                    matter_id=self.matter_id,
                    mode=self.mode,
                    question=self.question,
                    status="running",
                    started_at=self._started,
                    capability_snapshot=capability_matrix(),
                    parent_trace_id=self.parent_trace_id,
                )
            )
        event_bus.publish(
            self.trace_id,
            {"type": "trace_started", "trace_id": self.trace_id, "question": self.question},
        )
        logger.info("trace_opened", mode=self.mode, matter_id=self.matter_id)

    def close(
        self,
        *,
        status: str = "success",
        error: str | None = None,
        answer: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        completed = utcnow()
        total_ms = (completed - self._started).total_seconds() * 1000
        with session_scope() as session:
            trace = session.get(PipelineTrace, self.trace_id)
            if trace is not None:
                trace.status = status
                trace.error = error
                trace.completed_at = completed
                trace.total_ms = total_ms
                if answer is not None:
                    trace.answer = _json_safe(answer)
                if summary is not None:
                    trace.summary = _json_safe(summary)
        event_bus.publish(
            self.trace_id,
            {
                "type": "trace_completed",
                "trace_id": self.trace_id,
                "status": status,
                "total_ms": total_ms,
                "error": error,
            },
        )
        logger.info("trace_closed", status=status, total_ms=round(total_ms, 1))

    # ------------------------------------------------------------------ stage
    def start_stage(
        self,
        stage_key: StageKey | str,
        *,
        service: str,
        transport: str | None = None,
        label: str | None = None,
        input: dict[str, Any] | None = None,
    ) -> StageHandle:
        self._ordinal += 1
        key = stage_key.value if isinstance(stage_key, StageKey) else str(stage_key)
        resolved_label = label or STAGE_LABELS.get(StageKey(key), key) if _is_stage(key) else key
        stage_id = new_id("stg_")
        transport_value = transport or get_settings().service_transport
        with session_scope() as session:
            session.add(
                TraceStage(
                    id=stage_id,
                    trace_id=self.trace_id,
                    ordinal=self._ordinal,
                    stage_key=key,
                    label=resolved_label,
                    service=service,
                    transport=transport_value,
                    status=StageStatus.RUNNING.value,
                    started_at=utcnow(),
                    input=_json_safe(input or {}),
                )
            )
        event_bus.publish(
            self.trace_id,
            {
                "type": "stage_started",
                "trace_id": self.trace_id,
                "stage_id": stage_id,
                "stage_key": key,
                "label": resolved_label,
                "service": service,
                "transport": transport_value,
                "ordinal": self._ordinal,
            },
        )
        return StageHandle(self, stage_id, key, self._ordinal)

    def finish_stage(
        self,
        handle: StageHandle,
        *,
        status: StageStatus = StageStatus.SUCCESS,
        output: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        merged = {**handle.output, **(output or {})}
        completed = utcnow()
        with session_scope() as session:
            stage = session.get(TraceStage, handle.stage_id)
            if stage is not None:
                started = stage.started_at
                stage.status = status.value
                stage.output = _json_safe(merged)
                stage.error = error
                stage.completed_at = completed
                stage.duration_ms = (completed - started).total_seconds() * 1000
                duration = stage.duration_ms
            else:  # pragma: no cover - defensive
                duration = None
        event_bus.publish(
            self.trace_id,
            {
                "type": "stage_finished",
                "trace_id": self.trace_id,
                "stage_id": handle.stage_id,
                "stage_key": handle.stage_key,
                "status": status.value,
                "duration_ms": duration,
                "error": error,
            },
        )

    @contextmanager
    def stage(
        self,
        stage_key: StageKey | str,
        *,
        service: str,
        transport: str | None = None,
        label: str | None = None,
        input: dict[str, Any] | None = None,
    ) -> Iterator[StageHandle]:
        """Context manager that records success or failure automatically.

        Failures are re-raised after being recorded, so a broken stage is visible in the UI
        instead of silently disappearing.
        """
        handle = self.start_stage(
            stage_key, service=service, transport=transport, label=label, input=input
        )
        try:
            yield handle
        except Exception as exc:
            self.finish_stage(handle, status=StageStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            self.finish_stage(handle, status=StageStatus.SUCCESS)

    def skip_stage(
        self,
        stage_key: StageKey | str,
        *,
        service: str,
        reason: str,
    ) -> None:
        """Record a stage that was deliberately not run (e.g. retrieval when RAG is off)."""
        handle = self.start_stage(stage_key, service=service, input={"reason": reason})
        self.finish_stage(handle, status=StageStatus.SKIPPED, output={"reason": reason})


def _is_stage(key: str) -> bool:
    try:
        StageKey(key)
    except ValueError:
        return False
    return True


def serialise_payload(value: Any) -> dict[str, Any]:
    """Best-effort conversion of a contract model into a trace payload."""
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))  # type: ignore[no-any-return]
    if isinstance(value, dict):
        return _json_safe(value)  # type: ignore[no-any-return]
    return {"value": _json_safe(value)}


def compact_json(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))
