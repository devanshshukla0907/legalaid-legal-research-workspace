"""LLM service logic.

Records a `ModelRun` row for every single call - provider, model, temperature, max tokens,
the exact system and user prompts, token counts, latency and status. That persisted record is
what the "What did the model see?" panel reads, so the UI shows real data rather than a
reconstruction.
"""

from __future__ import annotations

from legalaid.config import get_settings
from legalaid.core.errors import LLMUnavailableError
from legalaid.core.logging import get_logger
from legalaid.core.security import GenerationSlot
from legalaid.core.transport import register_endpoint
from legalaid.db.base import new_id, session_scope, utcnow
from legalaid.db.models import ModelRun
from legalaid.domain.contracts import (
    LLMGenerateRequest,
    LLMGenerateResponse,
    LLMStatus,
    ServiceHealth,
)
from legalaid.services.llm.ollama_client import OllamaClient

logger = get_logger(__name__)

SERVICE_NAME = "llm-service"


def _record_model_run(
    *,
    model_run_id: str,
    request: LLMGenerateRequest,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    raw_response: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    total_duration_ms: float,
    load_ms: float | None,
    eval_ms: float | None,
    tokens_per_second: float | None,
    status: str,
    error: str | None,
) -> None:
    with session_scope() as session:
        session.add(
            ModelRun(
                id=model_run_id,
                trace_id=request.trace_id,
                agent_run_id=request.agent_run_id,
                agent_key=request.agent_key.value,
                provider=provider,
                model=model,
                role="generation",
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
                raw_response=raw_response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_duration_ms=total_duration_ms,
                load_ms=load_ms,
                eval_ms=eval_ms,
                tokens_per_second=tokens_per_second,
                prompt_name=request.prompt_name,
                prompt_version=request.prompt_version,
                status=status,
                error=error,
                request_id=model_run_id,
                created_at=utcnow(),
            )
        )


@register_endpoint(SERVICE_NAME, "POST", "/generate", LLMGenerateResponse, LLMGenerateRequest)
async def generate(request: LLMGenerateRequest) -> LLMGenerateResponse:
    """Generate text with Ollama + the configured Code Llama model."""
    settings = get_settings()
    client = OllamaClient()
    model_run_id = new_id("mdl_")
    model_name = request.model or settings.llm_model
    temperature = (
        request.temperature if request.temperature is not None else settings.llm_temperature
    )
    max_tokens = request.max_tokens or settings.llm_max_tokens

    try:
        # One generation on the GPU at a time. Concurrent generations on a 6 GB card do not
        # run in parallel usefully - the model stops fully offloading and every caller gets
        # slower - so queueing here is the faster option as well as the safer one. Enforced in
        # the LLM service rather than at the endpoint so every caller is covered, including
        # agents, the compare endpoint and the evaluation harness.
        async with GenerationSlot():
            generation = await client.generate(
                prompt=request.user_prompt,
                system=request.system_prompt,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=request.json_mode,
                json_schema=request.json_schema,
                stop=request.stop,
            )
    except LLMUnavailableError as exc:
        _record_model_run(
            model_run_id=model_run_id,
            request=request,
            provider=settings.llm_provider,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            raw_response=None,
            input_tokens=None,
            output_tokens=None,
            total_duration_ms=0.0,
            load_ms=None,
            eval_ms=None,
            tokens_per_second=None,
            status="failed",
            error=exc.message,
        )
        logger.error("llm_generate_failed", model=model_name, error=exc.message)
        raise

    _record_model_run(
        model_run_id=model_run_id,
        request=request,
        provider=settings.llm_provider,
        model=generation.model,
        temperature=temperature,
        max_tokens=max_tokens,
        raw_response=generation.text,
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
        total_duration_ms=generation.total_duration_ms,
        load_ms=generation.load_ms,
        eval_ms=generation.eval_ms,
        tokens_per_second=generation.tokens_per_second,
        status="success",
        error=None,
    )

    logger.info(
        "llm_generate",
        agent=request.agent_key.value,
        model=generation.model,
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
        latency_ms=round(generation.total_duration_ms, 1),
    )

    return LLMGenerateResponse(
        model_run_id=model_run_id,
        provider=settings.llm_provider,
        model=generation.model,
        text=generation.text,
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
        total_duration_ms=generation.total_duration_ms,
        load_ms=generation.load_ms,
        eval_ms=generation.eval_ms,
        tokens_per_second=generation.tokens_per_second,
        temperature=temperature,
        max_tokens=max_tokens,
        status="success",
        created_at=generation.created_at,
    )


@register_endpoint(SERVICE_NAME, "GET", "/status", LLMStatus)
async def status() -> LLMStatus:
    return await OllamaClient().status()


@register_endpoint(SERVICE_NAME, "GET", "/health", ServiceHealth)
async def health() -> ServiceHealth:
    settings = get_settings()
    llm_status = await OllamaClient().status()
    return ServiceHealth(
        service=SERVICE_NAME,
        status="ok" if llm_status.reachable else "degraded",
        transport=settings.service_transport,
        detail={
            "ollama_reachable": llm_status.reachable,
            "base_url": llm_status.base_url,
            "llm_model": llm_status.configured_llm_model,
            "llm_model_available": llm_status.llm_model_available,
            "embedding_model": llm_status.configured_embedding_model,
            "embedding_model_available": llm_status.embedding_model_available,
            "error": llm_status.error,
        },
    )
