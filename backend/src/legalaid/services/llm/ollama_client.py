"""Ollama client.

This is the only place in the codebase that speaks to Ollama. Code Llama is the sole
generation path; there is no fallback to a hosted provider, because the application is
required to demonstrate Ollama + Code Llama specifically. If Ollama is unavailable the error
is reported honestly and surfaced in the UI.

Token counts, load time and eval time come from Ollama's own response fields, so the figures
shown in the Models and Pipeline pages are measured rather than estimated.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import httpx

from legalaid.config import get_settings
from legalaid.core.errors import LLMUnavailableError
from legalaid.core.logging import get_logger
from legalaid.db.base import utcnow
from legalaid.domain.contracts import LLMStatus, OllamaModelInfo

logger = get_logger(__name__)

NS_PER_MS = 1_000_000


class OllamaGeneration:
    """Normalised result of one Ollama generate call."""

    __slots__ = (
        "text",
        "model",
        "input_tokens",
        "output_tokens",
        "total_duration_ms",
        "load_ms",
        "eval_ms",
        "tokens_per_second",
        "created_at",
        "raw",
    )

    def __init__(
        self,
        *,
        text: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        total_duration_ms: float,
        load_ms: float | None,
        eval_ms: float | None,
        tokens_per_second: float | None,
        created_at: datetime,
        raw: dict[str, Any],
    ) -> None:
        self.text = text
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_duration_ms = total_duration_ms
        self.load_ms = load_ms
        self.eval_ms = eval_ms
        self.tokens_per_second = tokens_per_second
        self.created_at = created_at
        self.raw = raw


class OllamaClient:
    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._settings = settings

    # ------------------------------------------------------------------ status
    async def list_models(self) -> list[OllamaModelInfo]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()

        models: list[OllamaModelInfo] = []
        for entry in payload.get("models", []):
            details = entry.get("details") or {}
            models.append(
                OllamaModelInfo(
                    name=entry.get("name") or entry.get("model") or "unknown",
                    size_bytes=entry.get("size"),
                    parameter_size=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                    family=details.get("family"),
                    modified_at=entry.get("modified_at"),
                )
            )
        return models

    async def status(self) -> LLMStatus:
        settings = self._settings
        try:
            models = await self.list_models()
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return LLMStatus(
                provider=settings.llm_provider,
                base_url=self.base_url,
                reachable=False,
                configured_llm_model=settings.llm_model,
                configured_embedding_model=settings.embedding_model,
                llm_model_available=False,
                embedding_model_available=False,
                installed_models=[],
                error=(
                    f"Cannot reach Ollama at {self.base_url}: {exc}. "
                    "Install Ollama and run 'ollama serve'."
                ),
                checked_at=utcnow(),
            )

        installed = {m.name for m in models}
        return LLMStatus(
            provider=settings.llm_provider,
            base_url=self.base_url,
            reachable=True,
            configured_llm_model=settings.llm_model,
            configured_embedding_model=settings.embedding_model,
            llm_model_available=_model_present(settings.llm_model, installed),
            embedding_model_available=_model_present(settings.embedding_model, installed),
            installed_models=models,
            error=None,
            checked_at=utcnow(),
        )

    # -------------------------------------------------------------- generation
    async def generate(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
        stop: list[str] | None = None,
    ) -> OllamaGeneration:
        settings = self._settings
        model_name = model or settings.llm_model
        options: dict[str, Any] = {
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "num_predict": max_tokens or settings.llm_max_tokens,
            "num_ctx": settings.llm_num_ctx,
        }
        if stop:
            options["stop"] = stop

        body: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            body["system"] = system
        if json_schema is not None:
            # Ollama constrains decoding to a JSON Schema, which makes malformed or
            # off-schema output structurally impossible. This matters a great deal here:
            # Code Llama 7B is code-specialised, and when merely *asked* for a schema it
            # pattern-matched the retrieved statute instead, emitting {"73": [{"a": ...}]}.
            # Constrained decoding removes that failure mode at the decoder rather than
            # trying to repair it afterwards.
            body["format"] = json_schema
        elif json_mode:
            body["format"] = "json"

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=settings.llm_request_timeout) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=body)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Ollama request failed at {self.base_url}: {exc}",
                {
                    "base_url": self.base_url,
                    "model": model_name,
                    "hint": "Is Ollama running? Try 'ollama serve' and 'ollama list'.",
                },
            ) from exc

        if response.status_code == 404:
            raise LLMUnavailableError(
                f"Model '{model_name}' is not available in Ollama.",
                {
                    "model": model_name,
                    "hint": f"Run: ollama pull {model_name}",
                    "response": response.text[:500],
                },
            )
        if response.status_code >= 400:
            raise LLMUnavailableError(
                f"Ollama returned HTTP {response.status_code}",
                {"model": model_name, "response": response.text[:500]},
            )

        payload = response.json()
        wall_ms = (time.perf_counter() - started) * 1000

        total_ns = payload.get("total_duration")
        load_ns = payload.get("load_duration")
        eval_ns = payload.get("eval_duration")
        output_tokens = payload.get("eval_count")

        tokens_per_second: float | None = None
        if output_tokens and eval_ns:
            tokens_per_second = output_tokens / (eval_ns / 1_000_000_000)

        return OllamaGeneration(
            text=payload.get("response", ""),
            model=payload.get("model", model_name),
            input_tokens=payload.get("prompt_eval_count"),
            output_tokens=output_tokens,
            total_duration_ms=(total_ns / NS_PER_MS) if total_ns else wall_ms,
            load_ms=(load_ns / NS_PER_MS) if load_ns else None,
            eval_ms=(eval_ns / NS_PER_MS) if eval_ns else None,
            tokens_per_second=tokens_per_second,
            created_at=utcnow(),
            raw=payload,
        )

    # --------------------------------------------------------------- embedding
    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed texts via /api/embed, falling back to /api/embeddings on older Ollama."""
        settings = self._settings
        model_name = model or settings.embedding_model
        try:
            async with httpx.AsyncClient(timeout=settings.embedding_request_timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/embed", json={"model": model_name, "input": texts}
                )
                if response.status_code == 404:
                    return await self._embed_legacy(client, texts, model_name)
                if response.status_code >= 400:
                    raise LLMUnavailableError(
                        f"Ollama embedding returned HTTP {response.status_code}",
                        {
                            "model": model_name,
                            "hint": f"Run: ollama pull {model_name}",
                            "response": response.text[:500],
                        },
                    )
                payload = response.json()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Ollama embedding request failed at {self.base_url}: {exc}",
                {"model": model_name, "hint": "Is Ollama running?"},
            ) from exc

        vectors = payload.get("embeddings")
        if not vectors:
            raise LLMUnavailableError(
                "Ollama returned no embeddings", {"model": model_name, "payload_keys": list(payload)}
            )
        return [[float(v) for v in vector] for vector in vectors]

    async def _embed_legacy(
        self, client: httpx.AsyncClient, texts: list[str], model_name: str
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            response = await client.post(
                f"{self.base_url}/api/embeddings", json={"model": model_name, "prompt": text}
            )
            if response.status_code >= 400:
                raise LLMUnavailableError(
                    f"Ollama legacy embedding returned HTTP {response.status_code}",
                    {"model": model_name, "hint": f"Run: ollama pull {model_name}"},
                )
            vector = response.json().get("embedding")
            if not vector:
                raise LLMUnavailableError("Ollama returned an empty embedding", {"model": model_name})
            vectors.append([float(v) for v in vector])
        return vectors


def _model_present(configured: str, installed: set[str]) -> bool:
    """Ollama reports 'codellama:7b-instruct'; treat a bare name as matching ':latest'."""
    if configured in installed:
        return True
    if ":" not in configured and f"{configured}:latest" in installed:
        return True
    return any(name.split(":")[0] == configured.split(":")[0] and ":" not in configured
               for name in installed)
