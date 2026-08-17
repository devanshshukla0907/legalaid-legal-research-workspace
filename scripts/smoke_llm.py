"""Smoke-test the LLM service end to end: service -> Ollama -> Code Llama -> ModelRun row.

Usage: python scripts/smoke_llm.py
"""

from __future__ import annotations

import asyncio
import sys

from legalaid.db.base import create_all, session_scope
from legalaid.db.models import ModelRun, PipelineTrace
from legalaid.domain.contracts import LLMGenerateRequest
from legalaid.domain.enums import AgentKey
from legalaid.services.llm import service as llm


async def main() -> int:
    create_all()

    status = await llm.status()
    print("ollama reachable      :", status.reachable)
    print("base url              :", status.base_url)
    print("llm model             :", status.configured_llm_model, "available:", status.llm_model_available)
    print("embedding model       :", status.configured_embedding_model, "available:", status.embedding_model_available)
    if not status.reachable:
        print("ERROR:", status.error)
        return 1

    # A PipelineTrace row must exist first because ModelRun.trace_id references it.
    trace_id = "trace_smoke_llm"
    with session_scope() as s:
        if s.get(PipelineTrace, trace_id) is None:
            s.add(PipelineTrace(id=trace_id, question="smoke test", status="running"))

    req = LLMGenerateRequest(
        trace_id=trace_id,
        agent_key=AgentKey.ORCHESTRATOR,
        system_prompt="You are terse. Answer in one short sentence.",
        user_prompt="Under Indian contract law, what does Section 73 of the Indian Contract Act deal with?",
        max_tokens=60,
        prompt_name="smoke",
        prompt_version="v1",
    )
    print("\ngenerating (first call may take ~90s cold)...")
    resp = await llm.generate(req)
    print("model      :", resp.model)
    print("tokens     : in", resp.input_tokens, "out", resp.output_tokens)
    print("latency ms :", round(resp.total_duration_ms, 1), "| tok/s:", resp.tokens_per_second)
    print("text       :", (resp.text or "").strip()[:300])

    with session_scope() as s:
        row = s.get(ModelRun, resp.model_run_id)
        assert row is not None, "ModelRun was not persisted"
        print("\npersisted ModelRun:", row.id, "| status:", row.status, "| prompt:", row.prompt_name)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
