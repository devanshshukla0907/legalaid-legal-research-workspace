"""Agent 1 - Legal Research Agent.

Identifies applicable statutory provisions. Sees law chunks and matter-document chunks; case
law is withheld deliberately, because its job is statute identification and precedent would
invite it to reason about outcomes instead.
"""

from __future__ import annotations

from typing import Any

from legalaid.agents.base import Agent, AgentContext, fence_untrusted, format_chunks
from legalaid.domain.contracts import RetrievedChunk
from legalaid.domain.enums import LAW_COLLECTIONS, AgentKey, Collection


class LegalResearchAgent(Agent):
    agent_key = AgentKey.LEGAL_RESEARCH
    prompt_name = "legal_research"

    def build_context(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        matter_summary: str = "",
        # Budgets are tight on purpose. At ~12 tok/s on this hardware, prompt tokens are the
        # dominant cost of the whole pipeline, and 5 well-ranked provisions answer these
        # questions as well as 8 did while keeping the agent inside a usable latency.
        max_law_chunks: int = 5,
        max_doc_chunks: int = 2,
    ) -> AgentContext:
        context = AgentContext(agent_key=self.agent_key, question=question)
        context.extra["matter_summary"] = matter_summary

        law_taken = 0
        doc_taken = 0
        for chunk in chunks:
            if chunk.collection in LAW_COLLECTIONS:
                if law_taken < max_law_chunks:
                    context.include(chunk, kind="law", reason="applicable-law candidate")
                    law_taken += 1
                else:
                    context.exclude(
                        chunk, kind="law", reason=f"beyond top-{max_law_chunks} law budget"
                    )
            elif chunk.collection is Collection.USER_DOCUMENTS:
                if doc_taken < max_doc_chunks:
                    context.include(
                        chunk, kind="user_document", reason="matter fact context"
                    )
                    doc_taken += 1
                else:
                    context.exclude(
                        chunk, kind="user_document", reason="beyond document budget"
                    )
            else:
                # Case law is withheld on purpose - this agent identifies statute, and
                # precedent would pull it toward predicting outcomes.
                context.exclude(
                    chunk,
                    kind="case",
                    reason="case law is out of scope for the Legal Research Agent",
                )
        return context

    def render(self, context: AgentContext) -> dict[str, Any]:
        law = [c for c in _chunks(context) if c.kind == "law"]
        docs = [c for c in _chunks(context) if c.kind == "user_document"]
        return {
            "question": context.question,
            "matter_summary": context.extra.get("matter_summary") or "(no matter summary)",
            "law_context": context.extra["law_context"],
            "document_context": context.extra["document_context"],
        }

    def validate(self, output: dict[str, Any], context: AgentContext) -> list[str]:
        problems: list[str] = []
        allowed = set(context.allowed_chunk_ids)
        sections = output.get("sections")
        if not isinstance(sections, list):
            return ["'sections' must be a list"]
        for entry in sections:
            if not isinstance(entry, dict):
                problems.append("each section must be an object")
                continue
            chunk_id = entry.get("chunk_id")
            if not chunk_id:
                problems.append(
                    f"section {entry.get('section')!r} has no chunk_id - every citation "
                    "must name its source chunk"
                )
            elif chunk_id not in allowed:
                problems.append(
                    f"section {entry.get('section')!r} cites chunk_id {chunk_id!r}, which was "
                    "not supplied. Use only the chunk_ids in LAW CONTEXT."
                )
        return problems


def _chunks(context: AgentContext) -> list[Any]:
    return context.included


def prepare(
    agent: LegalResearchAgent,
    *,
    question: str,
    chunks: list[RetrievedChunk],
    matter_summary: str = "",
) -> AgentContext:
    """Build the context and pre-render the prompt sections."""
    context = agent.build_context(
        question=question, chunks=chunks, matter_summary=matter_summary
    )
    by_id = {c.chunk_id: c for c in chunks}
    law_chunks = [
        by_id[i.chunk_id] for i in context.included if i.kind == "law" and i.chunk_id in by_id
    ]
    doc_chunks = [
        by_id[i.chunk_id]
        for i in context.included
        if i.kind == "user_document" and i.chunk_id in by_id
    ]
    context.extra["law_context"] = format_chunks(law_chunks, kind="law")
    # Statutory text is trusted (it came from the ingested corpus); user documents are not.
    context.extra["document_context"] = fence_untrusted(
        format_chunks(doc_chunks, kind="document")
    )
    return context
