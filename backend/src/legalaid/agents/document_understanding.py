"""Agent 4 - Document Understanding Agent.

Runs first when a matter has documents. Extracts parties, dates, amounts and obligations from
the uploaded file text, each tagged with the chunk it came from.

Its output becomes the "verified facts" every downstream agent is permitted to rely on, which
is what stops the drafting agent inventing a party name or a date. It sees only matter
documents - no statute, no case law - so it cannot blur what the file says with what the law
provides.
"""

from __future__ import annotations

from typing import Any

from legalaid.agents.base import Agent, AgentContext, fence_untrusted, format_chunks
from legalaid.domain.contracts import RetrievedChunk
from legalaid.domain.enums import AgentKey, Collection


class DocumentUnderstandingAgent(Agent):
    agent_key = AgentKey.DOCUMENT_UNDERSTANDING
    prompt_name = "document_understanding"

    def build_context(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        matter_title: str = "",
        max_chunks: int = 6,
    ) -> AgentContext:
        context = AgentContext(agent_key=self.agent_key, question=question)
        context.extra["matter_title"] = matter_title or "(untitled matter)"

        taken = 0
        for chunk in chunks:
            if chunk.collection is not Collection.USER_DOCUMENTS:
                context.exclude(
                    chunk,
                    kind="knowledge",
                    reason="only matter documents are in scope for fact extraction",
                )
                continue
            if taken < max_chunks:
                context.include(chunk, kind="user_document", reason="matter document text")
                taken += 1
            else:
                context.exclude(
                    chunk, kind="user_document", reason=f"beyond {max_chunks}-chunk budget"
                )
        return context

    def render(self, context: AgentContext) -> dict[str, Any]:
        return {
            "matter_title": context.extra["matter_title"],
            "document_context": context.extra["document_context"],
        }

    def validate(self, output: dict[str, Any], context: AgentContext) -> list[str]:
        """Every extracted fact must name the chunk it came from, or it is unverifiable."""
        problems: list[str] = []
        allowed = set(context.allowed_chunk_ids)
        for field in ("parties", "key_dates", "amounts", "obligations"):
            entries = output.get(field)
            if entries is None:
                continue
            if not isinstance(entries, list):
                problems.append(f"'{field}' must be a list")
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                chunk_id = entry.get("chunk_id")
                if chunk_id and chunk_id not in allowed:
                    problems.append(
                        f"{field} entry cites chunk_id {chunk_id!r}, which was not supplied"
                    )
        return problems


def prepare(
    agent: DocumentUnderstandingAgent,
    *,
    question: str,
    chunks: list[RetrievedChunk],
    matter_title: str = "",
) -> AgentContext:
    context = agent.build_context(
        question=question, chunks=chunks, matter_title=matter_title
    )
    by_id = {c.chunk_id: c for c in chunks}
    doc_chunks = [by_id[i.chunk_id] for i in context.included if i.chunk_id in by_id]
    # This agent exists to read untrusted documents, so the fence matters most here.
    context.extra["document_context"] = fence_untrusted(
        format_chunks(doc_chunks, kind="document")
    )
    return context
