"""Agent 2 - Similar Case Agent.

Finds and assesses precedents. Sees case-law chunks plus a small number of law provisions for
grounding; matter documents are withheld because its job is comparing the legal issue against
reported judgments, not re-reading the client's file.
"""

from __future__ import annotations

from typing import Any

from legalaid.agents.base import Agent, AgentContext, format_chunks
from legalaid.domain.contracts import RetrievedChunk
from legalaid.domain.enums import LAW_COLLECTIONS, AgentKey, Collection


class SimilarCaseAgent(Agent):
    agent_key = AgentKey.SIMILAR_CASES
    prompt_name = "similar_cases"

    def build_context(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        matter_facts: str = "",
        legal_issues: list[str] | None = None,
        # One chunk per judgment, four judgments: enough to compare precedents without
        # pushing the prompt past a workable latency at ~12 tok/s.
        max_case_chunks: int = 4,
        max_law_chunks: int = 2,
    ) -> AgentContext:
        context = AgentContext(agent_key=self.agent_key, question=question)
        context.extra["matter_facts"] = matter_facts
        context.extra["legal_issues"] = legal_issues or []

        case_taken = 0
        law_taken = 0
        seen_cases: set[str] = set()
        for chunk in chunks:
            if chunk.collection is Collection.CASE_LAW:
                # One chunk per judgment keeps six *different* precedents in view rather
                # than six fragments of the same one.
                case_key = str(chunk.metadata.get("case_db_id") or chunk.source_title)
                if case_key in seen_cases:
                    context.exclude(
                        chunk, kind="case", reason="another chunk of this judgment is already included"
                    )
                    continue
                if case_taken < max_case_chunks:
                    seen_cases.add(case_key)
                    context.include(chunk, kind="case", reason="precedent candidate")
                    case_taken += 1
                else:
                    context.exclude(
                        chunk, kind="case", reason=f"beyond top-{max_case_chunks} case budget"
                    )
            elif chunk.collection in LAW_COLLECTIONS:
                if law_taken < max_law_chunks:
                    context.include(
                        chunk, kind="law", reason="provision context for comparison"
                    )
                    law_taken += 1
                else:
                    context.exclude(chunk, kind="law", reason="beyond provision budget")
            else:
                context.exclude(
                    chunk,
                    kind="user_document",
                    reason="matter documents are out of scope for the Similar Case Agent",
                )
        return context

    def render(self, context: AgentContext) -> dict[str, Any]:
        issues = context.extra.get("legal_issues") or []
        return {
            "question": context.question,
            "matter_facts": context.extra.get("matter_facts") or "(no facts extracted)",
            "legal_issues": "\n".join(f"- {i}" for i in issues) if issues else "(none identified)",
            "case_context": context.extra["case_context"],
        }

    def validate(self, output: dict[str, Any], context: AgentContext) -> list[str]:
        problems: list[str] = []
        allowed = set(context.allowed_chunk_ids)
        cases = output.get("cases")
        if not isinstance(cases, list):
            return ["'cases' must be a list"]

        # Titles and citations actually supplied, so a hallucinated precedent is detectable.
        supplied_titles = {
            str(i.label).lower() for i in context.included if i.kind == "case"
        }
        for entry in cases:
            if not isinstance(entry, dict):
                problems.append("each case must be an object")
                continue
            chunk_id = entry.get("chunk_id")
            if not chunk_id:
                problems.append(
                    f"case {entry.get('case_title')!r} has no chunk_id"
                )
            elif chunk_id not in allowed:
                problems.append(
                    f"case {entry.get('case_title')!r} cites chunk_id {chunk_id!r}, which was "
                    "not supplied. Fabricated precedent is unacceptable."
                )
            title = str(entry.get("case_title") or "").lower()
            if title and not any(title[:24] in s or s[:24] in title for s in supplied_titles):
                problems.append(
                    f"case title {entry.get('case_title')!r} does not match any supplied "
                    "judgment. Copy titles exactly from CASE CONTEXT."
                )
        return problems


def prepare(
    agent: SimilarCaseAgent,
    *,
    question: str,
    chunks: list[RetrievedChunk],
    matter_facts: str = "",
    legal_issues: list[str] | None = None,
) -> AgentContext:
    context = agent.build_context(
        question=question,
        chunks=chunks,
        matter_facts=matter_facts,
        legal_issues=legal_issues,
    )
    by_id = {c.chunk_id: c for c in chunks}
    case_chunks = [
        by_id[i.chunk_id] for i in context.included if i.kind == "case" and i.chunk_id in by_id
    ]
    law_chunks = [
        by_id[i.chunk_id] for i in context.included if i.kind == "law" and i.chunk_id in by_id
    ]
    case_block = format_chunks(case_chunks, kind="case")
    if law_chunks:
        case_block += "\n\nPROVISIONS FOR REFERENCE\n" + format_chunks(law_chunks, kind="law")
    context.extra["case_context"] = case_block
    return context
