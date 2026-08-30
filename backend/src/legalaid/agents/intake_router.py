"""Agent 0 - Intake Router.

Runs before any retrieval. It answers two questions the rest of the pipeline depends on:

1. **What does the user actually want?** Free-form input might be a question, a pasted FIR, a
   legal notice, or a question *about* an attached document. Each needs a different set of
   agents, and running all of them every time wastes 30-60 s per unnecessary agent at ~12 tok/s.

2. **What text should be embedded for the vector search?** This matters more than it looks.
   Embedding the raw user input means embedding the conversational wrapper too - "Hi, can you
   please help me, my client Mr Sharma is very worried about..." - which dilutes the legal
   signal in a 768-dimensional vector. The router rewrites the input into a dense paragraph of
   legal substance, and *that* is what gets matched.

The rewritten query is shown in the UI next to the original, so the transformation is visible
rather than a hidden preprocessing step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from legalaid.agents.base import Agent, AgentContext, fence_untrusted
from legalaid.domain.enums import AgentKey, DraftType

AGENT_MENU = """
- legal_research: finds applicable Acts, sections, rules and regulations
- similar_cases: finds prior Indian judgments that are factually or legally similar
- drafting: produces a lawyer-reviewable draft (reply, submission, summary, arguments)
- document_understanding: extracts parties, dates, amounts and obligations from a document
""".strip()


@dataclass(slots=True)
class RoutingDecision:
    """The plan the orchestrator executes, plus the query it will embed."""

    input_kind: str = "question"
    document_type: str = "none"
    task: str = "full_research"
    needs_law: bool = True
    needs_cases: bool = True
    needs_drafting: bool = False
    needs_document_facts: bool = False
    # Fails closed. A default of True meant any router failure waved every input through the
    # scope gate, which is the wrong direction for a system that must decline out-of-domain
    # questions. Callers that know the input is legal set it explicitly.
    is_legal_matter: bool = False
    search_query: str = ""
    legal_concepts: list[str] = field(default_factory=list)
    suggested_draft_type: str = "none"
    reasoning: str = ""
    original_input: str = ""
    used_fallback: bool = False

    @property
    def draft_type(self) -> DraftType:
        mapping = {
            "notice_reply": DraftType.NOTICE_REPLY,
            "petitioner_arguments": DraftType.PETITIONER_ARGUMENTS,
            "written_submission": DraftType.WRITTEN_SUBMISSION,
            "case_summary": DraftType.CASE_SUMMARY,
            "research_questions": DraftType.RESEARCH_QUESTIONS,
        }
        return mapping.get(self.suggested_draft_type, DraftType.CASE_SUMMARY)

    @property
    def agents_requested(self) -> list[str]:
        out: list[str] = []
        if self.needs_document_facts:
            out.append("document_understanding")
        if self.needs_law:
            out.append("legal_research")
        if self.needs_cases:
            out.append("similar_cases")
        if self.needs_drafting:
            out.append("drafting")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_kind": self.input_kind,
            "document_type": self.document_type,
            "task": self.task,
            "needs_law": self.needs_law,
            "needs_cases": self.needs_cases,
            "needs_drafting": self.needs_drafting,
            "needs_document_facts": self.needs_document_facts,
            "is_legal_matter": self.is_legal_matter,
            "original_input": self.original_input,
            "search_query": self.search_query,
            "legal_concepts": self.legal_concepts,
            "suggested_draft_type": self.suggested_draft_type,
            "agents_requested": self.agents_requested,
            "reasoning": self.reasoning,
            "used_fallback": self.used_fallback,
        }


class IntakeRouterAgent(Agent):
    agent_key = AgentKey.INTAKE_ROUTER
    prompt_name = "intake_router"

    def build_context(
        self,
        *,
        user_input: str,
        document_summaries: list[str] | None = None,
    ) -> AgentContext:
        context = AgentContext(agent_key=self.agent_key, question=user_input)
        context.extra["user_input"] = user_input
        context.extra["document_summaries"] = document_summaries or []
        return context

    def render(self, context: AgentContext) -> dict[str, Any]:
        docs = [fence_untrusted(str(d), tag="attachment") for d in
                (context.extra.get("document_summaries") or [])]
        return {
            # Fenced: this is the rawest untrusted input in the system, and the router's
            # decisions gate every downstream agent, so an injected instruction here would
            # have the widest reach of anywhere in the pipeline.
            "user_input": fence_untrusted(
                context.extra["user_input"][:4000], tag="user_request"
            ),
            "attached_documents": (
                "\n".join(f"- {d}" for d in docs) if docs else "(no documents attached)"
            ),
            "available_agents": AGENT_MENU,
        }

    def validate(self, output: dict[str, Any], context: AgentContext) -> list[str]:
        problems: list[str] = []
        query = str(output.get("search_query") or "").strip()
        if len(query) < 12:
            problems.append(
                "'search_query' is too short to retrieve anything useful; write 25-60 words "
                "of the legal substance."
            )
        if not any(
            output.get(flag)
            for flag in ("needs_law", "needs_cases", "needs_drafting", "needs_document_facts")
        ) and output.get("is_legal_matter"):
            problems.append(
                "a legal matter must require at least one agent; set needs_law or needs_cases."
            )
        return problems


def decision_from(output: dict[str, Any], *, user_input: str) -> RoutingDecision:
    """Build a RoutingDecision, falling back safely when the model output is unusable."""
    if not output:
        return fallback_decision(user_input)

    query = str(output.get("search_query") or "").strip()
    if len(query) < 12:
        # A useless query would silently poison retrieval, so prefer the raw input.
        query = user_input.strip()

    return RoutingDecision(
        input_kind=str(output.get("input_kind") or "question"),
        document_type=str(output.get("document_type") or "none"),
        task=str(output.get("task") or "full_research"),
        needs_law=bool(output.get("needs_law", True)),
        needs_cases=bool(output.get("needs_cases", True)),
        needs_drafting=bool(output.get("needs_drafting", False)),
        needs_document_facts=bool(output.get("needs_document_facts", False)),
        # Absent means "the model did not assert this", which is not the same as "yes".
        is_legal_matter=bool(output.get("is_legal_matter", False)),
        search_query=query,
        legal_concepts=[str(c) for c in (output.get("legal_concepts") or [])][:8],
        suggested_draft_type=str(output.get("suggested_draft_type") or "none"),
        reasoning=str(output.get("reasoning") or ""),
        original_input=user_input,
    )


def fallback_decision(
    user_input: str,
    *,
    has_documents: bool = False,
    is_legal_matter: bool = True,
) -> RoutingDecision:
    """Deterministic routing when the router itself fails.

    The pipeline must never stall because the smallest agent had a bad generation, so this
    falls back to the broadest useful plan using the raw input as the search query. It is
    marked `used_fallback` so the UI shows that routing was not model-driven for this run.

    `is_legal_matter` is supplied by the caller rather than assumed, because when the LLM
    router has failed the only trustworthy scope signal is the deterministic guardrail
    (`core.guardrails`), not this function's guesswork.
    """
    lowered = user_input.lower()
    wants_draft = any(
        word in lowered
        for word in ("draft", "reply to", "prepare a", "write a", "notice reply", "submission")
    )
    return RoutingDecision(
        is_legal_matter=is_legal_matter,
        input_kind="question_with_document" if has_documents else "question",
        task="full_research",
        needs_law=True,
        needs_cases=True,
        needs_drafting=wants_draft,
        needs_document_facts=has_documents,
        search_query=user_input.strip(),
        suggested_draft_type="notice_reply" if "notice" in lowered else "case_summary",
        reasoning="Router unavailable or returned unusable output; using the full pipeline "
        "with the original text as the search query.",
        original_input=user_input,
        used_fallback=True,
    )
