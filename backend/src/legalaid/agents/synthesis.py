"""Agent 6 - Synthesis Agent.

The pipeline used to end with structured findings: a list of sections and a list of cases. That
is research output, not an answer. A lawyer asking "what provisions apply and are there similar
cases?" expects a colleague's reply - the answer, the reasoning, what is uncertain, and what to
do next.

This agent produces that reply. Like the drafting agent it sees **no raw chunks**, only the
verified findings of the agents before it, so it cannot introduce an authority that was never
retrieved. Its own reasoning is reported separately under `inference`, which keeps the boundary
between what a source says and what the system concluded visible in the UI.
"""

from __future__ import annotations

import json
from typing import Any

from legalaid.agents.base import Agent, AgentContext, ContextItem
from legalaid.agents.drafting import _authority_key
from legalaid.domain.enums import AgentKey


class SynthesisAgent(Agent):
    agent_key = AgentKey.SYNTHESIS
    prompt_name = "synthesis"
    # This is the text the lawyer actually reads, so it is the most important thing in the
    # system to audit for invented authority.
    prose_fields = ("answer_markdown",)

    def build_context(
        self,
        *,
        question: str,
        facts: dict[str, Any] | None,
        legal_findings: dict[str, Any],
        case_findings: dict[str, Any],
        draft: dict[str, Any] | None = None,
        upstream_items: list[ContextItem] | None = None,
    ) -> AgentContext:
        context = AgentContext(agent_key=self.agent_key, question=question)
        context.extra.update(
            {
                "facts": facts or {},
                "legal_findings": legal_findings or {},
                "case_findings": case_findings or {},
                "draft": draft or {},
                "allowed_authorities": _collect_authorities(legal_findings, case_findings),
            }
        )
        # Record the evidence chain even though the agent never sees the chunks themselves.
        # `source_label` preserves each item's real source label, which is what lets the
        # grounding audit check sections and case names quoted in `answer_markdown`.
        for item in upstream_items or []:
            context.included.append(
                ContextItem(
                    chunk_id=item.chunk_id,
                    kind="upstream_case" if item.kind == "case" else "upstream_evidence",
                    label=f"evidence behind the findings ({item.chunk_id})",
                    source_label=item.evidence_label,
                    reason="cited by an upstream research agent",
                )
            )
        return context

    def render(self, context: AgentContext) -> dict[str, Any]:
        return {
            "question": context.question,
            "facts": _render_facts(context.extra["facts"]),
            "legal_findings": _render_legal(context.extra["legal_findings"]),
            "case_findings": _render_cases(context.extra["case_findings"]),
            "draft_summary": _render_draft(context.extra["draft"]),
        }

    def validate(self, output: dict[str, Any], context: AgentContext) -> list[str]:
        problems: list[str] = []
        answer = str(output.get("answer_markdown") or "").strip()
        if len(answer) < 80:
            problems.append(
                "'answer_markdown' is too short to be a useful answer; write 2-5 paragraphs."
            )

        allowed = context.extra.get("allowed_authorities") or []
        if allowed:
            allowed_keys = {_authority_key(a) for a in allowed}
            allowed_keys.discard("")
            for authority in output.get("authorities_cited") or []:
                if _authority_key(str(authority)) not in allowed_keys:
                    problems.append(
                        f"authority {authority!r} was not established by the research agents."
                    )
        return problems


def _collect_authorities(legal: dict[str, Any], cases: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for section in (legal or {}).get("sections") or []:
        if isinstance(section, dict):
            label = f"{section.get('section', '')} {section.get('act', '')}".strip()
            if label:
                out.append(label)
    for case in (cases or {}).get("cases") or []:
        if isinstance(case, dict):
            for key in ("case_title", "citation"):
                if case.get(key):
                    out.append(str(case[key]))
    return out


def _render_facts(facts: dict[str, Any]) -> str:
    if not facts:
        return "(no documents were supplied, so no facts have been verified)"
    lines: list[str] = []
    if facts.get("factual_summary"):
        lines.append(str(facts["factual_summary"]))
    for party in facts.get("parties") or []:
        lines.append(f"- party: {party.get('name')} ({party.get('role')})")
    for date in facts.get("key_dates") or []:
        lines.append(f"- date: {date.get('date')} — {date.get('event')}")
    for amount in facts.get("amounts") or []:
        lines.append(f"- amount: {amount.get('amount')} — {amount.get('what_for')}")
    for issue in facts.get("potential_issues") or []:
        lines.append(f"- issue: {issue}")
    return "\n".join(lines) or "(nothing extracted)"


def _render_legal(findings: dict[str, Any]) -> str:
    sections = (findings or {}).get("sections") or []
    if not sections:
        return "(the legal research agent found no applicable provisions in the retrieved material)"
    lines: list[str] = []
    for issue in findings.get("legal_issues") or []:
        lines.append(f"issue: {issue}")
    for section in sections:
        if not isinstance(section, dict):
            continue
        lines.append(
            f"{section.get('section')} of {section.get('act')} — "
            f"{section.get('what_it_provides')} [relevance: {section.get('relevance')}]"
        )
    for note in findings.get("uncertainties") or []:
        lines.append(f"uncertainty: {note}")
    if findings.get("evidence_sufficient") is False:
        lines.append("NOTE: the agent reported the statutory evidence as insufficient.")
    return "\n".join(lines)


def _render_cases(findings: dict[str, Any]) -> str:
    cases = (findings or {}).get("cases") or []
    if not cases:
        note = (findings or {}).get("note") or "no relevant precedents were identified"
        return f"(no precedents available: {note})"
    lines: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        lines.append(
            f"{case.get('case_title')} {case.get('citation')} "
            f"[{case.get('strength', 'unknown')} authority] — {case.get('relevant_principle')}"
        )
        for difference in case.get("distinguishing_factors") or []:
            lines.append(f"    distinguishable: {difference}")
    if findings.get("evidence_sufficient") is False:
        lines.append("NOTE: the agent reported the precedent match as weak.")
    return "\n".join(lines)


def _render_draft(draft: dict[str, Any]) -> str:
    if not draft or not draft.get("draft_markdown"):
        return "(no draft was requested)"
    return f"{draft.get('title', 'Draft')}: {str(draft['draft_markdown'])[:600]}"


def fallback_answer(
    *,
    question: str,
    legal: dict[str, Any],
    cases: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic answer if the synthesis model call fails.

    The lawyer must never be shown an empty result, so this assembles a plain summary from the
    structured findings. It is marked low confidence and flagged in the UI so nobody mistakes it
    for model-written analysis.
    """
    sections = [s for s in (legal or {}).get("sections") or [] if isinstance(s, dict)]
    precedents = [c for c in (cases or {}).get("cases") or [] if isinstance(c, dict)]

    parts: list[str] = []
    if sections:
        listed = "; ".join(
            f"{s.get('section')} of {s.get('act')} ({s.get('what_it_provides')})"
            for s in sections[:4]
        )
        parts.append(f"The retrieved material points to {listed}.")
    else:
        parts.append(
            "The retrieved statutory material does not clearly answer this question."
        )
    if precedents:
        listed = "; ".join(
            f"{c.get('case_title')} {c.get('citation')}" for c in precedents[:3]
        )
        parts.append(f"Potentially comparable judgments: {listed}.")
    else:
        parts.append("No sufficiently similar judgment was found in the curated case corpus.")
    parts.append(
        "This summary was assembled directly from the structured findings because the synthesis "
        "step did not complete. Review the agent outputs below."
    )

    return {
        "answer_markdown": "\n\n".join(parts),
        "key_points": [
            f"{s.get('section')} of {s.get('act')}" for s in sections[:4]
        ]
        or ["No applicable provision was identified in the retrieved material."],
        "inference": [],
        "uncertainties": [
            "Synthesis did not complete, so no legal reasoning has been applied to these findings."
        ],
        "next_steps": [
            "Re-run the query, or review the Legal Research and Similar Case outputs directly."
        ],
        "authorities_cited": _collect_authorities(legal, cases)[:8],
        "confidence": "low",
        "used_fallback": True,
    }
