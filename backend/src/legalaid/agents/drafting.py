"""Agent 3 - Legal Drafting Agent.

Runs last, and sees *no raw chunks at all*: only verified facts and the two research agents'
structured outputs. That is deliberate. Giving a drafting model raw statutory text invites it to
paraphrase provisions into the draft as if they were its own findings, which is exactly how
fabricated authority gets into a document. Working from verified findings keeps every authority
in the draft traceable to an agent that cited a chunk for it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from legalaid.agents.base import Agent, AgentContext, ContextItem
from legalaid.domain.enums import AgentKey, DraftLength, DraftTone, DraftType


class DraftingAgent(Agent):
    agent_key = AgentKey.DRAFTING
    prompt_name = "drafting"
    # A draft goes out over a lawyer's name, so invented authority in its prose is audited
    # just as strictly as in the structured citation list.
    prose_fields = ("draft_markdown",)

    def build_context(
        self,
        *,
        question: str,
        verified_facts: dict[str, Any] | None,
        legal_findings: dict[str, Any],
        case_findings: dict[str, Any],
        draft_type: DraftType = DraftType.CASE_SUMMARY,
        tone: DraftTone = DraftTone.FORMAL,
        length: DraftLength = DraftLength.STANDARD,
        instructions: str = "",
        upstream_items: list[ContextItem] | None = None,
    ) -> AgentContext:
        context = AgentContext(agent_key=self.agent_key, question=question)
        context.extra.update(
            {
                "verified_facts": verified_facts or {},
                "legal_findings": legal_findings,
                "case_findings": case_findings,
                "draft_type": draft_type.value,
                "tone": tone.value,
                "length": length.value,
                "instructions": instructions or f"Produce a {draft_type.value.replace('_', ' ')}.",
                # Authorities the upstream agents actually cited; used by validation.
                "allowed_authorities": _authorities(legal_findings, case_findings),
            }
        )

        # Record the provenance chain: the evidence behind the findings this draft rests on.
        # The drafting agent never sees these chunks, but the trace must still show which
        # evidence ultimately supports the draft - and the grounding audit needs each item's
        # real source label to check a citation against.
        for item in upstream_items or []:
            context.included.append(_provenance_item(item))
        return context

    def render(self, context: AgentContext) -> dict[str, Any]:
        return {
            "draft_type": context.extra["draft_type"],
            "tone": context.extra["tone"],
            "length": context.extra["length"],
            "instructions": context.extra["instructions"],
            "verified_facts": _render_facts(context.extra["verified_facts"]),
            "legal_findings": _render_legal(context.extra["legal_findings"]),
            "case_findings": _render_cases(context.extra["case_findings"]),
        }

    def validate(self, output: dict[str, Any], context: AgentContext) -> list[str]:
        problems: list[str] = []
        draft = output.get("draft_markdown")
        if not isinstance(draft, str) or len(draft.strip()) < 40:
            problems.append("'draft_markdown' is missing or too short to be a usable draft")

        allowed = context.extra.get("allowed_authorities", [])
        if allowed:
            allowed_keys = {_authority_key(a) for a in allowed}
            allowed_keys.discard("")
            for authority in output.get("authorities_relied_on") or []:
                if _authority_key(str(authority)) not in allowed_keys:
                    problems.append(
                        f"authority {authority!r} was not established by the research agents. "
                        "Rely only on the supplied findings."
                    )
        return problems


def _provenance_item(item: ContextItem) -> ContextItem:
    """Provenance stub for an agent that reasons from findings rather than raw chunks.

    `label` stays a provenance note because that is what the trace UI should show. `source_label`
    carries the upstream evidence's real label so the grounding audit has something concrete to
    verify a cited section or case against.
    """
    return ContextItem(
        chunk_id=item.chunk_id,
        kind="upstream_case" if item.kind == "case" else "upstream_evidence",
        label=f"evidence behind upstream findings ({item.chunk_id})",
        source_label=item.evidence_label,
        reason="cited by the Legal Research or Similar Case Agent",
        included=True,
    )


def _authority_key(authority: str) -> str:
    """Normalise an authority reference so equivalent citations compare equal.

    Necessary because agents legitimately format the same authority differently. Observed:
    the Legal Research Agent established ``section 73 | Indian Contract Act, 1872`` and the
    drafting agent wrote ``Section 73 of Indian Contract Act, 1872``. A naive substring
    comparison flagged that as a fabricated authority and burned a whole refinement iteration
    on a false positive - worse than useless, because it teaches the model to drop a citation
    that was in fact correct.

    A section number plus a distinguishing act word is enough to identify a provision; for
    case law, the citation or the leading words of the title.
    """
    text = authority.lower()
    section = re.search(r"(?:section|s\.)\s*(\d+[a-z]{0,2})", text)
    if section:
        # Anchor on the section number plus the first meaningful act word.
        act_words = [
            w
            for w in re.findall(r"[a-z]{4,}", text)
            if w not in {"section", "of", "the", "and", "act"}
        ]
        anchor = act_words[0] if act_words else ""
        return f"s{section.group(1)}:{anchor}"

    citation = re.search(r"\[?\d{4}\]?\s*\d*\s*s\.?c\.?r\.?\s*\d+", text)
    if citation:
        return re.sub(r"[^a-z0-9]", "", citation.group(0))

    return re.sub(r"[^a-z0-9]", "", text)[:28]


def _authorities(legal: dict[str, Any], cases: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for section in legal.get("sections") or []:
        if isinstance(section, dict):
            label = " ".join(
                str(section.get(k) or "") for k in ("section", "act")
            ).strip()
            if label:
                out.append(label)
    for case in cases.get("cases") or []:
        if isinstance(case, dict):
            if case.get("case_title"):
                out.append(str(case["case_title"]))
            if case.get("citation"):
                out.append(str(case["citation"]))
    return out


def _render_facts(facts: dict[str, Any]) -> str:
    if not facts:
        return "(no documents were uploaded, so no facts have been verified)"
    lines: list[str] = []
    if facts.get("factual_summary"):
        lines.append(f"Summary: {facts['factual_summary']}")
    for party in facts.get("parties") or []:
        lines.append(f"Party: {party.get('name')} ({party.get('role')})")
    for date in facts.get("key_dates") or []:
        lines.append(f"Date: {date.get('date')} - {date.get('event')}")
    for amount in facts.get("amounts") or []:
        lines.append(f"Amount: {amount.get('amount')} - {amount.get('what_for')}")
    for obligation in facts.get("obligations") or []:
        lines.append(f"Obligation: {obligation.get('who')} - {obligation.get('what')}")
    return "\n".join(lines) or "(no facts extracted)"


def _render_legal(findings: dict[str, Any]) -> str:
    if not findings:
        return "(legal research produced no findings)"
    lines: list[str] = []
    for issue in findings.get("legal_issues") or []:
        lines.append(f"Issue: {issue}")
    for section in findings.get("sections") or []:
        if isinstance(section, dict):
            lines.append(
                f"{section.get('section')} of {section.get('act')}: "
                f"{section.get('what_it_provides')} (relevance: {section.get('relevance')})"
            )
    for note in findings.get("uncertainties") or []:
        lines.append(f"Uncertainty: {note}")
    return "\n".join(lines) or json.dumps(findings)[:1500]


def _render_cases(findings: dict[str, Any]) -> str:
    cases = findings.get("cases") or []
    if not cases:
        note = findings.get("note") or "no relevant precedents were identified"
        return f"(no precedents to rely on: {note})"
    lines: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        lines.append(
            f"{case.get('case_title')} {case.get('citation')} "
            f"[{case.get('strength', 'unknown')} authority] - "
            f"{case.get('relevant_principle')} (outcome: {case.get('outcome')})"
        )
        for difference in case.get("distinguishing_factors") or []:
            lines.append(f"    distinguishable: {difference}")
    return "\n".join(lines)
