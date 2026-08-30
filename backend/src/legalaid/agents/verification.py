"""Agent 5 - Verification Agent (critic).

Audits another agent's output against the context that agent was actually given, and drives the
orchestrator's refinement loop.

Two layers, in this order:

1. **Deterministic checks** (`audit`) - set membership of cited chunk_ids, section numbers and
   case titles. These are exact, cheap and cannot themselves hallucinate, so they run first and
   are authoritative.
2. **LLM review** - only for the judgement calls a set operation cannot make: overclaiming,
   conclusions unsupported by the evidence, confident language over thin material.

Doing it the other way round would be a mistake: asking a 7B model to check citation membership
is both slower and less reliable than a set lookup, and its verdict could not be trusted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from legalaid.agents.base import Agent, AgentContext
from legalaid.domain.enums import AgentKey

_SECTION_RE = re.compile(r"\b(?:section|s\.)\s*(\d+[A-Za-z]{0,2})\b", re.I)

# Indian reporter citation formats, e.g. "(2014) 7 SCC 123", "AIR 1973 SC 1461",
# "2023 SCC OnLine SC 1066". Regular enough that a fabricated citation inside prose can be
# detected by shape, then checked for membership in what was actually supplied.
_CITATION_RE = re.compile(
    r"(?:\(\d{4}\)\s*\d+\s*[A-Z]{2,5}\s*\d+"
    r"|AIR\s*\d{4}\s*[A-Z]{2,4}\s*\d+"
    r"|\d{4}\s*SCC\s*OnLine\s*[A-Z]{2,4}\s*\d+)",
    re.I,
)


@dataclass(slots=True)
class AuditReport:
    """Deterministic audit outcome."""

    agent: str
    unsupported_citations: list[str] = field(default_factory=list)
    unsupported_sections: list[str] = field(default_factory=list)
    unsupported_cases: list[str] = field(default_factory=list)
    structural_problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (
            self.unsupported_citations
            or self.unsupported_sections
            or self.unsupported_cases
            or self.structural_problems
        )

    @property
    def problem_count(self) -> int:
        return (
            len(self.unsupported_citations)
            + len(self.unsupported_sections)
            + len(self.unsupported_cases)
            + len(self.structural_problems)
        )

    def guidance(self) -> str:
        """Corrective instruction fed back into the failing agent's retry."""
        parts: list[str] = []
        if self.unsupported_citations:
            parts.append(
                "These chunk_ids were not supplied to you and must not be cited: "
                + ", ".join(self.unsupported_citations[:6])
                + ". Use only chunk_ids from the context given to you."
            )
        if self.unsupported_sections:
            parts.append(
                "These section numbers do not appear in your context: "
                + ", ".join(self.unsupported_sections[:6])
                + ". Remove them or replace them with sections that do appear."
            )
        if self.unsupported_cases:
            parts.append(
                "These case titles were not supplied: "
                + ", ".join(self.unsupported_cases[:4])
                + ". Never invent a precedent; copy titles exactly from CASE CONTEXT."
            )
        if self.structural_problems:
            parts.append("Structural problems: " + "; ".join(self.structural_problems[:5]))
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "passed": self.passed,
            "unsupported_citations": self.unsupported_citations,
            "unsupported_sections": self.unsupported_sections,
            "unsupported_cases": self.unsupported_cases,
            "structural_problems": self.structural_problems,
            "problem_count": self.problem_count,
        }


def audit(
    *,
    agent_name: str,
    output: dict[str, Any],
    context: AgentContext,
    structural_problems: list[str] | None = None,
    prose_fields: tuple[str, ...] = (),
) -> AuditReport:
    """Deterministic grounding audit. No model involved, so it cannot hallucinate.

    `prose_fields` names output keys whose free text must also be scanned. Structured
    citation fields are not the whole story: the lawyer reads `answer_markdown`, so a section
    number invented inside the prose is exactly as damaging as one invented in a citation
    list, and until this scan existed it was completely unchecked.
    """
    report = AuditReport(agent=agent_name, structural_problems=list(structural_problems or []))
    allowed_ids = set(context.allowed_chunk_ids)

    # Text of everything the agent was actually shown. `evidence_label` is used rather than
    # `label` because agents that see no raw chunks (drafting, synthesis) still carry the
    # upstream label here - without it the checks below have nothing to compare against and
    # would silently pass anything.
    supplied_text = " ".join(i.evidence_label for i in context.included).lower()
    supplied_sections = {
        m.group(1).upper() for m in _SECTION_RE.finditer(supplied_text)
    }
    supplied_titles = [
        i.evidence_label.lower()
        for i in context.included
        if i.kind in {"case", "upstream_case"}
    ]

    for chunk_id in _collect_chunk_ids(output):
        if chunk_id not in allowed_ids:
            report.unsupported_citations.append(chunk_id)

    for section in output.get("sections") or []:
        if not isinstance(section, dict):
            continue
        raw = str(section.get("section") or "")
        match = _SECTION_RE.search(raw) or re.search(r"(\d+[A-Za-z]{0,2})", raw)
        if match and supplied_sections and match.group(1).upper() not in supplied_sections:
            report.unsupported_sections.append(raw)

    for case in output.get("cases") or []:
        if not isinstance(case, dict):
            continue
        title = str(case.get("case_title") or "").lower().strip()
        if not title:
            continue
        if supplied_titles and not any(
            title[:24] in s or s[:24] in title for s in supplied_titles
        ):
            report.unsupported_cases.append(str(case.get("case_title")))

    # ------------------------------------------------------------------ prose scan
    for field_name in prose_fields:
        prose = output.get(field_name)
        if not isinstance(prose, str) or not prose.strip():
            continue
        if supplied_sections:
            for match in _SECTION_RE.finditer(prose):
                number = match.group(1).upper()
                if number not in supplied_sections:
                    label = f"Section {match.group(1)} (in {field_name})"
                    if label not in report.unsupported_sections:
                        report.unsupported_sections.append(label)
        # Indian reporter citations are highly regular, so a fabricated one is detectable
        # without needing to understand the surrounding sentence.
        if supplied_text:
            for match in _CITATION_RE.finditer(prose):
                citation = match.group(0).strip()
                if citation.lower() not in supplied_text:
                    label = f"{citation} (in {field_name})"
                    if label not in report.unsupported_cases:
                        report.unsupported_cases.append(label)

    return report


def _collect_chunk_ids(payload: Any) -> list[str]:
    """Every chunk_id mentioned anywhere in a nested output structure."""
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "chunk_id" and isinstance(value, str) and value:
                found.append(value)
            else:
                found.extend(_collect_chunk_ids(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_collect_chunk_ids(item))
    return found


class VerificationAgent(Agent):
    """LLM reviewer for the judgement calls deterministic checks cannot make."""

    agent_key = AgentKey.VERIFICATION
    prompt_name = "verification"

    def build_context(
        self,
        *,
        question: str,
        reviewed_agent: str,
        reviewed_output: dict[str, Any],
        reviewed_context: AgentContext,
    ) -> AgentContext:
        context = AgentContext(agent_key=self.agent_key, question=question)
        context.extra["reviewed_agent"] = reviewed_agent
        context.extra["reviewed_output"] = reviewed_output
        context.extra["allowed_chunk_ids"] = reviewed_context.allowed_chunk_ids
        context.extra["context_digest"] = "\n".join(
            f"- {i.chunk_id}: {i.label}" for i in reviewed_context.included
        ) or "(nothing was supplied)"
        # The critic inherits the reviewed agent's evidence set as its own frame of reference.
        for item in reviewed_context.included:
            context.included.append(item)
        return context

    def render(self, context: AgentContext) -> dict[str, Any]:
        return {
            "agent_name": context.extra["reviewed_agent"],
            "agent_output": json.dumps(context.extra["reviewed_output"], indent=2)[:6000],
            "allowed_chunk_ids": "\n".join(context.extra["allowed_chunk_ids"]) or "(none)",
            "context_digest": context.extra["context_digest"],
        }
