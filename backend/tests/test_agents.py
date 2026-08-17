"""Tests for agent context isolation, grounding audit and authority matching."""

from __future__ import annotations

from legalaid.agents.base import AgentContext, parse_json_response, trim_for_prompt
from legalaid.agents.drafting import DraftingAgent, _authority_key
from legalaid.agents.legal_research import LegalResearchAgent
from legalaid.agents.similar_cases import SimilarCaseAgent
from legalaid.agents.verification import audit
from legalaid.domain.contracts import RetrievedChunk
from legalaid.domain.enums import AgentKey, Collection, DraftType


def chunk(chunk_id: str, collection: Collection, *, title: str = "Act", section: str | None = None):
    return RetrievedChunk(
        chunk_id=chunk_id,
        collection=collection,
        rank=1,
        similarity=0.8,
        source_title=title,
        section=section,
        text=f"text of {chunk_id}",
        source_type="statute",
        token_count=50,
    )


class TestContextIsolation:
    def test_legal_research_excludes_case_law(self) -> None:
        """The Legal Research Agent must not see precedent; that is the Similar Case Agent's job."""
        chunks = [
            chunk("law1", Collection.LEGAL_ACTS, section="Section 73"),
            chunk("case1", Collection.CASE_LAW, title="A v. B"),
        ]
        context = LegalResearchAgent().build_context(question="q", chunks=chunks)
        assert "law1" in context.allowed_chunk_ids
        assert "case1" not in context.allowed_chunk_ids
        assert any(i.chunk_id == "case1" for i in context.excluded)

    def test_similar_cases_excludes_matter_documents(self) -> None:
        chunks = [
            chunk("case1", Collection.CASE_LAW, title="A v. B"),
            chunk("doc1", Collection.USER_DOCUMENTS, title="notice.pdf"),
        ]
        context = SimilarCaseAgent().build_context(question="q", chunks=chunks)
        assert "case1" in context.allowed_chunk_ids
        assert "doc1" not in context.allowed_chunk_ids

    def test_one_chunk_per_judgment(self) -> None:
        """Six fragments of one judgment would crowd out five other precedents."""
        chunks = [
            RetrievedChunk(
                chunk_id=f"c{i}",
                collection=Collection.CASE_LAW,
                rank=i,
                similarity=0.9 - i * 0.01,
                source_title="Same Case",
                text="t",
                source_type="judgment",
                token_count=10,
                metadata={"case_db_id": "case-1"},
            )
            for i in range(5)
        ]
        context = SimilarCaseAgent().build_context(question="q", chunks=chunks)
        assert len(context.allowed_chunk_ids) == 1
        assert len(context.excluded) == 4

    def test_exclusions_are_recorded_with_reasons(self) -> None:
        """Isolation is only provable if you can show what was withheld, and why."""
        chunks = [chunk("case1", Collection.CASE_LAW)]
        context = LegalResearchAgent().build_context(question="q", chunks=chunks)
        assert context.excluded[0].reason
        manifest = context.manifest()
        assert manifest["excluded_count"] == 1
        assert manifest["included_count"] == 0

    def test_budget_limits_are_enforced(self) -> None:
        chunks = [chunk(f"law{i}", Collection.LEGAL_ACTS) for i in range(12)]
        context = LegalResearchAgent().build_context(
            question="q", chunks=chunks, max_law_chunks=5
        )
        assert len(context.allowed_chunk_ids) == 5
        assert len(context.excluded) == 7


class TestGroundingAudit:
    def _context(self) -> AgentContext:
        context = AgentContext(agent_key=AgentKey.LEGAL_RESEARCH, question="q")
        context.include(
            chunk("law1", Collection.LEGAL_ACTS, title="Indian Contract Act, 1872", section="Section 73"),
            kind="law",
            reason="candidate",
        )
        return context

    def test_valid_citation_passes(self) -> None:
        report = audit(
            agent_name="legal_research",
            output={"sections": [{"section": "Section 73", "chunk_id": "law1"}]},
            context=self._context(),
        )
        assert report.passed

    def test_fabricated_chunk_id_is_caught(self) -> None:
        report = audit(
            agent_name="legal_research",
            output={"sections": [{"section": "Section 73", "chunk_id": "made_up"}]},
            context=self._context(),
        )
        assert not report.passed
        assert "made_up" in report.unsupported_citations
        assert "made_up" in report.guidance()

    def test_section_not_in_context_is_caught(self) -> None:
        report = audit(
            agent_name="legal_research",
            output={"sections": [{"section": "Section 999", "chunk_id": "law1"}]},
            context=self._context(),
        )
        assert not report.passed
        assert report.unsupported_sections

    def test_fabricated_case_title_is_caught(self) -> None:
        context = AgentContext(agent_key=AgentKey.SIMILAR_CASES, question="q")
        context.include(
            chunk("case1", Collection.CASE_LAW, title="Eldeco Housing v. Ashok Vidyarthi"),
            kind="case",
            reason="candidate",
        )
        report = audit(
            agent_name="similar_cases",
            output={"cases": [{"case_title": "Totally Invented v. Nobody", "chunk_id": "case1"}]},
            context=context,
        )
        assert not report.passed
        assert report.unsupported_cases

    def test_nested_chunk_ids_are_found(self) -> None:
        report = audit(
            agent_name="x",
            output={"a": {"b": [{"chunk_id": "ghost"}]}},
            context=self._context(),
        )
        assert "ghost" in report.unsupported_citations


class TestAuthorityMatching:
    def test_equivalent_citations_match(self) -> None:
        """Regression: differing formats caused a false 'fabricated authority' flag.

        The research agent produced "section 73 | Indian Contract Act, 1872" and the drafting
        agent wrote "Section 73 of Indian Contract Act, 1872"; a substring test rejected it and
        wasted a refinement iteration teaching the model to drop a correct citation.
        """
        a = _authority_key("Section 73 Indian Contract Act, 1872")
        b = _authority_key("Section 73 of Indian Contract Act, 1872")
        c = _authority_key("s. 73, Indian Contract Act 1872")
        assert a == b == c

    def test_different_sections_do_not_match(self) -> None:
        assert _authority_key("Section 73 Indian Contract Act") != _authority_key(
            "Section 74 Indian Contract Act"
        )

    def test_drafting_accepts_established_authority(self) -> None:
        agent = DraftingAgent()
        context = agent.build_context(
            question="q",
            verified_facts={},
            legal_findings={
                "sections": [{"section": "Section 73", "act": "Indian Contract Act, 1872"}]
            },
            case_findings={},
            draft_type=DraftType.CASE_SUMMARY,
        )
        problems = agent.validate(
            {
                "draft_markdown": "x" * 60,
                "authorities_relied_on": ["Section 73 of Indian Contract Act, 1872"],
            },
            context,
        )
        assert problems == []

    def test_drafting_rejects_invented_authority(self) -> None:
        agent = DraftingAgent()
        context = agent.build_context(
            question="q",
            verified_facts={},
            legal_findings={
                "sections": [{"section": "Section 73", "act": "Indian Contract Act, 1872"}]
            },
            case_findings={},
        )
        problems = agent.validate(
            {
                "draft_markdown": "x" * 60,
                "authorities_relied_on": ["Section 420 of the Penal Code"],
            },
            context,
        )
        assert problems


class TestPromptTrimming:
    def test_illustrations_are_stripped(self) -> None:
        """Illustrations tripled prompt size and gave the model a list pattern to copy."""
        text = (
            "When a contract has been broken, the party who suffers is entitled to "
            "compensation.\n\nIllustrations\n(a) A contracts to sell 50 maunds of saltpetre."
        )
        trimmed = trim_for_prompt(text, max_chars=2000)
        assert "entitled to" in trimmed
        assert "saltpetre" not in trimmed

    def test_long_text_is_cut_at_a_sentence_boundary(self) -> None:
        text = "First sentence here. " * 60
        trimmed = trim_for_prompt(text, max_chars=200)
        assert len(trimmed) <= 220
        assert trimmed.endswith("[...]")


class TestJsonRecovery:
    def test_plain_json(self) -> None:
        parsed, error = parse_json_response('{"a": 1}')
        assert parsed == {"a": 1} and error is None

    def test_fenced_json(self) -> None:
        parsed, _ = parse_json_response('Here:\n```json\n{"a": 1}\n```\nDone.')
        assert parsed == {"a": 1}

    def test_prose_wrapped_json(self) -> None:
        parsed, _ = parse_json_response('Sure thing! {"a": 1} Hope that helps.')
        assert parsed == {"a": 1}

    def test_trailing_comma_is_repaired(self) -> None:
        parsed, error = parse_json_response('{"a": 1,}')
        assert parsed == {"a": 1}
        assert error is not None

    def test_truncated_output_is_recovered(self) -> None:
        parsed, error = parse_json_response('{"a": [1, 2')
        assert parsed == {"a": [1, 2]}
        assert error and "truncated" in error

    def test_unrecoverable_returns_error(self) -> None:
        parsed, error = parse_json_response("I cannot answer that.")
        assert parsed is None and error
