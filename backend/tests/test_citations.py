"""Citation extraction and treatment classification.

The treatment tests carry the weight here. Extracting a citation is mechanical; deciding whether
a judgment *followed* or *overruled* the authority it cited is what determines whether the graph
can be trusted, and getting it backwards would let the system present overruled authority as
support.
"""

from __future__ import annotations

import pytest

from legalaid.ingestion.citations import (
    Treatment,
    build_edges,
    classify_treatment,
    extract_citations,
    extract_section_references,
    normalise_citation,
)


class TestExtraction:
    @pytest.mark.parametrize(
        ("text", "expected_kind"),
        [
            ("as held in (2014) 7 SCC 123 the position is settled", "scc_bracketed"),
            ("see AIR 1973 SC 1461 for the proposition", "air"),
            ("reported as 2023 SCC OnLine SC 1066", "scc_online"),
            ("reported at (2019) 4 SCR 88", "scr"),
            ("see 2019 (3) SCALE 45", "scale"),
            ("in Civil Appeal No. 1234 of 2019 this Court held", "case_number"),
        ],
    )
    def test_recognises_each_reporter_format(self, text: str, expected_kind: str) -> None:
        citations = extract_citations(text)
        assert citations, f"nothing extracted from {text!r}"
        assert citations[0].kind == expected_kind

    def test_ignores_ordinary_numbers(self) -> None:
        text = "The plaintiff paid Rs 18,40,000 on 12 March 2019 under clause 7 of the agreement."
        assert extract_citations(text) == []

    def test_deduplicates_by_canonical_form(self) -> None:
        text = "(2014) 7 SCC 123 ... later, again (2014) 7 S.C.C. 123 ... and (2014) 7 SCC 123"
        assert len(extract_citations(text)) == 1

    def test_dotted_and_undotted_forms_normalise_together(self) -> None:
        a = extract_citations("(2014) 7 SCC 123")[0]
        b = extract_citations("(2014) 7 S.C.C. 123")[0]
        assert a.normalised == b.normalised

    def test_captures_year_and_position(self) -> None:
        citation = extract_citations("prefix text (2014) 7 SCC 123")[0]
        assert citation.year == 2014
        assert citation.position > 0

    def test_multiple_distinct_citations(self) -> None:
        text = "compare (2014) 7 SCC 123 with AIR 1973 SC 1461 and 2023 SCC OnLine SC 1066"
        assert len(extract_citations(text)) == 3


class TestTreatment:
    @pytest.mark.parametrize(
        ("context", "expected"),
        [
            ("We overrule the decision in", Treatment.OVERRULED),
            ("the judgment is hereby reversed in", Treatment.REVERSED),
            ("we affirm the view taken in", Treatment.AFFIRMED),
            ("that decision is distinguishable from", Treatment.DISTINGUISHED),
            ("we doubt the correctness of", Treatment.DOUBTED),
            ("we follow the ratio in", Treatment.FOLLOWED),
            ("learned counsel relied upon", Treatment.RELIED_ON),
            ("referred to the decision in", Treatment.REFERRED),
        ],
    )
    def test_classifies_treatment_language(self, context: str, expected: Treatment) -> None:
        text = f"{context} (2014) 7 SCC 123 and proceeded."
        citation = extract_citations(text)[0]
        treatment, _evidence = classify_treatment(text, citation.position)
        assert treatment is expected

    def test_defaults_to_cited_without_a_cue(self) -> None:
        text = "The provision was considered in (2014) 7 SCC 123."
        citation = extract_citations(text)[0]
        treatment, _evidence = classify_treatment(text, citation.position)
        assert treatment is Treatment.CITED

    def test_strongest_cue_wins(self) -> None:
        """'We overrule X, following Y' must not mark X as merely followed."""
        text = "We overrule the reasoning, while following other authority, in (2014) 7 SCC 123."
        citation = extract_citations(text)[0]
        treatment, _evidence = classify_treatment(text, citation.position)
        assert treatment is Treatment.OVERRULED

    def test_evidence_is_returned(self) -> None:
        text = "We follow the ratio in (2014) 7 SCC 123 without hesitation."
        citation = extract_citations(text)[0]
        _treatment, evidence = classify_treatment(text, citation.position)
        assert "follow" in evidence.lower()


class TestSectionReferences:
    def test_extracts_single_and_plural_forms(self) -> None:
        assert extract_section_references("under Section 73 of the Act") == ["73"]
        assert extract_section_references("see s. 10A thereof") == ["10A"]

    def test_expands_lists(self) -> None:
        assert extract_section_references("sections 73, 74 and 75") == ["73", "74", "75"]

    def test_expands_ranges(self) -> None:
        assert extract_section_references("ss. 73 to 75") == ["73", "74", "75"]

    def test_ignores_absurd_ranges(self) -> None:
        """A 'range' spanning hundreds of sections is a misparse, not a citation."""
        assert extract_section_references("sections 1 to 500") == []

    def test_deduplicates_preserving_order(self) -> None:
        assert extract_section_references("s. 74, then s. 73, then s. 74 again") == ["74", "73"]


class TestGraphEdges:
    def test_resolves_a_citation_to_a_case_in_the_corpus(self) -> None:
        cases = [
            ("case_a", "We follow the ratio in (2014) 7 SCC 123.", ["AIR 2020 SC 1"]),
            ("case_b", "Some unrelated reasoning.", ["(2014) 7 SCC 123"]),
        ]
        edges, stats = build_edges(cases)
        resolved = [e for e in edges if e.resolved]
        assert len(resolved) == 1
        assert resolved[0].source_case_id == "case_a"
        assert resolved[0].target_case_id == "case_b"
        assert resolved[0].treatment is Treatment.FOLLOWED
        assert stats["resolved"] == 1

    def test_unresolved_citations_are_reported_not_dropped(self) -> None:
        """A citation pointing outside a small corpus is normal and must stay visible."""
        cases = [("case_a", "As held in (1999) 2 SCC 500, the rule applies.", [])]
        edges, stats = build_edges(cases)
        assert len(edges) == 1
        assert edges[0].resolved is False
        assert edges[0].target_case_id is None
        assert stats["unresolved"] == 1

    def test_self_citation_is_skipped(self) -> None:
        cases = [("case_a", "This judgment, reported as (2014) 7 SCC 123, holds.", ["(2014) 7 SCC 123"])]
        edges, _stats = build_edges(cases)
        assert edges == []

    def test_treatment_counts_are_reported(self) -> None:
        cases = [
            ("a", "We overrule (2014) 7 SCC 123.", []),
            ("b", "We follow (2015) 1 SCC 1.", []),
        ]
        _edges, stats = build_edges(cases)
        assert stats["treatment_overruled"] == 1
        assert stats["treatment_followed"] == 1


def test_normalise_citation_is_stable_across_spellings() -> None:
    import re

    from legalaid.ingestion.citations import _PATTERNS

    pattern = dict(_PATTERNS)["air"]
    a = normalise_citation("air", pattern.search("AIR 1973 SC 1461"))
    b = normalise_citation("air", pattern.search("A.I.R. 1973 S.C. 1461"))
    assert a == b
    assert isinstance(re.compile(""), re.Pattern)
