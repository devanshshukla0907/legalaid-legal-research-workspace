"""Tests for hybrid retrieval: BM25 tokenisation, citation boosting and rank fusion."""

from __future__ import annotations

from legalaid.services.retrieval.fusion import reciprocal_rank_fusion
from legalaid.services.retrieval.keyword import is_citation_token, tokenize


class TestTokenisation:
    def test_section_reference_becomes_one_token(self) -> None:
        """Regression: an underscore-excluding token pattern split section_73 back apart.

        That mattered because "section" appears in ~71% of the corpus and "73" in ~1.4%, so
        splitting destroyed the precision the citation carried.
        """
        assert "section_73" in tokenize("What does Section 73 deal with?")
        assert "section" not in tokenize("Section 73")

    def test_citation_abbreviations_normalise_together(self) -> None:
        for variant in ("Section 73", "section 73", "S. 73", "sec 73", "Sections 73"):
            assert "section_73" in tokenize(variant), variant

    def test_order_rule_citation_is_combined_and_split(self) -> None:
        tokens = tokenize("Order VII Rule 11 rejection of plaint")
        assert "order_vii_rule_11" in tokens  # the precise combined citation
        assert "order_vii" in tokens
        assert "rule_11" in tokens

    def test_article_reference(self) -> None:
        assert "article_14" in tokenize("Article 14 of the Constitution")

    def test_alphanumeric_sections(self) -> None:
        assert "section_10a" in tokenize("Section 10A of the IT Act")

    def test_stopwords_and_short_tokens_removed(self) -> None:
        tokens = tokenize("the party of a contract is in breach")
        assert "the" not in tokens and "of" not in tokens and "is" not in tokens
        assert "party" in tokens and "contract" in tokens and "breach" in tokens

    def test_citation_token_detection(self) -> None:
        assert is_citation_token("section_73")
        assert is_citation_token("order_vii_rule_11")
        assert not is_citation_token("compensation")


class TestReciprocalRankFusion:
    def test_agreement_between_retrievers_wins(self) -> None:
        """A chunk both retrievers rank should beat one only a single retriever ranks."""
        dense = [("a", 0.90), ("b", 0.88)]
        keyword = [("b", 12.0, ["section_73"])]
        fused = reciprocal_rank_fusion(dense, keyword)
        assert fused[0].chunk_id == "b"
        assert fused[0].agreed is True

    def test_dense_only_hits_are_still_returned(self) -> None:
        fused = reciprocal_rank_fusion([("a", 0.9)], [])
        assert [h.chunk_id for h in fused] == ["a"]
        assert fused[0].retrievers == ["dense"]

    def test_keyword_only_hits_are_still_returned(self) -> None:
        fused = reciprocal_rank_fusion([], [("z", 8.0, ["section_74"])])
        assert [h.chunk_id for h in fused] == ["z"]
        assert fused[0].retrievers == ["keyword"]

    def test_scores_do_not_need_calibration(self) -> None:
        """BM25 scores are unbounded; fusion must not be skewed by their magnitude."""
        small = reciprocal_rank_fusion([("a", 0.9)], [("b", 2.0, [])])
        huge = reciprocal_rank_fusion([("a", 0.9)], [("b", 9999.0, [])])
        assert [h.chunk_id for h in small] == [h.chunk_id for h in huge]

    def test_weights_are_respected(self) -> None:
        dense = [("d", 0.9)]
        keyword = [("k", 5.0, [])]
        dense_first = reciprocal_rank_fusion(dense, keyword, dense_weight=1.0, keyword_weight=0.1)
        keyword_first = reciprocal_rank_fusion(dense, keyword, dense_weight=0.1, keyword_weight=1.0)
        assert dense_first[0].chunk_id == "d"
        assert keyword_first[0].chunk_id == "k"

    def test_limit_is_applied(self) -> None:
        dense = [(f"c{i}", 0.9 - i * 0.01) for i in range(10)]
        assert len(reciprocal_rank_fusion(dense, [], limit=3)) == 3

    def test_empty_inputs(self) -> None:
        assert reciprocal_rank_fusion([], []) == []
