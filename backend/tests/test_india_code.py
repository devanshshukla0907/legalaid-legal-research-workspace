"""Tests for the India Code fetcher's parsing and text cleaning.

The HTML fixtures below are trimmed copies of real responses from indiacode.nic.in, so these
tests pin the behaviour that actually matters: statutory text must survive ingestion intact,
because every citation in the application is derived from it.
"""

from __future__ import annotations

import pytest

from legalaid.ingestion.india_code import (
    parse_act_title,
    parse_pdf_url,
    parse_section_index,
    strip_html,
)

# Real markup shape from GET /SectionPageContent (Indian Contract Act s.73).
SECTION_73_CONTENT = (
    '<span style="margin-left:15px;"></span>When a contract has been\r\n'
    "broken, the party who suffers by such breach is entitled to receive, from the party "
    "who has broken the\r\ncontract, compensation for any loss or damage caused to him "
    "thereby.</br><hr class=\"hr1\"/>\r\n"
    '<span style="margin-left:15px;"></span>Such compensation is not to be given for any '
    "remote and indirect loss.</br><hr class=\"hr1\"/>\r\n"
    '<span style="margin-left:15px;"></span><i>Explanation</i>.In estimating the loss or '
    "damage arising from a breach of contract, the means which\r\nexisted of remedying the "
    "inconvenience must be taken into\r\naccount.</br><hr class=\"hr1\"/><hr class=\"hr1\"/>\r\n"
    "<i>Illustrations</i></br><hr class=\"hr1\"/>\r\n"
    '<span style="margin-left:15px;"></span>(a) A contracts to sell and deliver 50 maunds '
    "of saltpetre to B.</br>"
)

ACT_PAGE = """
<html><head><title>India Code: Indian Contract Act, 1872</title></head><body>
<a href="/bitstream/123456789/2187/3/H1872-09.pdf">Hindi</a>
<a href="/bitstream/123456789/2187/2/A187209.pdf">English</a>
<tr><td><div id="accordion74" class="hideshowsection panel-default">
<a id="AC_X#38677#AC_X" class="title" target="_blank"
 href=/show-data?abv=CEN&statehandle=123456789/1362&actid=AC_CEN_3_20_00035_187209_1523268996428&sectionId=38677&sectionno=73&orderno=74&orgactid=AC_CEN_3_20_00035_187209_1523268996428>
<span style="font-size: 14px;" class="label label-info"> Section 73.</span>&nbsp;Compensation for loss or damage caused by breach of contract.</a>
</div></td></tr>
<tr><td><div id="accordion75" class="hideshowsection panel-default">
<a id="AC_X#38678#AC_X" class="title" target="_blank"
 href=/show-data?abv=CEN&statehandle=123456789/1362&actid=AC_CEN_3_20_00035_187209_1523268996428&sectionId=38678&sectionno=74&orderno=75&orgactid=AC_CEN_3_20_00035_187209_1523268996428>
<span style="font-size: 14px;" class="label label-info"> Section 74.</span>&nbsp;Compensation for breach of contract where penalty stipulated for.</a>
</div></td></tr>
</body></html>
"""


class TestStripHtml:
    def test_pdf_line_wraps_are_rejoined_not_treated_as_breaks(self) -> None:
        """A newline mid-sentence is a PDF wrap artifact and must not fragment the sentence."""
        text = strip_html(SECTION_73_CONTENT)
        assert "has been broken" in text
        assert "has been\nbroken" not in text
        assert "the means which existed of remedying" in text

    def test_inline_tags_do_not_weld_words_together(self) -> None:
        text = strip_html(SECTION_73_CONTENT)
        assert "Explanation. In estimating" in text
        assert "Explanation.In" not in text

    def test_paragraph_breaks_are_preserved(self) -> None:
        text = strip_html(SECTION_73_CONTENT)
        assert "\n\n" in text
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        assert len(paragraphs) >= 4  # main rule, proviso, explanation, illustrations

    def test_illustrations_survive_with_their_labels(self) -> None:
        text = strip_html(SECTION_73_CONTENT)
        assert "Illustrations" in text
        assert "(a) A contracts to sell" in text

    def test_no_html_or_carriage_returns_remain(self) -> None:
        text = strip_html(SECTION_73_CONTENT)
        assert "<" not in text and ">" not in text
        assert "\r" not in text

    def test_entities_are_unescaped(self) -> None:
        assert strip_html("A &amp; B&nbsp;Ltd.") == "A & B Ltd."

    def test_empty_input(self) -> None:
        assert strip_html("") == ""


class TestParseSectionIndex:
    def test_extracts_sections_with_numbers_and_headings(self) -> None:
        refs = parse_section_index(ACT_PAGE)
        assert [r.section_number for r in refs] == ["73", "74"]
        assert refs[0].heading == "Compensation for loss or damage caused by breach of contract"
        assert refs[0].section_id == "38677"
        assert refs[0].order_no == 74

    def test_carries_actid_needed_for_the_content_endpoint(self) -> None:
        ref = parse_section_index(ACT_PAGE)[0]
        assert ref.actid == "AC_CEN_3_20_00035_187209_1523268996428"
        assert "SectionPageContent" in ref.content_url
        assert "sectionID=38677" in ref.content_url

    def test_results_are_ordered_by_order_no(self) -> None:
        refs = parse_section_index(ACT_PAGE)
        assert [r.order_no for r in refs] == sorted(r.order_no for r in refs)

    def test_empty_page_yields_nothing(self) -> None:
        assert parse_section_index("<html></html>") == []


class TestParseMetadata:
    def test_act_title(self) -> None:
        assert parse_act_title(ACT_PAGE) == "Indian Contract Act, 1872"

    def test_prefers_english_pdf_over_hindi(self) -> None:
        url = parse_pdf_url(ACT_PAGE)
        assert url is not None
        assert url.endswith("/A187209.pdf")

    def test_returns_none_when_no_pdf(self) -> None:
        assert parse_pdf_url("<html>no links</html>") is None


class TestCorpusManifest:
    def test_manifest_covers_the_practitioner_core(self) -> None:
        from legalaid.ingestion.corpus import all_acts, estimated_provisions

        acts = all_acts()
        assert len(acts) >= 15
        # Expanded from the original 5-act MVP. The upper bound is a guard against someone
        # adding a very large code unfiltered and quietly tripling embedding time.
        assert 1200 <= estimated_provisions() <= 6000

    def test_every_act_is_resolvable(self) -> None:
        """Each act needs a pinned id or a search title, or ingestion cannot find it."""
        from legalaid.ingestion.corpus import all_acts

        for spec in all_acts():
            assert spec.act_id or spec.search_title, f"{spec.key} has no way to be resolved"
            assert spec.resolve_title, f"{spec.key} has no resolvable title"

    def test_act_keys_are_unique(self) -> None:
        from legalaid.ingestion.corpus import all_acts

        keys = [spec.key for spec in all_acts()]
        assert len(keys) == len(set(keys))

    def test_section_filter_gating(self) -> None:
        from legalaid.ingestion.corpus import get_act, wants_section

        contract = get_act("contract_act_1872")
        assert wants_section(contract, "73")  # no filter -> everything in scope

        # BNSS is the one act deliberately reduced, because 531 sections of criminal
        # procedure are mostly outside what a commercial matter reaches.
        bnss = get_act("bnss_2023")
        assert wants_section(bnss, "173")
        assert not wants_section(bnss, "999")

    def test_unknown_act_key_raises(self) -> None:
        from legalaid.ingestion.corpus import get_act

        with pytest.raises(ValueError, match="Unknown act"):
            get_act("does_not_exist")
