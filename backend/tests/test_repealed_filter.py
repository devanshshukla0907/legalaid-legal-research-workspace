"""Repealed provisions must never enter the knowledge base.

Regression cover for an observed retrieval failure: asking about "Section 73" ranked
"Section 259 - Repealed ... Rep. by s. 73 and the Second Schedule" third, because the repeal
note mentions s.73. A legal research tool that cites a dead provision as applicable law is
worse than one that returns nothing.
"""

from __future__ import annotations

import pytest

from legalaid.ingestion.india_code import FetchedSection
from legalaid.services.knowledge.ingest import is_repealed


def make(heading: str, text: str) -> FetchedSection:
    return FetchedSection(
        section_number="259",
        heading=heading,
        text=text,
        footnotes="",
        order_no=1,
        section_id="1",
        source_url="https://www.indiacode.nic.in/SectionPageContent?actid=X&sectionID=1",
    )


class TestRepealedDetection:
    @pytest.mark.parametrize(
        "heading",
        ["Repealed", "repealed", "Repealed.", "Omitted", "[Omitted.]", "Rep. by Act 4 of 1937"],
    )
    def test_repealed_headings_are_detected(self, heading: str) -> None:
        assert is_repealed(make(heading, "[Some old provision.] Rep. by s. 73, ibid."))

    def test_the_real_world_case(self) -> None:
        section = make(
            "Repealed",
            "[Obligations, to firm, of partner carrying on business.] Rep. by s. 73 and the "
            "Second Schedule, ibid.",
        )
        assert is_repealed(section)

    def test_short_body_that_is_only_a_repeal_note(self) -> None:
        assert is_repealed(make("Seller not bound to deliver", "Rep. by s. 65, ibid."))

    def test_live_provisions_are_kept(self) -> None:
        section = make(
            "Compensation for loss or damage caused by breach of contract",
            "When a contract has been broken, the party who suffers by such breach is entitled "
            "to receive, from the party who has broken the contract, compensation for any loss "
            "or damage caused to him thereby.",
        )
        assert not is_repealed(section)

    def test_live_provision_mentioning_repeal_is_kept(self) -> None:
        """A long, substantive section that merely discusses repeals must survive."""
        section = make(
            "Repeal and savings",
            "The enactments mentioned in the Schedule are hereby repealed to the extent "
            "specified. Notwithstanding such repeal, anything done or any action taken under "
            "the repealed provisions shall be deemed to have been done or taken under the "
            "corresponding provisions of this Act, and shall continue in force accordingly "
            "until superseded by anything done under this Act. " * 2,
        )
        assert not is_repealed(section)
