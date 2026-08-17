"""Tests for subsection packing in the statute chunker.

Regression cover for a real defect: a definitions section split at every subsection marker
produced 28 chunks, a dozen of them under 300 characters. Fragments that small embed poorly
and crowd out genuinely relevant results, so chunks must be packed toward the target size.
"""

from __future__ import annotations

from legalaid.config import get_settings
from legalaid.ingestion.chunking import chunk_statute

# Shaped like Information Technology Act s.2: many short lettered definitions.
DEFINITIONS_SECTION = "2. Definitions\n\n(1) In this Act, unless the context otherwise requires,--\n" + "\n".join(
    f'({chr(97 + i)}) "term{i}" means a defined expression used in this Act for the purposes '
    f"of identifying subject matter number {i} and related concepts;"
    for i in range(24)
)


class TestSubsectionPacking:
    def test_definitions_are_packed_not_fragmented(self) -> None:
        chunks = chunk_statute(text=DEFINITIONS_SECTION, act_name="Information Technology Act, 2000")
        # Before packing this produced ~25+ chunks; packing should collapse it dramatically.
        assert len(chunks) <= 6, f"expected packed chunks, got {len(chunks)}"

    def test_no_tiny_chunks_when_more_than_one_chunk(self) -> None:
        settings = get_settings()
        chunks = chunk_statute(text=DEFINITIONS_SECTION, act_name="Information Technology Act, 2000")
        if len(chunks) > 1:
            # Allow the final chunk to be a short remainder, but nothing else.
            for chunk in chunks[:-1]:
                assert chunk.char_count >= settings.chunk_min_chars * 2, (
                    f"chunk of {chunk.char_count} chars is too small to retrieve well"
                )

    def test_chunks_respect_the_max_size(self) -> None:
        settings = get_settings()
        chunks = chunk_statute(text=DEFINITIONS_SECTION, act_name="Information Technology Act, 2000")
        for chunk in chunks:
            assert chunk.char_count <= settings.chunk_max_chars

    def test_no_clause_text_is_lost(self) -> None:
        chunks = chunk_statute(text=DEFINITIONS_SECTION, act_name="Information Technology Act, 2000")
        combined = "\n".join(c.text for c in chunks)
        for i in range(24):
            assert f'"term{i}"' in combined, f"definition {i} was dropped"

    def test_chunks_do_not_begin_mid_clause(self) -> None:
        """Packing must still split on subsection boundaries."""
        chunks = chunk_statute(text=DEFINITIONS_SECTION, act_name="Information Technology Act, 2000")
        for chunk in chunks:
            # Strip the injected "Act - Section" header line to see the body's first line.
            body = chunk.text.split("\n\n", 1)[-1].strip()
            first = body.split("\n", 1)[0]
            assert first.startswith("(") or first.startswith("In this Act"), (
                f"chunk starts mid-clause: {first[:60]!r}"
            )

    def test_short_section_stays_whole(self) -> None:
        short = (
            "4. Legal recognition of electronic records\n\n"
            "Where any law provides that information shall be in writing, such requirement "
            "shall be deemed satisfied if it is rendered in an electronic form."
        )
        chunks = chunk_statute(text=short, act_name="Information Technology Act, 2000")
        assert len(chunks) == 1

    def test_provenance_survives_packing(self) -> None:
        chunks = chunk_statute(
            text=DEFINITIONS_SECTION,
            act_name="Information Technology Act, 2000",
            source_url="https://www.indiacode.nic.in/SectionPageContent?actid=X&sectionID=1",
            act_year=2000,
            extra_metadata={"citation": "Section 2, Information Technology Act, 2000"},
        )
        for chunk in chunks:
            assert chunk.source_url is not None
            assert chunk.metadata["act_year"] == 2000
            assert chunk.metadata["citation"] == "Section 2, Information Technology Act, 2000"
            assert "Information Technology Act, 2000" in chunk.text
