"""Tests for the legal-document-aware chunkers."""

from __future__ import annotations

from legalaid.domain.enums import ChunkStrategy, Collection
from legalaid.ingestion.chunking import (
    chunk_judgment,
    chunk_statute,
    chunk_uploaded_document,
    clean_legal_text,
    detect_headings,
    parse_statute_structure,
    segment_judgment_text,
    split_oversized,
)

ACT_TEXT = """
THE INDIAN CONTRACT ACT, 1872

CHAPTER VI - OF THE CONSEQUENCES OF BREACH OF CONTRACT

73. Compensation for loss or damage caused by breach of contract.
When a contract has been broken, the party who suffers by such breach is entitled to receive,
from the party who has broken the contract, compensation for any loss or damage caused to him
thereby, which naturally arose in the usual course of things from such breach.

Such compensation is not to be given for any remote and indirect loss or damage sustained by
reason of the breach.

74. Compensation for breach of contract where penalty stipulated for.
When a contract has been broken, if a sum is named in the contract as the amount to be paid in
case of such breach, the party complaining of the breach is entitled to receive from the party
who has broken the contract reasonable compensation not exceeding the amount so named.

75. Party rightfully rescinding contract, entitled to compensation.
A person who rightfully rescinds a contract is entitled to compensation for any damage which he
has sustained through the non-fulfilment of the contract.
"""


def test_parse_statute_structure_finds_chapter_and_sections() -> None:
    sections = parse_statute_structure(ACT_TEXT)
    numbers = [s.section_number for s in sections if s.section_number]
    assert numbers == ["73", "74", "75"]
    section73 = next(s for s in sections if s.section_number == "73")
    assert section73.chapter is not None
    assert "Chapter VI" in section73.chapter
    assert "Compensation for loss" in (section73.heading or "")
    assert "naturally arose" in section73.text


def test_parse_statute_keeps_preamble() -> None:
    sections = parse_statute_structure(ACT_TEXT)
    assert any(s.section_number is None for s in sections)


def test_chunk_statute_preserves_provenance() -> None:
    chunks = chunk_statute(
        text=ACT_TEXT,
        act_name="Indian Contract Act, 1872",
        act_year=1872,
        source_url="https://www.indiacode.nic.in/example",
    )
    assert chunks
    assert all(c.collection is Collection.LEGAL_ACTS for c in chunks)
    assert all(c.chunk_strategy is ChunkStrategy.STATUTE_SECTION for c in chunks)
    s73 = [c for c in chunks if c.section == "Section 73"]
    assert s73, "section 73 must produce at least one chunk"
    assert s73[0].source_url == "https://www.indiacode.nic.in/example"
    assert s73[0].metadata["act_name"] == "Indian Contract Act, 1872"
    assert s73[0].metadata["section_number"] == "73"
    # the act name is embedded so a retrieved chunk is self-describing
    assert "Indian Contract Act" in s73[0].text
    assert s73[0].token_count > 0


def test_statute_sections_are_not_cross_contaminated() -> None:
    chunks = chunk_statute(text=ACT_TEXT, act_name="Indian Contract Act, 1872")
    s74 = [c for c in chunks if c.section == "Section 74"]
    assert s74
    joined = " ".join(c.text for c in s74)
    assert "penalty stipulated" in joined or "sum is named" in joined
    assert "rightfully rescinds" not in joined


def test_numbered_list_is_not_mistaken_for_section() -> None:
    text = "5. Interpretation clause.\nThe amount of 1.5 lakh is payable.\n"
    sections = parse_statute_structure(text)
    assert [s.section_number for s in sections if s.section_number] == ["5"]


JUDGMENT = """
IN THE SUPREME COURT OF INDIA
Civil Appeal No. 1234 of 2019

FACTS
The appellant supplied goods to the respondent under a written agreement dated 1 April 2015.
The respondent failed to pay the agreed consideration despite repeated demands.

ISSUES
Whether the respondent is liable to pay compensation under Section 73 of the Contract Act.

SUBMISSIONS
Learned counsel for the appellant submitted that the breach was admitted.

ANALYSIS
We have heard both sides. The measure of damages must follow naturally from the breach.

HELD
The appeal is allowed. The respondent shall pay compensation with interest.
"""


def test_segment_judgment_text_identifies_segments() -> None:
    segments = segment_judgment_text(JUDGMENT)
    assert "facts" in segments
    assert "issues" in segments
    assert "arguments" in segments
    assert "reasoning" in segments
    assert "decision" in segments
    assert "supplied goods" in segments["facts"]


def test_segment_judgment_falls_back_without_headings() -> None:
    segments = segment_judgment_text("A plain order with no recognised headings at all.")
    assert set(segments) == {"judgment"}


def test_chunk_judgment_uses_case_law_collection() -> None:
    segments = segment_judgment_text(JUDGMENT)
    chunks = chunk_judgment(
        metadata_text="Supreme Court of India, Civil Appeal No. 1234 of 2019",
        segments=segments,
        case_title="Appellant v Respondent",
        source_url="https://example.org/judgment.pdf",
    )
    assert chunks
    assert all(c.collection is Collection.CASE_LAW for c in chunks)
    assert all(c.chunk_strategy is ChunkStrategy.JUDGMENT_SECTION for c in chunks)
    assert {c.metadata["segment"] for c in chunks} & {"facts", "issues", "decision"}


def test_chunk_uploaded_document_preserves_page_numbers() -> None:
    pages = [
        (1, "LEGAL NOTICE\n\nYou have failed to pay the agreed amount of Rs 5,00,000.\n\n" + "x" * 50),
        (2, "REPLY\n\nThe claim is denied in its entirety for the reasons stated below.\n\n" + "y" * 50),
    ]
    chunks = chunk_uploaded_document(
        pages=pages, filename="notice.pdf", document_id="doc_1", matter_id="mat_1"
    )
    assert chunks
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.collection is Collection.USER_DOCUMENTS for c in chunks)
    assert all(c.metadata["document_id"] == "doc_1" for c in chunks)
    assert all("Page" in c.text for c in chunks)


def test_uploaded_document_never_silently_drops_a_page() -> None:
    chunks = chunk_uploaded_document(
        pages=[(1, "SHORT")], filename="tiny.pdf", document_id="doc_2"
    )
    assert chunks
    assert chunks[0].page == 1


def test_clean_legal_text_removes_artefacts_but_keeps_structure() -> None:
    raw = "Section 73\n\nPage 4 of 12\n\ncompen-\nsation is  payable\n\n\n\nnext para"
    cleaned = clean_legal_text(raw)
    assert "Page 4 of 12" not in cleaned
    assert "compensation is payable" in cleaned
    assert "\n\n\n" not in cleaned
    assert "Section 73" in cleaned


def test_split_oversized_respects_limit() -> None:
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(12))
    parts = split_oversized(text, max_chars=800, overlap=100)
    assert len(parts) > 1
    assert all(len(p) <= 900 for p in parts)
    assert "Paragraph 0" in parts[0]


def test_split_oversized_returns_single_part_when_small() -> None:
    assert split_oversized("short text", max_chars=500, overlap=50) == ["short text"]


def test_detect_headings() -> None:
    headings = detect_headings("LEGAL NOTICE\nsome body text here\nSTATEMENT OF FACTS\nmore text")
    assert "LEGAL NOTICE" in headings
    assert "STATEMENT OF FACTS" in headings
