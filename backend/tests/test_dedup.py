"""Duplicate detection.

The important assertions are the negative ones: two provisions that discuss the same subject in
different words must NOT be flagged, or the report becomes noise and gets ignored.
"""

from __future__ import annotations

from legalaid.ingestion.dedup import (
    DedupItem,
    exact_hash,
    find_duplicates,
    hamming,
    normalise,
    shingles,
    simhash,
)

S73 = (
    "When a contract has been broken, the party who suffers by such breach is entitled to "
    "receive, from the party who has broken the contract, compensation for any loss or damage "
    "caused to him thereby, which naturally arose in the usual course of things from such "
    "breach, or which the parties knew, when they made the contract, to be likely to result "
    "from the breach of it."
)


def test_normalise_strips_case_and_punctuation() -> None:
    assert normalise("Section 73;  Compensation!") == "section 73 compensation"


def test_exact_hash_ignores_whitespace_and_case() -> None:
    assert exact_hash("Section  73\n\nCompensation") == exact_hash("section 73 compensation")


def test_exact_hash_differs_on_real_change() -> None:
    assert exact_hash(S73) != exact_hash(S73.replace("compensation", "damages"))


def test_shingles_are_word_level_and_overlap() -> None:
    result = shingles("a b c d e", size=3)
    assert result == ["a b c", "b c d", "c d e"]


def test_shingles_handles_text_shorter_than_the_window() -> None:
    assert shingles("a b", size=4) == ["a b"]
    assert shingles("", size=4) == []


def test_simhash_is_deterministic() -> None:
    assert simhash(S73) == simhash(S73)


def test_simhash_is_close_for_a_one_word_amendment() -> None:
    """The case an exact hash cannot see: same provision, one word amended."""
    amended = S73.replace("naturally arose", "ordinarily arose")
    distance = hamming(simhash(S73), simhash(amended))
    assert distance <= 6, f"expected a small distance for a one-word change, got {distance}"


def test_simhash_is_far_for_unrelated_provisions() -> None:
    other = (
        "A bailment is the delivery of goods by one person to another for some purpose, upon a "
        "contract that they shall, when the purpose is accomplished, be returned or otherwise "
        "disposed of according to the directions of the person delivering them."
    )
    assert hamming(simhash(S73), simhash(other)) > 12


def test_finds_exact_duplicates() -> None:
    items = [
        DedupItem("a", S73),
        DedupItem("b", S73),
        DedupItem("c", "something entirely different about bailment and delivery of goods"),
    ]
    report = find_duplicates(items)
    assert report.exact_duplicates == 1
    assert report.exact_groups[0].members == ["a", "b"]


def test_exact_duplicates_are_not_double_counted_as_near() -> None:
    report = find_duplicates([DedupItem("a", S73), DedupItem("b", S73)])
    assert report.exact_duplicates == 1
    assert report.near_duplicates == 0


def test_finds_near_duplicates() -> None:
    items = [
        DedupItem("original", S73),
        DedupItem("amended", S73.replace("naturally arose", "ordinarily arose")),
    ]
    report = find_duplicates(items, near_distance=8)
    assert report.near_duplicates == 1


def test_does_not_flag_distinct_provisions_on_the_same_subject() -> None:
    """s.73 and s.74 both concern compensation for breach but are different provisions."""
    s74 = (
        "When a contract has been broken, if a sum is named in the contract as the amount to be "
        "paid in case of such breach, or if the contract contains any other stipulation by way "
        "of penalty, the party complaining of the breach is entitled, whether or not actual "
        "damage or loss is proved to have been caused thereby, to receive from the party who has "
        "broken the contract reasonable compensation not exceeding the amount so named."
    )
    report = find_duplicates([DedupItem("s73", S73), DedupItem("s74", s74)])
    assert report.exact_duplicates == 0
    assert report.near_duplicates == 0, "distinct provisions must not be reported as duplicates"


def test_structural_duplicates_detect_the_same_section_twice() -> None:
    items = [
        DedupItem("x", "text one", act="Indian Contract Act, 1872", section="73"),
        DedupItem("y", "text two", act="Indian Contract Act, 1872", section="73"),
        DedupItem("z", "text three", act="Indian Contract Act, 1872", section="74"),
    ]
    report = find_duplicates(items)
    assert report.structural_duplicates == 1


def test_short_texts_are_excluded_from_near_comparison() -> None:
    """Every act has a near-identical short-title clause; those are not duplicates worth
    reporting."""
    items = [
        DedupItem("a", "This Act may be called the First Act, 1900."),
        DedupItem("b", "This Act may be called the Second Act, 1901."),
    ]
    report = find_duplicates(items, min_chars=120)
    assert report.near_duplicates == 0


def test_duplicate_rate_is_a_fraction_of_the_whole() -> None:
    items = [DedupItem("a", S73), DedupItem("b", S73)] + [
        DedupItem(f"u{i}", f"unique provision number {i} " * 12) for i in range(8)
    ]
    report = find_duplicates(items)
    assert report.total == 10
    assert 0.0 < report.duplicate_rate <= 0.2
