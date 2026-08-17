"""Tests for judgment text cleaning and structure parsing.

These pin defects found against real e-SCR PDFs: ligature-split words, S.C.R. gutter letters,
running page headers, and a headnote terminator that did not exist in the documents.
"""

from __future__ import annotations

from legalaid.ingestion.case_law import clean_judgment_text, parse_judgment_structure

REAL_HEADNOTE = """[2023] 16 S.C.R. 872 : 2023 INSC 1043
872
CASE DETAILS
ELDECO HOUSING AND INDUSTRIES LIMITED
v.
ASHOK VIDYARTHI AND OTHERS
HEADNOTES
Issue for consideration: A suit for specifi c performance was fi led
seeking enforcement of MoU dated 31.08.1998. It is in the aforesaid suit that
application was fi led by respondent No. 1 under Order VII Rule 11(d) CPC.
Code of Civil Procedure, 1908 - Order VII Rule 11(d) - Whether evidence
can be examined at that stage:
Held: the order passed by the High Court cannot be legally sustained.
"""


class TestLigatureRepair:
    def test_split_ligature_words_are_rejoined(self) -> None:
        """PDFs emit "specifi c" and "fi led"; left alone these corrupt embeddings."""
        cleaned = clean_judgment_text("A suit for specifi c performance was fi led today.")
        assert "specific performance" in cleaned
        assert "filed" in cleaned
        assert "specifi c" not in cleaned

    def test_real_legal_phrases_are_not_welded(self) -> None:
        """The repair must be vocabulary-driven, not a generic short-word rule."""
        cleaned = clean_judgment_text("It is a matter of the contract, if he so pleads.")
        assert "of the contract" in cleaned
        assert "if he so pleads" in cleaned

    def test_various_ligature_words(self) -> None:
        cleaned = clean_judgment_text(
            "The offi cer gave notifi cation and the affi davit was verifi ed as suffi cient."
        )
        for expected in ("officer", "notification", "affidavit", "verified", "sufficient"):
            assert expected in cleaned


class TestLayoutFurniture:
    def test_scr_gutter_letters_are_removed(self) -> None:
        cleaned = clean_judgment_text("A B C D E F G H 600 the appellant contended that")
        assert "A B C D E F G H" not in cleaned
        assert "the appellant contended that" in cleaned

    def test_running_header_is_removed(self) -> None:
        cleaned = clean_judgment_text(
            "600 SUPREME COURT REPORTS [2023] 7 S.C.R. right to apply for appointment"
        )
        assert "SUPREME COURT REPORTS" not in cleaned
        assert "right to apply for appointment" in cleaned

    def test_hyphenated_line_breaks_rejoined(self) -> None:
        assert "performance" in clean_judgment_text("specific perfor-\nmance was sought")

    def test_pdf_wraps_are_rejoined(self) -> None:
        cleaned = clean_judgment_text("the party who suffers by\nsuch breach is entitled")
        assert "suffers by such breach" in cleaned


class TestJudgmentStructure:
    def test_headnote_is_extracted(self) -> None:
        """Regression: the terminator "CASE LAW" does not occur in these documents, so a
        lazy match anchored on it captured nothing at all."""
        headnote, issues, provisions = parse_judgment_structure(
            clean_judgment_text(REAL_HEADNOTE)
        )
        assert headnote is not None
        assert len(headnote) > 200
        assert "specific performance" in headnote

    def test_issues_are_extracted(self) -> None:
        _headnote, issues, _provisions = parse_judgment_structure(
            clean_judgment_text(REAL_HEADNOTE)
        )
        assert issues is not None
        assert "specific performance" in issues

    def test_cited_provisions_are_identified(self) -> None:
        _headnote, _issues, provisions = parse_judgment_structure(
            clean_judgment_text(REAL_HEADNOTE)
        )
        assert any("Civil Procedure" in p for p in provisions), provisions

    def test_missing_headnote_does_not_crash(self) -> None:
        headnote, issues, provisions = parse_judgment_structure("Plain text, no structure.")
        assert headnote is None
        assert provisions == []
