"""Supreme Court of India judgment fetcher.

Source: the Indian Supreme Court Judgments open dataset on AWS Open Data, sourced from the
official Supreme Court e-SCR portal (scr.sci.gov.in). Licensed **CC-BY-4.0**, so it may be
redistributed with attribution - which is why it was chosen over IL-PCSR and ILDC/CJPE, both
CC-BY-NC with no redistribution rights.

The full dataset is ~52 GB. We never download that. Per-year metadata parquet files are under
1 MB each, and individual judgment PDFs are fetched only for the curated subset, so the whole
MVP corpus costs a few tens of megabytes.

Layout discovered by listing the public bucket (the README does not spell out the object
naming):

    metadata/parquet/year=YYYY/metadata.parquet     ~940 KB, 856 rows for 2023
    data/pdf/year=YYYY/english/{path}_EN.pdf        note the **_EN** suffix

Structured metadata (citation, judges, decision date, disposal nature) comes from the parquet;
judgment text comes from the PDF. Nothing is synthesised.
"""

from __future__ import annotations

import io
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime

from legalaid.core.logging import get_logger

logger = get_logger(__name__)

BUCKET_URL = "https://indian-supreme-court-judgments.s3.amazonaws.com"
SOURCE_AUTHORITY = "Supreme Court of India (e-SCR), via AWS Open Data"
LICENCE = "CC-BY-4.0"
ATTRIBUTION = (
    "Judgment texts from the Indian Supreme Court Judgments open dataset (CC-BY-4.0), "
    "sourced from the Supreme Court of India e-SCR portal."
)
USER_AGENT = "LegalAId/0.1 academic-legal-research"

# PDF extraction of these judgments yields typographic ligatures that would otherwise corrupt
# both embeddings and any quoted citation ("speci<fi>c performance").
_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\ufb05": "st", "\ufb06": "st",
    # The e-SCR PDFs also encode fi/fl via a private-use mapping that decodes to this run.
    "\u2229\u00ac\u00fc": "fi",
    "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u00a0": " ",
}


# Words that the e-SCR PDFs reliably split after an fi/fl ligature. Restricted to real legal
# vocabulary so no legitimate two-word phrase is ever welded together.
_LIGATURE_FIXES: dict[str, str] = {
    "fi led": "filed",
    "fi ling": "filing",
    "fi le": "file",
    "fi les": "files",
    "fi nal": "final",
    "fi nally": "finally",
    "fi ndings": "findings",
    "fi nding": "finding",
    "fi rst": "first",
    "fi rm": "firm",
    "fi ne": "fine",
    "fi xed": "fixed",
    "specifi c": "specific",
    "specifi cally": "specifically",
    "specifi ed": "specified",
    "justifi ed": "justified",
    "justifi cation": "justification",
    "notifi cation": "notification",
    "notifi ed": "notified",
    "certifi cate": "certificate",
    "certifi ed": "certified",
    "satisfi ed": "satisfied",
    "benefi t": "benefit",
    "benefi ciary": "beneficiary",
    "suffi cient": "sufficient",
    "offi cer": "officer",
    "offi cial": "official",
    "offi ce": "office",
    "affi davit": "affidavit",
    "affi rmed": "affirmed",
    "diffi cult": "difficult",
    "suffi ciently": "sufficiently",
    "confi rmed": "confirmed",
    "confl ict": "conflict",
    "infl uence": "influence",
    "profi t": "profit",
    "profi ts": "profits",
    "verifi ed": "verified",
    "modifi ed": "modified",
    "identifi ed": "identified",
    "classifi ed": "classified",
    "qualifi ed": "qualified",
    "insuffi cient": "insufficient",
    "effi cacy": "efficacy",
    "signifi cant": "significant",
    "signifi cance": "significance",
    "benefi cial": "beneficial",
    "defi ned": "defined",
    "defi nition": "definition",
    "fl ow": "flow",
    "refl ect": "reflect",
}

_LIGATURE_SPLIT_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in sorted(_LIGATURE_FIXES, key=len, reverse=True)) + r")\b",
    re.I,
)


@dataclass(slots=True)
class CaseRecord:
    """A judgment with its metadata and text, ready for chunking."""

    case_id: str
    title: str
    petitioner: str
    respondent: str
    court: str
    citation: str
    neutral_citation: str | None
    decision_date: str | None
    judges: list[str]
    author_judge: str | None
    disposal_nature: str | None
    year: int
    path: str
    source_url: str
    text: str
    page_count: int
    headnote: str | None = None
    issues: str | None = None
    legal_provisions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def short_title(self) -> str:
        if self.petitioner and self.respondent:
            return f"{_titlecase(self.petitioner)} v. {_titlecase(self.respondent)}"
        return _titlecase(self.title)


class CaseLawError(RuntimeError):
    """Raised when the dataset cannot be fetched or parsed."""


def _get(path: str, *, timeout: float = 180.0, retries: int = 3) -> bytes:
    url = f"{BUCKET_URL}/{path}"
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise CaseLawError(f"Not found: {url}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
        if attempt < retries:
            time.sleep(2.0 * attempt)
    raise CaseLawError(f"GET failed after {retries} attempts: {url} ({last})") from last


def clean_judgment_text(text: str) -> str:
    """Repair ligatures and PDF line wrapping in judgment text.

    The e-SCR PDFs embed fi/fl/ff as separate glyphs that extract with a stray space after
    them, so naive substitution yields "specifi c performance" and "fi led". Left alone this
    corrupts embeddings (the token is "fi", not "specific") and makes any quoted citation
    visibly wrong, so the split word is stitched back together.
    """
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    # Drop any remaining private-use glyphs rather than embedding mojibake.
    text = re.sub(r"[\ue000-\uf8ff]", "", text)

    # Rejoin ligature-split words. A general rule like "<...fi> <short word>" would wrongly
    # weld real phrases ("of the", "if he"), so only known legal vocabulary is repaired.
    text = _LIGATURE_SPLIT_RE.sub(lambda m: _LIGATURE_FIXES[m.group(0).lower()], text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # S.C.R. pages carry marginal line-reference letters (A-H) down the gutter, which extract
    # as a standalone "A B C D E F G H" run. They are layout furniture, not text: left in, they
    # pollute embeddings and appear inside quoted passages.
    text = re.sub(r"^\s*(?:[A-H]\s+){3,}[A-H]\s*$", "", text, flags=re.M)
    text = re.sub(r"(?:\b[A-H]\s+){4,}(?=[A-Z0-9])", "", text)

    # Running headers/footers repeated on every page add nothing and skew term frequencies.
    text = re.sub(r"^\s*\d{1,4}\s+SUPREME COURT REPORTS\s+\[\d{4}\]\s+\d+\s+S\.C\.R\.\s*$", "", text, flags=re.M | re.I)
    text = re.sub(r"\b\d{1,4}\s+SUPREME COURT REPORTS\s+\[\d{4}\]\s+\d+\s+S\.C\.R\.", " ", text, flags=re.I)

    # Rejoin hyphenated line breaks: "perfor-\nmance" -> "performance".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # A single newline inside a paragraph is a PDF wrap, not a break.
    text = re.sub(r"(?<![\n.;:])\n(?![\n])", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _titlecase(name: str) -> str:
    """e-SCR stores parties in caps; title-case them for readable citations."""
    cleaned = " ".join(name.split())
    if not cleaned:
        return ""
    small = {"and", "of", "the", "for", "in", "v.", "vs", "versus", "ors", "anr"}
    words = []
    for word in cleaned.lower().split():
        words.append(word if word in small and words else word.capitalize())
    return " ".join(words)


def load_year_metadata(year: int) -> list[dict[str, object]]:
    """Download and parse one year of judgment metadata (well under 1 MB)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise CaseLawError("pyarrow is required to read the case-law metadata") from exc

    raw = _get(f"metadata/parquet/year={year}/metadata.parquet")
    table = pq.read_table(io.BytesIO(raw))
    rows = table.to_pylist()
    logger.info("case_metadata_loaded", year=year, rows=len(rows), bytes=len(raw))
    return rows


# Headnote structure in e-SCR judgments, used to pull issues and cited provisions.
_ISSUE_RE = re.compile(
    r"Issue[s]?\s+for\s+consideration\s*:?\s*(.{40,1200}?)(?=\n\n|HELD|Held|CASE|$)",
    re.S | re.I,
)
# The headnote runs from the HEADNOTES marker to whichever structural marker comes next.
# "CASE LAW" is NOT one of them in these documents, and a lazy quantifier anchored on an
# absent terminator matches nothing, so the terminators are enumerated explicitly and the
# match is greedy up to the first one that appears.
_HEADNOTE_RE = re.compile(
    r"HEADNOTES?\s*[:.]?\s*(.+?)"
    r"(?=\n\s*(?:CASE\s+LAW|JUDGMENT|CIVIL\s+APPELLATE|CRIMINAL\s+APPELLATE|"
    r"ORDER\s*$|Appearances|List\s+of\s+Acts|List\s+of\s+Keywords)|\Z)",
    re.S | re.I,
)
_PROVISION_RE = re.compile(
    r"\b((?:Indian\s+)?(?:Contract|Evidence|Penal|Companies|Consumer\s+Protection|"
    r"Information\s+Technology|Specific\s+Relief|Arbitration\s+and\s+Conciliation)\s+Act,?\s*\d{4}"
    r"|Code\s+of\s+(?:Civil|Criminal)\s+Procedure,?\s*\d{4}"
    r"|Bharatiya\s+(?:Nyaya\s+Sanhita|Sakshya\s+Adhiniyam|Nagarik\s+Suraksha\s+Sanhita),?\s*\d{4}"
    r"|Constitution\s+of\s+India)",
    re.I,
)
_SECTION_RE = re.compile(r"\b(?:Section|S\.|Order)\s+(\d+[A-Z]{0,2})", re.I)
_NEUTRAL_CITATION_RE = re.compile(r"\b(\d{4}\s+INSC\s+\d+)\b", re.I)


def parse_judgment_structure(text: str) -> tuple[str | None, str | None, list[str]]:
    """Extract (headnote, issues, cited provisions) from judgment text."""
    headnote_match = _HEADNOTE_RE.search(text)
    headnote = headnote_match.group(1).strip() if headnote_match else None

    issue_match = _ISSUE_RE.search(text)
    issues = " ".join(issue_match.group(1).split()) if issue_match else None

    provisions: list[str] = []
    seen: set[str] = set()
    scope = headnote or text[:6000]
    for match in _PROVISION_RE.finditer(scope):
        name = " ".join(match.group(1).split())
        key = name.lower().replace(",", "")
        if key not in seen:
            seen.add(key)
            provisions.append(name)
    return headnote, issues, provisions[:10]


def fetch_judgment(row: dict[str, object], *, timeout: float = 180.0) -> CaseRecord:
    """Fetch and parse one judgment PDF."""
    import pymupdf

    path = str(row.get("path") or "").strip()
    if not path:
        raise CaseLawError("metadata row has no path")
    year = int(str(row.get("year") or path.split("_")[0]))

    object_key = f"data/pdf/year={year}/english/{path}_EN.pdf"
    pdf_bytes = _get(object_key, timeout=timeout)

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = [document[i].get_text() for i in range(document.page_count)]
    page_count = document.page_count
    document.close()

    text = clean_judgment_text("\n\n".join(pages))
    headnote, issues, provisions = parse_judgment_structure(text)

    judge_field = str(row.get("judge") or "")
    judges = [j.strip() for j in re.split(r",| AND |&", judge_field, flags=re.I) if j.strip()]

    neutral = None
    neutral_match = _NEUTRAL_CITATION_RE.search(text[:2000])
    if neutral_match:
        neutral = neutral_match.group(1).upper()

    keywords = sorted(
        {m.group(0).title() for m in _SECTION_RE.finditer(headnote or text[:4000])}
    )[:12]

    return CaseRecord(
        case_id=str(row.get("case_id") or path),
        title=" ".join(str(row.get("title") or "").split()),
        petitioner=str(row.get("petitioner") or ""),
        respondent=str(row.get("respondent") or ""),
        court=str(row.get("court") or "Supreme Court of India"),
        citation=str(row.get("citation") or ""),
        neutral_citation=neutral,
        decision_date=str(row.get("decision_date") or "") or None,
        judges=judges,
        author_judge=str(row.get("author_judge") or "") or None,
        disposal_nature=str(row.get("disposal_nature") or "") or None,
        year=year,
        path=path,
        source_url=f"{BUCKET_URL}/{object_key}",
        text=text,
        page_count=page_count,
        headnote=headnote,
        issues=issues,
        legal_provisions=provisions,
        keywords=keywords,
    )
