"""India Code fetcher — genuine, section-level Indian statutory text.

India Code (https://www.indiacode.nic.in) is the official Government of India legislation
repository. It exposes no REST or OAI API, so this module uses the same endpoints the site's
own front end uses. That route was reverse-engineered and verified against the live site; see
`docs/data-sources.md` for the evidence trail.

Route
-----
1. ``GET /handle/123456789/{handle}`` — the act page. It embeds the complete section index as
   anchors of the form ``/show-data?...&actid=...&sectionId=...&sectionno=...&orderno=...``.
   HEAD requests are refused by the server, so always use GET.
2. ``GET /SectionPageContent?actid={actid}&sectionID={sid}`` — JSON ``{content, footnote}``
   carrying the actual provision text and its amendment footnotes. The ``/show-data`` HTML is
   *not* usable directly because the body is populated by AJAX after page load.
3. ``GET /bitstream/...pdf`` — the consolidated act PDF, retained for the page-level document
   viewer.

Nothing here fabricates legal text. If a section cannot be fetched it is reported as a failure
rather than filled in.
"""

from __future__ import annotations

import html as htmllib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

from legalaid.core.logging import get_logger

logger = get_logger(__name__)

# Migrated August 2026: www.indiacode.nic.in now serves only a redirect notice and returns 404
# for every /handle/ path. The new host keeps the same URL structure, so only the origin
# changed. The old host is kept as a fallback in case the migration is rolled back.
BASE_URL = "https://indiacode.gov.in"
LEGACY_BASE_URL = "https://www.indiacode.nic.in"
SOURCE_AUTHORITY = "India Code, Government of India (Legislative Department)"

# India Code sits behind a WAF that rejects User-Agent strings containing parentheses with a
# 403 - the conventional "Product/1.0 (comment)" form does not work. A bare product token does.
# urllib's default header is separately rejected with a 404, so a UA must still be sent.
USER_AGENT = "LegalAId/0.1 academic-legal-research"

# Matches one section entry in the act page's accordion index.
_SECTION_ANCHOR = re.compile(
    r"href=/show-data\?"
    r"abv=(?P<abv>[^&]*)"
    r"&statehandle=(?P<statehandle>[^&]*)"
    r"&actid=(?P<actid>[^&]*)"
    r"&sectionId=(?P<section_id>\d+)"
    r"&sectionno=(?P<section_no>[^&]*)"
    r"&orderno=(?P<order_no>\d+)"
    r"&orgactid=(?P<orgactid>[^ >]+)>"
    r"\s*<span[^>]*>\s*Section\s*(?P<label>[^<]*?)\s*</span>"
    r"(?P<title>[^<]*)",
    re.I,
)

_PDF_HREF = re.compile(r'href="(/bitstream/[^"]+\.pdf)"', re.I)
_TITLE = re.compile(r"<title>\s*India Code:\s*(?P<title>.*?)\s*</title>", re.S | re.I)


@dataclass(frozen=True, slots=True)
class SectionRef:
    """A section as advertised by the act page index, before its text is fetched."""

    section_id: str
    section_number: str
    heading: str
    order_no: int
    actid: str

    @property
    def content_url(self) -> str:
        query = urllib.parse.urlencode({"actid": self.actid, "sectionID": self.section_id})
        return f"{BASE_URL}/SectionPageContent?{query}"

    def show_data_url(self, abv: str, statehandle: str) -> str:
        query = urllib.parse.urlencode(
            {
                "abv": abv,
                "statehandle": statehandle,
                "actid": self.actid,
                "sectionId": self.section_id,
                "sectionno": self.section_number,
                "orderno": self.order_no,
                "orgactid": self.actid,
            }
        )
        return f"{BASE_URL}/show-data?{query}"


@dataclass(slots=True)
class FetchedSection:
    """A section with its verified statutory text."""

    section_number: str
    heading: str
    text: str
    footnotes: str
    order_no: int
    section_id: str
    source_url: str
    # Publisher-supplied repeal flag. Available only from the DSpace API; the old HTML route
    # forced repeal status to be inferred from heading text, which is why the inference in
    # `services.knowledge.ingest.is_repealed` still exists as a second line of defence.
    # None means "the source did not say", not "not repealed".
    repealed: bool | None = None
    # Content hash of the normalised text, for change detection that a length comparison cannot
    # do: an amendment that preserves length would otherwise be invisible.
    content_sha256: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(slots=True)
class FetchedAct:
    handle: str
    actid: str
    title: str
    source_url: str
    pdf_url: str | None
    sections: list[FetchedSection] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def section_count(self) -> int:
        return len(self.sections)


class IndiaCodeError(RuntimeError):
    """Raised when India Code cannot be fetched or parsed."""


def _get(url: str, *, timeout: float, retries: int = 4, backoff: float = 3.0) -> bytes:
    """GET with retries and rate-limit-aware backoff.

    Two observed behaviours of the live site drive this implementation:

    * HEAD is refused outright, so only GET is used.
    * Rapid sequential requests trigger ``403 Forbidden`` for a short period. It is throttling,
      not authorisation: the same URL succeeds again after a pause. A 403 therefore gets a
      longer, escalating sleep rather than being treated as fatal.

    A ``User-Agent`` must be sent; urllib's default header yields ``404``.
    """
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            # 403/429 mean "slow down"; back off harder than for a transport error.
            throttled = exc.code in (403, 429)
            sleep_for = backoff * attempt * (3.0 if throttled else 1.0)
            if attempt < retries:
                logger.warning(
                    "india_code_retry",
                    url=url,
                    attempt=attempt,
                    status=exc.code,
                    throttled=throttled,
                    sleep=round(sleep_for, 1),
                )
                time.sleep(sleep_for)
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt < retries:
                sleep_for = backoff * attempt
                logger.warning(
                    "india_code_retry", url=url, attempt=attempt, error=str(exc), sleep=sleep_for
                )
                time.sleep(sleep_for)
    raise IndiaCodeError(f"GET failed after {retries} attempts: {url} ({last})") from last


def strip_html(fragment: str) -> str:
    """Convert an India Code HTML fragment to clean, embeddable statutory text.

    The upstream markup has two quirks that must be handled deliberately, because both
    corrupt embeddings and quoted citations if ignored:

    * Hard ``\\r\\n`` line breaks are **PDF line-wrap artifacts**, not paragraph boundaries -
      they fall mid-sentence ("has been\\r\\nbroken"). They are joined back into flowing text.
      True paragraph breaks are marked up as ``</br><hr class="hr1"/>`` instead.
    * Inline tags sit flush against following text (``<i>Explanation</i>.In estimating``), so
      naive tag removal welds words together. Tags become a space, and punctuation spacing is
      repaired afterwards.
    """
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", fragment, flags=re.S | re.I)

    # Real paragraph separators in this markup.
    text = re.sub(r"(?:</br>\s*)?(?:<hr[^>]*>\s*)+", "\n\n", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n\n", text, flags=re.I)
    text = re.sub(r"</(p|div|li|tr|h[1-6])>", "\n\n", text, flags=re.I)

    # Remaining tags become a space so words never fuse together.
    text = re.sub(r"<[^>]+>", " ", text)
    text = htmllib.unescape(text)
    text = text.replace("\u00a0", " ")

    # Rejoin PDF line wraps: a single newline inside a paragraph is not a break.
    text = re.sub(r"(?<!\n)\r?\n(?!\n)", " ", text)
    text = text.replace("\r", "")

    # Repair spacing artifacts left by tag removal.
    text = re.sub(r"\s+([.,;:)\]])", r"\1", text)
    text = re.sub(r"([([])\s+", r"\1", text)
    # "Explanation.In estimating" -> "Explanation. In estimating"
    text = re.sub(r"(?<=[a-z])\.(?=[A-Z][a-z])", ". ", text)

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_section_index(page_html: str) -> list[SectionRef]:
    """Extract the ordered section index from an act page."""
    refs: list[SectionRef] = []
    seen: set[str] = set()
    for match in _SECTION_ANCHOR.finditer(page_html):
        section_id = match.group("section_id")
        if section_id in seen:
            continue
        seen.add(section_id)
        label = htmllib.unescape(match.group("label") or "").strip().rstrip(".")
        number = htmllib.unescape(match.group("section_no") or "").strip() or label
        heading = strip_html(match.group("title") or "").strip(" .\u00a0")
        refs.append(
            SectionRef(
                section_id=section_id,
                section_number=number,
                heading=heading,
                order_no=int(match.group("order_no")),
                actid=urllib.parse.unquote(match.group("actid")),
            )
        )
    refs.sort(key=lambda r: r.order_no)
    return refs


def parse_act_title(page_html: str) -> str | None:
    match = _TITLE.search(page_html)
    if not match:
        return None
    title = htmllib.unescape(match.group("title")).strip()
    return title or None


def parse_pdf_url(page_html: str) -> str | None:
    """Prefer the English-language bitstream (``A...``) over the Hindi one (``H...``)."""
    hrefs = [m.group(1) for m in _PDF_HREF.finditer(page_html)]
    if not hrefs:
        return None
    for href in hrefs:
        filename = href.rsplit("/", 1)[-1]
        if filename.upper().startswith("A"):
            return f"{BASE_URL}{href}"
    return f"{BASE_URL}{hrefs[0]}"


def fetch_section_text(ref: SectionRef, *, timeout: float = 60.0) -> FetchedSection:
    """Fetch one section's provision text and amendment footnotes."""
    raw = _get(ref.content_url, timeout=timeout)
    try:
        payload = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise IndiaCodeError(f"SectionPageContent returned non-JSON for {ref.section_id}") from exc

    return FetchedSection(
        section_number=ref.section_number,
        heading=ref.heading,
        text=strip_html(str(payload.get("content") or "")),
        footnotes=strip_html(str(payload.get("footnote") or "")),
        order_no=ref.order_no,
        section_id=ref.section_id,
        source_url=ref.content_url,
    )


def fetch_act(
    handle: str,
    *,
    timeout: float = 60.0,
    delay: float = 0.4,
    max_sections: int | None = None,
    progress: bool = False,
    only_sections: Collection[str] | None = None,
) -> FetchedAct:
    """Fetch an act and the text of each section in scope.

    `delay` throttles requests out of courtesy to a government server.

    `only_sections` restricts fetching to specific section numbers. This is applied *before*
    any section text is requested, which matters: the Bharatiya Nyaya Sanhita has 358 sections
    but the corpus manifest keeps 14, so filtering afterwards would mean 344 pointless requests
    to a government server and about three wasted minutes per run.
    """
    act_url = f"{BASE_URL}/handle/123456789/{handle}"
    page = _get(act_url, timeout=timeout).decode("utf-8", "replace")

    refs = parse_section_index(page)
    if not refs:
        raise IndiaCodeError(
            f"No sections parsed from {act_url}. The page structure may have changed."
        )
    if only_sections is not None:
        wanted = {str(s).strip().upper() for s in only_sections}
        refs = [r for r in refs if r.section_number.strip().upper() in wanted]
    if max_sections is not None:
        refs = refs[:max_sections]

    act = FetchedAct(
        handle=handle,
        actid=refs[0].actid,
        title=parse_act_title(page) or f"India Code handle {handle}",
        source_url=act_url,
        pdf_url=parse_pdf_url(page),
    )

    for index, ref in enumerate(refs, start=1):
        try:
            section = fetch_section_text(ref, timeout=timeout)
        except IndiaCodeError as exc:
            act.failures.append({"section": ref.section_number, "error": str(exc)})
            logger.warning("india_code_section_failed", section=ref.section_number, error=str(exc))
            continue
        if section.is_empty:
            # Chapter headings and repealed sections legitimately have no body text.
            act.failures.append({"section": ref.section_number, "error": "empty content"})
        else:
            act.sections.append(section)
        if progress and index % 25 == 0:
            logger.info(
                "india_code_progress", act=act.title, fetched=index, total=len(refs)
            )
        if delay:
            time.sleep(delay)

    logger.info(
        "india_code_act_fetched",
        act=act.title,
        sections=act.section_count,
        skipped=len(act.failures),
    )
    return act


def iter_section_records(act: FetchedAct) -> Iterator[dict[str, object]]:
    """Yield provenance-complete records ready for chunking and storage."""
    for section in act.sections:
        yield {
            "act_title": act.title,
            "act_handle": act.handle,
            "actid": act.actid,
            "section_number": section.section_number,
            "heading": section.heading,
            "text": section.text,
            "footnotes": section.footnotes,
            "order_no": section.order_no,
            "source_url": section.source_url,
            "act_url": act.source_url,
            "pdf_url": act.pdf_url,
            "source_authority": SOURCE_AUTHORITY,
            "retrieved_at": act.retrieved_at.isoformat(),
        }
