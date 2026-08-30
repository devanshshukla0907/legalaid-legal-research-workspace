"""India Code via the DSpace 9 REST API.

Why this module exists
----------------------
India Code migrated in August 2026 from ``www.indiacode.nic.in`` to ``indiacode.gov.in`` and
became a DSpace 9.1 Angular application. Both foundations of the previous HTML scraper are gone:

* every ``/handle/`` path on the old host now returns 404 (the host serves only a migration
  notice);
* on the new host the act page is an empty SPA shell, so there is no section index to parse, and
  the custom ``/SectionPageContent`` endpoint returns 502.

The replacement is strictly better than what it replaces. DSpace exposes a documented REST API,
and India Code models **each section as its own item** carrying:

===============================  ====================================================
``dc.identifier.collection``     ``ACT`` or ``SECTION``
``dc.identifier.section_number`` the section number
``dc.title``                     the section heading
``dc.identifier.section_page_note``  the provision text, as HTML
``dc.identifier.act_id``         links a section to its act
``dc.identifier.state_name``     ``CENTRAL`` or the state, so state copies are separable
``dc.identifier.repealed``       per-section repeal flag, from the publisher
``dc.date.enforcement_date``     commencement date on the act item
``dc.identifier.page_number``    page in the consolidated PDF
===============================  ====================================================

Two of those - a publisher-supplied repeal flag and a real commencement date - were not
available at all through the old route and had to be inferred from heading text.

`FetchedAct` and `FetchedSection` are deliberately the same shapes the scraper produced, so the
ingestion pipeline is unchanged by the migration.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime

from legalaid.core.logging import get_logger
from legalaid.ingestion.india_code import (
    FetchedAct,
    FetchedSection,
    IndiaCodeError,
    strip_html,
)

logger = get_logger(__name__)

HOST = "https://indiacode.gov.in"
API = f"{HOST}/server/api"
SOURCE_AUTHORITY = "India Code, Government of India (Legislative Department)"

# DSpace sits behind nginx that returns 502 under burst load, so requests are paced and retried.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
PAGE_SIZE = 100


@dataclass(frozen=True, slots=True)
class ActRef:
    """An act located in the API, before its sections are fetched."""

    act_id: str
    handle: str
    title: str
    year: int | None
    act_number: str
    enforcement_date: str | None
    repealed: bool
    state_name: str

    @property
    def source_url(self) -> str:
        return f"{HOST}/handle/{self.handle}"

    @property
    def is_central(self) -> bool:
        return self.state_name.strip().upper() == "CENTRAL"


def _get(url: str, *, timeout: float = 60.0, retries: int = 4, backoff: float = 3.0) -> dict:
    """GET JSON with retries. 502s from the fronting nginx are common under load."""
    last_error = ""
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url)
        request.add_header("User-Agent", USER_AGENT)
        request.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            # 502/503 are transient; 404 is not worth retrying.
            if exc.code not in {429, 500, 502, 503, 504}:
                raise IndiaCodeError(f"{url} -> {last_error}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            sleep_for = backoff * attempt
            logger.warning(
                "india_code_api_retry", url=url[:120], attempt=attempt, error=last_error,
                sleep=sleep_for,
            )
            time.sleep(sleep_for)
    raise IndiaCodeError(f"{url} failed after {retries} attempts: {last_error}")


def _meta(item: dict, key: str, default: str = "") -> str:
    entries = (item.get("metadata") or {}).get(key) or []
    return str(entries[0].get("value", default)) if entries else default


def _content_hash(text: str) -> str:
    """Stable hash over whitespace-normalised text, so reformatting is not a false change."""
    normalised = " ".join((text or "").split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:32]


def _search(query: str, *, page: int = 0, size: int = PAGE_SIZE, sort: str | None = None) -> dict:
    # `embed=indexableObject` inlines each hit's full item, including its metadata. Without it
    # the response carries only links and every section costs a separate request - 268 extra
    # round trips for one act, which turned a 20-second fetch into several minutes.
    params = {"query": query, "page": page, "size": size, "embed": "indexableObject"}
    if sort:
        params["sort"] = sort
    return _get(f"{API}/discover/search/objects?{urllib.parse.urlencode(params)}")


def _objects(payload: dict) -> tuple[list[dict], int]:
    """Extract embedded items and the total count from a discover response."""
    result = payload.get("_embedded", {}).get("searchResult", {})
    raw = result.get("_embedded", {}).get("objects", [])
    total = int(result.get("page", {}).get("totalElements") or 0)
    items: list[dict] = []
    for obj in raw:
        embedded = obj.get("_embedded", {}).get("indexableObject")
        if embedded:
            items.append(embedded)
            continue
        href = obj.get("_links", {}).get("indexableObject", {}).get("href")
        if href:
            items.append(_get(href))
    return items, total


def _to_act_ref(item: dict) -> ActRef:
    year_raw = _meta(item, "dc.date.act_year")
    return ActRef(
        act_id=_meta(item, "dc.identifier.act_id"),
        handle=item.get("handle", ""),
        title=_meta(item, "dc.title") or item.get("name", ""),
        year=int(year_raw) if year_raw.isdigit() else None,
        act_number=_meta(item, "dc.identifier.act_number"),
        enforcement_date=_meta(item, "dc.date.enforcement_date") or None,
        repealed=_meta(item, "dc.identifier.repealed").lower() == "true",
        state_name=_meta(item, "dc.identifier.state_name"),
    )


def find_acts(title: str, *, central_only: bool = True, limit: int = 10) -> list[ActRef]:
    """Locate acts by title.

    `central_only` matters: India Code holds state-adopted copies of central acts, and a search
    for "Indian Contract Act" returns a Chhattisgarh copy before the central one. Ingesting the
    wrong copy would produce plausible but wrong provenance.
    """
    # CENTRAL is filtered server-side rather than after the fact. India Code holds dozens of
    # state adoptions of each central act and the relevance ranking does not prefer the central
    # copy - the central Right to Information Act ranked eighth - so post-filtering a small page
    # of results silently reported acts as missing when they were simply further down the list.
    clauses = [f'dc.title:"{title}"', 'dc.identifier.collection:"ACT"']
    if central_only:
        clauses.append('dc.identifier.state_name:"CENTRAL"')
    payload = _search(" AND ".join(clauses), size=limit)
    items, _total = _objects(payload)
    refs = [_to_act_ref(i) for i in items if _meta(i, "dc.identifier.collection") == "ACT"]
    if central_only:
        refs = [r for r in refs if r.is_central]
    # Prefer an exact title match; the search is fuzzy and ranks loosely.
    wanted = title.strip().lower().lstrip("the ")
    refs.sort(key=lambda r: (wanted not in r.title.lower().lstrip("the "), len(r.title)))
    return refs


def find_act(title: str) -> ActRef:
    refs = find_acts(title)
    if not refs:
        raise IndiaCodeError(f"No central act found on India Code for {title!r}")
    return refs[0]


def fetch_sections(act_id: str, *, delay: float = 0.4, progress: bool = False) -> list[FetchedSection]:
    """Every section of one act, in statutory order.

    Sections are paginated; `order_number` is the publisher's own ordering and is what the list
    is sorted by, rather than the section number, because section numbers are not numerically
    sortable ("10A" follows "10").
    """
    query = f'dc.identifier.act_id:"{act_id}" AND dc.identifier.collection:"SECTION"'
    sections: list[FetchedSection] = []
    page = 0
    total = None

    while True:
        payload = _search(query, page=page, size=PAGE_SIZE)
        items, reported_total = _objects(payload)
        if total is None:
            total = reported_total
        if not items:
            break

        for item in items:
            note = _meta(item, "dc.identifier.section_page_note")
            text = strip_html(note) if note else ""
            order_raw = _meta(item, "dc.identifier.order_number", "0")
            sections.append(
                FetchedSection(
                    section_number=_meta(item, "dc.identifier.section_number"),
                    heading=_meta(item, "dc.title") or item.get("name", ""),
                    text=text,
                    # The API returns text and footnotes in one field; footnotes are already
                    # inside `section_page_note` after an <hr>, and strip_html keeps them, so a
                    # separate footnote field would duplicate content rather than add any.
                    footnotes="",
                    order_no=int(order_raw) if order_raw.isdigit() else 0,
                    section_id=_meta(item, "dc.identifier.section_id"),
                    source_url=f"{HOST}/handle/{item.get('handle', '')}",
                    # The publisher's own flag, rather than inferring repeal from heading text.
                    repealed=_meta(item, "dc.identifier.repealed").lower() == "true",
                    content_sha256=_content_hash(text),
                )
            )

        page += 1
        if progress:
            logger.info("india_code_api_progress", act_id=act_id, fetched=len(sections),
                        total=total)
        if total is not None and len(sections) >= total:
            break
        if page > 60:  # pragma: no cover - guard against a pagination bug looping forever
            logger.warning("india_code_api_pagination_guard", act_id=act_id, pages=page)
            break
        if delay:
            time.sleep(delay)

    sections.sort(key=lambda s: s.order_no)
    return sections


def fetch_act(
    title_or_ref: str | ActRef,
    *,
    delay: float = 0.4,
    progress: bool = False,
) -> FetchedAct:
    """Fetch an act and all its sections through the API.

    Accepts a title (resolved via search) or an already-resolved `ActRef`. Returns the same
    `FetchedAct` shape the HTML scraper produced, so the ingestion pipeline is unaffected by the
    migration.
    """
    ref = title_or_ref if isinstance(title_or_ref, ActRef) else find_act(title_or_ref)
    sections = fetch_sections(ref.act_id, delay=delay, progress=progress)
    if not sections:
        raise IndiaCodeError(
            f"Found act {ref.title!r} ({ref.act_id}) but it reported no sections. "
            "Either the act id is wrong or the API shape has changed again."
        )

    act = FetchedAct(
        handle=ref.handle,
        actid=ref.act_id,
        title=ref.title,
        source_url=ref.source_url,
        # Bundles were empty for the acts probed, so there is no consolidated PDF to record
        # rather than a broken link to one.
        pdf_url=None,
        sections=sections,
        retrieved_at=datetime.now(UTC),
    )
    logger.info(
        "india_code_api_act_fetched",
        act=ref.title,
        act_id=ref.act_id,
        sections=len(sections),
        enforcement_date=ref.enforcement_date,
    )
    return act


def fetch_act_metadata(act_id: str) -> ActRef | None:
    """The act item itself, for commencement date and repeal status."""
    payload = _search(
        f'dc.identifier.act_id:"{act_id}" AND dc.identifier.collection:"ACT"', size=5
    )
    items, _total = _objects(payload)
    for item in items:
        ref = _to_act_ref(item)
        if ref.is_central:
            return ref
    return _to_act_ref(items[0]) if items else None
