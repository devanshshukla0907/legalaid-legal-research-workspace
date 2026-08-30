"""Can section-level statutory text still be fetched after the India Code migration?

Two candidate routes emerged from the API probe:

  A. The legacy `/SectionPageContent?actid=..&sectionID=..` endpoint, which returned 200 on the
     new host. If it still serves real provision text, the existing ingester survives with only
     a BASE_URL change plus a new way to enumerate sections.

  B. DSpace items. The search results included entries like "Who are competent to contract."
     and "'Contingent contract' defined." - individual sections appear to be indexed as items,
     which would make the whole corpus reachable through the documented REST API.

Route B would be strictly better: a real API with metadata instead of HTML scraping. This script
establishes which routes actually work before any ingester is rewritten.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

HOST = "https://indiacode.gov.in"
API = f"{HOST}/server/api"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
# Central Indian Contract Act id, from dc.identifier.refact on the state copy.
CENTRAL_CONTRACT_ACT = "AC_CEN_3_20_00035_187209_1523268996428"


def get(url: str, *, accept: str = "application/json") -> tuple[int, str]:
    request = urllib.request.Request(url)
    request.add_header("User-Agent", UA)
    request.add_header("Accept", accept)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(400).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def meta(item: dict, key: str) -> str:
    entries = (item.get("metadata") or {}).get(key) or []
    return entries[0].get("value", "") if entries else ""


def route_a() -> None:
    print("=== ROUTE A: legacy /SectionPageContent ===")
    for section_id in ("1", "73", "1523269", "18729"):
        params = urllib.parse.urlencode(
            {"actid": CENTRAL_CONTRACT_ACT, "sectionID": section_id}
        )
        status, body = get(f"{HOST}/SectionPageContent?{params}")
        preview = body.strip().replace("\n", " ")[:110]
        print(f"  sectionID={section_id:<10} {status}  len={len(body):<7} {preview}")


def route_b() -> None:
    print("\n=== ROUTE B: are sections real DSpace items? ===")
    params = urllib.parse.urlencode(
        {"query": '"Compensation for loss or damage caused by breach of contract"', "size": 4}
    )
    status, body = get(f"{API}/discover/search/objects?{params}")
    if status != 200:
        print(f"  search failed: {status}")
        return
    payload = json.loads(body)
    objects = (
        payload.get("_embedded", {})
        .get("searchResult", {})
        .get("_embedded", {})
        .get("objects", [])
    )
    print(f"  hits: {len(objects)}")
    for obj in objects:
        href = obj.get("_links", {}).get("indexableObject", {}).get("href")
        if not href:
            continue
        istatus, ibody = get(href)
        if istatus != 200:
            continue
        item = json.loads(ibody)
        print(f"\n  --- {item.get('handle')}  {meta(item, 'dc.title')[:60]}")
        interesting = [
            "dc.identifier.collection",
            "dc.identifier.act_id",
            "dc.identifier.refact",
            "dc.identifier.sectionno",
            "dc.identifier.section_no",
            "dc.identifier.orderno",
            "dc.date.enforcement_date",
            "dc.identifier.repealed",
            "dc.identifier.state_name",
            "dc.description",
            "dc.description.abstract",
        ]
        for key in interesting:
            value = meta(item, key)
            if value:
                print(f"      {key:<32} {str(value)[:90]}")
        # Any metadata key we have not listed but which might carry the text.
        for key, entries in sorted((item.get("metadata") or {}).items()):
            if key in interesting or key == "dc.title":
                continue
            value = entries[0].get("value", "") if entries else ""
            if len(str(value)) > 120:
                print(f"      LONG {key:<28} {str(value)[:140]}...")


def route_b_sections_of_act() -> None:
    print("\n=== ROUTE B2: can we list every section of one act? ===")
    # Sections of the central Contract Act should share its act id.
    for field in ("dc.identifier.refact", "dc.identifier.act_id"):
        params = urllib.parse.urlencode(
            {"query": f'{field}:"{CENTRAL_CONTRACT_ACT}"', "size": 5}
        )
        status, body = get(f"{API}/discover/search/objects?{params}")
        if status != 200:
            print(f"  {field}: HTTP {status}")
            continue
        payload = json.loads(body)
        total = (
            payload.get("_embedded", {})
            .get("searchResult", {})
            .get("page", {})
            .get("totalElements")
        )
        print(f"  {field:<28} totalElements={total}")


if __name__ == "__main__":
    route_a()
    route_b()
    route_b_sections_of_act()
