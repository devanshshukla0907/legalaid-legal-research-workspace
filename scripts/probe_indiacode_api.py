"""Map the new India Code (DSpace 9.1) API well enough to decide if re-ingestion is viable.

Context: India Code migrated in August 2026 to indiacode.gov.in running DSpace 9.1 as an Angular
SPA. The old scraper depended on two things that no longer exist - server-rendered act pages
carrying a section index, and a custom /SectionPageContent JSON endpoint. Both are gone.

The question this script answers is narrow and decisive: **is section-level statutory text
reachable through the new API, or only whole-act PDFs?** Section-level text is what the corpus
is built on; page-level PDF text would be a significant downgrade in citation precision.
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


def get(path: str) -> tuple[int, dict | str]:
    url = path if path.startswith("http") else API + path
    request = urllib.request.Request(url)
    request.add_header("User-Agent", UA)
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8", "replace")
            try:
                return response.status, json.loads(raw)
            except json.JSONDecodeError:
                return response.status, raw[:400]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(300).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def search(query: str, size: int = 5) -> list[dict]:
    params = urllib.parse.urlencode({"query": query, "size": size})
    status, payload = get(f"/discover/search/objects?{params}")
    if status != 200 or not isinstance(payload, dict):
        print(f"  search failed: {status} {payload}")
        return []
    objects = (
        payload.get("_embedded", {})
        .get("searchResult", {})
        .get("_embedded", {})
        .get("objects", [])
    )
    out = []
    for obj in objects:
        embedded = obj.get("_embedded", {}).get("indexableObject", {})
        if embedded:
            out.append(embedded)
        else:
            href = obj.get("_links", {}).get("indexableObject", {}).get("href")
            if href:
                status, item = get(href)
                if status == 200 and isinstance(item, dict):
                    out.append(item)
    return out


def metadata_value(item: dict, key: str) -> str:
    entries = (item.get("metadata") or {}).get(key) or []
    return entries[0].get("value", "") if entries else ""


def main() -> None:
    print("=== 1. Can we find the Indian Contract Act? ===")
    hits = search("Indian Contract Act 1872", size=5)
    for item in hits:
        print(
            f"  {item.get('type','?'):<12} {item.get('handle','-'):<22} "
            f"{metadata_value(item, 'dc.title')[:56]}"
        )

    act = next(
        (i for i in hits if "contract act" in metadata_value(i, "dc.title").lower()),
        hits[0] if hits else None,
    )
    if not act:
        print("  no candidate item found; stopping")
        return

    uuid = act.get("uuid")
    print(f"\n=== 2. Item detail for {uuid} ===")
    print(f"  handle : {act.get('handle')}")
    print(f"  name   : {act.get('name')}")
    print("  metadata keys:")
    for key, entries in sorted((act.get("metadata") or {}).items()):
        value = entries[0].get("value", "") if entries else ""
        print(f"    {key:<34} {str(value)[:56]}")

    print("\n=== 3. Bundles and bitstreams (is there anything but a PDF?) ===")
    status, bundles = get(f"/core/items/{uuid}/bundles")
    if status == 200 and isinstance(bundles, dict):
        for bundle in bundles.get("_embedded", {}).get("bundles", []):
            print(f"  bundle: {bundle.get('name')}")
            href = bundle.get("_links", {}).get("bitstreams", {}).get("href")
            if not href:
                continue
            bstatus, bits = get(href)
            if bstatus == 200 and isinstance(bits, dict):
                for bit in bits.get("_embedded", {}).get("bitstreams", []):
                    size = bit.get("sizeBytes", 0)
                    fmt = metadata_value(bit, "dc.format") or bit.get("name", "")
                    print(f"    - {bit.get('name','?')[:48]:<50} {size/1024:>8.0f} KB  {fmt}")
    else:
        print(f"  bundles: {status} {str(bundles)[:160]}")

    print("\n=== 4. Is there a child structure holding individual sections? ===")
    for probe in (
        f"/core/items/{uuid}/relationships",
        f"/core/items/{uuid}/mappedCollections",
        f"/discover/search/objects?query=%22Indian+Contract+Act%22+AND+Section&size=5",
    ):
        status, payload = get(probe)
        label = probe.split("?")[0].split("/")[-1]
        if isinstance(payload, dict):
            total = payload.get("page", {}).get("totalElements")
            print(f"  {label:<20} {status}  totalElements={total}")
        else:
            print(f"  {label:<20} {status}  {str(payload)[:80]}")

    print("\n=== 5. Do the old section endpoints still exist anywhere? ===")
    for probe in (
        f"{HOST}/SectionPageContent?actid=AC_CEN_3_20_00035_187209_1523268996428&sectionID=1",
        f"{HOST}/show-data?actid=AC_CEN_3_20_00035_187209_1523268996428&sectionId=1",
        f"{HOST}/server/api/sections?actid=1",
    ):
        status, payload = get(probe)
        print(f"  {status}  {probe.split('/')[-1][:60]}")


if __name__ == "__main__":
    main()
