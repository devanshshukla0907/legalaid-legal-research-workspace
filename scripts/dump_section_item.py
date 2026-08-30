"""Dump every metadata field on one section item, to find the section number and text fields."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

API = "https://indiacode.gov.in/server/api"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
ACT_ID = "AC_CEN_3_20_00035_187209_1523268996428"  # central Indian Contract Act


def get(url: str) -> dict:
    request = urllib.request.Request(url)
    request.add_header("User-Agent", UA)
    request.add_header("Accept", "application/json")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def main() -> None:
    params = urllib.parse.urlencode(
        {
            "query": f'dc.identifier.act_id:"{ACT_ID}"',
            "size": 6,
            "sort": "dc.title,ASC",
        }
    )
    payload = get(f"{API}/discover/search/objects?{params}")
    result = payload.get("_embedded", {}).get("searchResult", {})
    print(f"total items for this act: {result.get('page', {}).get('totalElements')}\n")

    for obj in result.get("_embedded", {}).get("objects", [])[:6]:
        href = obj.get("_links", {}).get("indexableObject", {}).get("href")
        if not href:
            continue
        item = get(href)
        print("=" * 78)
        print(f"handle: {item.get('handle')}   name: {item.get('name', '')[:60]}")
        for key, entries in sorted((item.get("metadata") or {}).items()):
            for entry in entries:
                value = str(entry.get("value", ""))
                shown = value if len(value) <= 160 else value[:160] + f"...[{len(value)} chars]"
                print(f"  {key:<40} {shown}")
        print()


if __name__ == "__main__":
    main()
